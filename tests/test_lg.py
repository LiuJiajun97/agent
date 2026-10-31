"""LangGraph 原生版测试。

覆盖四部分：
- 图的结构与 build_model 默认参数（纯逻辑）
- 端到端对话（真实调用网关，需要 .env 里配好 OPENAI_API_KEY / OPENAI_BASE_URL）
- 状态与 checkpointer 行为（真实调用）
- 工具出错时的回传（真实调用 + 临时替换工具实现）
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import agent.tools
import agent_lc
import agent_lg
from tests.live import require_gateway


class GraphShapeTests(unittest.TestCase):
    """A. 图的结构：应与 create_agent 生成的一致。"""

    def test_graph_has_same_shape_as_create_agent(self) -> None:
        graph = agent_lg.build_graph()
        drawn = graph.get_graph()
        self.assertEqual(
            list(drawn.nodes.keys()), ["__start__", "model", "tools", "__end__"]
        )

        edges = {(e.source, e.target) for e in drawn.edges}
        self.assertIn(("__start__", "model"), edges)
        self.assertIn(("model", "tools"), edges)
        self.assertIn(("model", "__end__"), edges)
        self.assertIn(("tools", "model"), edges)

    def test_conditional_edges_are_marked(self) -> None:
        graph = agent_lg.build_graph()
        conditional = {(e.source, e.target) for e in graph.get_graph().edges if e.conditional}
        self.assertIn(("model", "tools"), conditional)
        self.assertIn(("model", "__end__"), conditional)

    def test_state_schema_has_messages(self) -> None:
        graph = agent_lg.build_graph()
        properties = graph.get_input_jsonschema().get("properties", {})
        self.assertIn("messages", properties)

    def test_build_graph_defaults_to_checkpointer(self) -> None:
        """缺省也必须挂上 checkpointer，否则多轮记忆会静默失效。"""
        graph = agent_lg.build_graph()
        self.assertIsNotNone(getattr(graph, "checkpointer", None))

    def test_history_accumulates_across_turns(self) -> None:
        """两轮对话后，状态里应同时保留两轮的消息。"""
        require_gateway()
        graph = agent_lg.build_graph()

        agent_lg.run_turn(graph, "用一句话说明什么是质数", thread_id="lg0")
        agent_lg.run_turn(graph, "用一句话说明什么是合数", thread_id="lg0")

        state = graph.get_state({"configurable": {"thread_id": "lg0"}})
        messages = state.values.get("messages", [])
        self.assertGreaterEqual(len(messages), 4)
        humans = [m for m in messages if m.type == "human"]
        self.assertEqual(len(humans), 2)
        self.assertEqual(humans[0].content, "用一句话说明什么是质数")
        self.assertEqual(humans[1].content, "用一句话说明什么是合数")

    def test_tool_names_match_handwritten_version(self) -> None:
        """手写版与 LangGraph 版的工具集必须等价，否则对照实验不成立。"""
        handwritten = [t["name"] for t in agent.tools.TOOLS]
        self.assertEqual([t.name for t in agent_lg.build_tools()], handwritten)

    def test_build_model_uses_responses_api(self) -> None:
        """框架版必须走 Responses API，才能和手写版对照同一端点。"""
        model = agent_lg.build_model()
        self.assertTrue(getattr(model, "use_responses_api", False))
        self.assertEqual(getattr(model, "output_version", None), "responses/v1")


class EndToEndTests(unittest.TestCase):
    """B. 真实网关端到端。"""

    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        """真实调用：模型发起 calculate，工具结果回传进状态。"""
        require_gateway()
        graph = agent_lg.build_graph()

        answer = agent_lg.run_turn(
            graph, "请使用 calculate 工具计算 123 * 45", thread_id="lg1"
        )

        self.assertTrue(answer.strip())

        state = graph.get_state({"configurable": {"thread_id": "lg1"}})
        tool_messages = [
            m for m in state.values.get("messages", []) if m.type == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)

        payload = json.loads(tool_messages[0].content)
        self.assertTrue(payload["ok"])
        self.assertIsInstance(payload["result"], (int, float))

    def test_tool_result_is_json_and_matches_handwritten_format(self) -> None:
        """工具结果回传给模型时是 JSON 文本，与手写版 json.dumps 保持同构。"""
        require_gateway()
        graph = agent_lg.build_graph()

        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "请使用 calculate 工具计算 123 * 45"}
                ]
            },
            config={"configurable": {"thread_id": "lg-json"}},
        )
        tool_messages = [m for m in result["messages"] if m.type == "tool"]

        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0].content)
        self.assertIs(payload["ok"], True)
        self.assertIsInstance(payload["result"], (int, float))

    def test_tool_error_is_returned_to_model(self) -> None:
        """工具出错时错误要回传给模型，循环继续，而不是直接崩掉。"""
        require_gateway()
        graph = agent_lg.build_graph()

        # 让真实工具失败：错误要经过工具层的包装，再回传给模型。
        with mock.patch.object(
            agent_lc.tools, "_calculate", side_effect=ValueError("表达式不合法")
        ):
            answer = agent_lg.run_turn(
                graph, "请使用 calculate 工具计算 123 * 45", thread_id="lg2"
            )

        self.assertTrue(answer.strip())

        state = graph.get_state({"configurable": {"thread_id": "lg2"}})
        tool_messages = [
            m for m in state.values.get("messages", []) if m.type == "tool"
        ]
        self.assertGreaterEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0].content)
        self.assertFalse(payload["ok"])
        self.assertIn("ValueError", payload["error"])

    def test_system_prompt_is_not_stored_in_state(self) -> None:
        """系统提示词只在调用模型时拼进去，不应污染对话状态。"""
        require_gateway()
        graph = agent_lg.build_graph()

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config={"configurable": {"thread_id": "lg-sys"}},
        )
        roles = [m.type for m in result["messages"]]

        self.assertNotIn("system", roles)

    def test_multi_turn_uses_checkpointer(self) -> None:
        """第二轮提问时，历史应当已被 checkpointer 保留。"""
        require_gateway()
        checkpointer = agent_lg.build_checkpointer()
        graph = agent_lg.build_graph(checkpointer=checkpointer)

        first = agent_lg.run_turn(graph, "用一句话说明什么是质数", thread_id="lg3")
        second = agent_lg.run_turn(graph, "用一句话说明什么是合数", thread_id="lg3")

        self.assertTrue(first.strip())
        self.assertTrue(second.strip())

        state = graph.get_state({"configurable": {"thread_id": "lg3"}})
        humans = [m for m in state.values.get("messages", []) if m.type == "human"]
        self.assertEqual(len(humans), 2)

    def test_clear_thread_resets_history(self) -> None:
        """清空线程后历史确实少了，但线程还能继续用。"""
        require_gateway()
        checkpointer = agent_lg.build_checkpointer()
        graph = agent_lg.build_graph(checkpointer=checkpointer)

        agent_lg.run_turn(graph, "用一句话说明什么是质数", thread_id="lg4")
        before = len(
            graph.get_state({"configurable": {"thread_id": "lg4"}})
            .values.get("messages", [])
        )

        agent_lg.clear_thread(checkpointer, "lg4")
        after = len(
            graph.get_state({"configurable": {"thread_id": "lg4"}})
            .values.get("messages", [])
        )

        self.assertLess(after, before)

        # 清空后线程可以继续使用。
        answer = agent_lg.run_turn(graph, "用一句话说明什么是合数", thread_id="lg4")
        self.assertTrue(answer.strip())


class StreamingTests(unittest.TestCase):
    """C. 流式输出。"""

    def test_stream_yields_text_incrementally(self) -> None:
        """真实流式：分片拼起来就是完整回答。"""
        require_gateway()
        graph = agent_lg.build_graph()

        chunks = list(
            agent_lg.run_turn_stream(
                graph, "用一句话说明什么是冒泡排序", thread_id="lg-s1"
            )
        )

        self.assertTrue(chunks)
        self.assertTrue("".join(chunks).strip())

    def test_stream_does_not_leak_tool_output(self) -> None:
        """工具结果不应混进流式回答里。"""
        require_gateway()
        graph = agent_lg.build_graph()

        chunks = list(
            agent_lg.run_turn_stream(
                graph, "请使用 calculate 工具计算 123 * 45", thread_id="lg-s2"
            )
        )

        text = "".join(chunks)
        self.assertTrue(text.strip())
        # 工具结果是 JSON，不该出现在给用户看的回答里。
        self.assertNotIn('"ok"', text)
        self.assertNotIn('"result"', text)


if __name__ == "__main__":
    unittest.main()
