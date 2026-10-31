"""LangChain 版测试。

覆盖四部分：
- 工具层与 message_text 的 content 形态适配（纯逻辑）
- 图构建与 build_model 的默认参数（纯逻辑）
- 端到端对话（真实调用网关，需要 .env 里配好 OPENAI_API_KEY / OPENAI_BASE_URL）
- 工具出错时的回传（真实调用 + 临时替换工具实现）
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from langchain_core.messages import AIMessage

import agent.tools
import agent_lc
from tests.live import require_gateway


class ToolTests(unittest.TestCase):
    """A. 工具层。"""

    def test_tool_names_match_handwritten_version(self) -> None:
        """两套实现的工具集必须等价，否则对照实验不成立。"""
        handwritten = [t["name"] for t in agent.tools.TOOLS]
        self.assertEqual([t.name for t in agent_lc.build_tools()], handwritten)
    def test_calculate_tool_metadata(self) -> None:
        tool = agent_lc.calculate
        self.assertEqual(tool.name, "calculate")
        self.assertIn("数学表达式", tool.description)
        self.assertEqual(tool.args["expression"]["type"], "string")

    def test_calculate_tool_returns_ok_result(self) -> None:
        result = agent_lc.calculate.invoke({"expression": "(128 * 37 + 9) / 5"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], 949.0)

    def test_calculate_tool_wraps_error(self) -> None:
        result = agent_lc.calculate.invoke({"expression": "a + 1"})
        self.assertFalse(result["ok"])
        self.assertIn("ValueError", result["error"])

    def test_current_time_tool_shape(self) -> None:
        result = agent_lc.get_current_time.invoke({})
        self.assertTrue(result["ok"])
        self.assertIn("local_time", result)
        self.assertIn("utc_offset_seconds", result)


class MessageTextTests(unittest.TestCase):
    """A2. Responses API 的 content 块列表适配。"""

    def test_plain_string_content(self) -> None:
        """Chat Completions 风格的字符串 content。"""
        msg = AIMessage(content="你好")
        self.assertEqual(agent_lc.message_text(msg), "你好")

    def test_responses_block_list_content(self) -> None:
        """Responses API 风格的块列表 content，要拼接其中的 text 块。"""
        msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [], "id": "rs_1"},
                {"type": "text", "text": "答案是 ", "index": 1},
                {"type": "text", "text": "949", "index": 2},
            ]
        )
        self.assertEqual(agent_lc.message_text(msg), "答案是 949")

    def test_block_list_skips_non_text_blocks(self) -> None:
        """reasoning / tool_call 等非 text 块不能混进回答。"""
        msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "思考中"}]},
                {"type": "text", "text": "只有这句"},
            ]
        )
        text = agent_lc.message_text(msg)
        self.assertEqual(text, "只有这句")
        self.assertNotIn("思考中", text)

    def test_empty_content(self) -> None:
        self.assertEqual(agent_lc.message_text(AIMessage(content="")), "")


class GraphTests(unittest.TestCase):
    """B. 图构建。"""

    def test_build_agent_returns_compiled_graph(self) -> None:
        agent = agent_lc.build_agent()
        self.assertTrue(hasattr(agent, "invoke"))
        self.assertTrue(hasattr(agent, "get_graph"))

    def test_build_agent_accepts_injected_model(self) -> None:
        """可以注入自己的模型（这里是真实模型），建出来的图能正常对话。"""
        require_gateway()
        agent = agent_lc.build_agent(model=agent_lc.build_model())

        answer = agent_lc.run_turn(agent, "用一句话说明什么是哈希表", thread_id="t1")

        self.assertTrue(answer.strip())

    def test_build_model_defaults_to_responses_api(self) -> None:
        """默认走 Responses API，才能和手写版对照同一端点。"""
        model = agent_lc.build_model()
        self.assertTrue(getattr(model, "use_responses_api", False))
        self.assertEqual(getattr(model, "output_version", None), "responses/v1")

    def test_build_model_can_use_chat_completions(self) -> None:
        """切到 chat 时不再走 Responses，也不设置 Responses 专属的 output_version。"""
        model = agent_lc.build_model(api_style="chat")
        self.assertFalse(getattr(model, "use_responses_api", True))
        self.assertNotEqual(getattr(model, "output_version", None), "responses/v1")


class EndToEndTests(unittest.TestCase):
    """C. 真实网关端到端。"""

    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        """真实调用：模型发起 calculate，工具结果作为 ToolMessage 回传。"""
        require_gateway()
        agent = agent_lc.build_agent()

        answer = agent_lc.run_turn(
            agent, "请使用 calculate 工具计算 123 * 45", thread_id="t2"
        )

        self.assertTrue(answer.strip())

        state = agent.get_state({"configurable": {"thread_id": "t2"}})
        tool_messages = [m for m in state.values.get("messages", []) if m.type == "tool"]
        self.assertEqual(len(tool_messages), 1)

        payload = json.loads(tool_messages[0].content)
        self.assertTrue(payload["ok"])
        self.assertIsInstance(payload["result"], (int, float))

    def test_tool_error_is_returned_to_model(self) -> None:
        """工具出错时错误要回传给模型，循环继续，而不是直接崩掉。"""
        require_gateway()
        agent = agent_lc.build_agent()

        # 让真实工具失败：错误要经过工具层的包装，再回传给模型。
        with mock.patch.object(
            agent_lc.tools, "_calculate", side_effect=ValueError("表达式不合法")
        ):
            answer = agent_lc.run_turn(
                agent, "请使用 calculate 工具计算 123 * 45", thread_id="t3"
            )

        self.assertTrue(answer.strip())

        state = agent.get_state({"configurable": {"thread_id": "t3"}})
        tool_messages = [m for m in state.values.get("messages", []) if m.type == "tool"]
        self.assertGreaterEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0].content)
        self.assertFalse(payload["ok"])
        self.assertIn("ValueError", payload["error"])

    def test_multi_turn_uses_checkpointer(self) -> None:
        """第二轮提问时，历史应当已被 checkpointer 保留。"""
        require_gateway()
        agent = agent_lc.build_agent()

        first = agent_lc.run_turn(agent, "用一句话说明什么是质数", thread_id="t4")
        second = agent_lc.run_turn(agent, "用一句话说明什么是合数", thread_id="t4")

        self.assertTrue(first.strip())
        self.assertTrue(second.strip())

        state = agent.get_state({"configurable": {"thread_id": "t4"}})
        messages = state.values.get("messages", [])
        users = [m for m in messages if m.type == "human"]
        self.assertEqual(len(users), 2)
        self.assertEqual(users[0].content, "用一句话说明什么是质数")
        self.assertEqual(users[1].content, "用一句话说明什么是合数")

    def test_clear_thread_resets_history(self) -> None:
        """清空线程后历史确实少了，但线程还能继续用。"""
        require_gateway()
        checkpointer = agent_lc.build_checkpointer()
        agent = agent_lc.build_agent(checkpointer=checkpointer)

        agent_lc.run_turn(agent, "用一句话说明什么是质数", thread_id="t5")
        before = len(
            agent.get_state({"configurable": {"thread_id": "t5"}})
            .values.get("messages", [])
        )

        agent_lc.clear_thread(checkpointer, "t5")
        after = len(
            agent.get_state({"configurable": {"thread_id": "t5"}})
            .values.get("messages", [])
        )

        self.assertLess(after, before)

        # 清空后线程可以继续使用。
        answer = agent_lc.run_turn(agent, "用一句话说明什么是合数", thread_id="t5")
        self.assertTrue(answer.strip())


class StreamingTests(unittest.TestCase):
    """D. 流式输出。"""

    def test_stream_yields_text_incrementally(self) -> None:
        """真实流式：分片拼起来就是完整回答。"""
        require_gateway()
        agent = agent_lc.build_agent()

        chunks = list(
            agent_lc.run_turn_stream(
                agent, "用一句话说明什么是冒泡排序", thread_id="lc-s1"
            )
        )

        self.assertTrue(chunks)
        self.assertTrue("".join(chunks).strip())

    def test_stream_does_not_leak_tool_output(self) -> None:
        """工具结果不应混进流式回答里。"""
        require_gateway()
        agent = agent_lc.build_agent()

        chunks = list(
            agent_lc.run_turn_stream(
                agent, "请使用 calculate 工具计算 123 * 45", thread_id="lc-s2"
            )
        )

        text = "".join(chunks)
        self.assertTrue(text.strip())
        # 工具结果是 JSON，不该出现在给用户看的回答里。
        self.assertNotIn('"ok"', text)
        self.assertNotIn('"result"', text)

    def test_stream_skips_empty_trailing_chunk(self) -> None:
        """收尾的空内容块不应产出空字符串。"""
        require_gateway()
        agent = agent_lc.build_agent()

        chunks = list(
            agent_lc.run_turn_stream(
                agent, "用一句话说明什么是哈希表", thread_id="lc-s3"
            )
        )

        self.assertTrue(all(chunks))
        self.assertTrue("".join(chunks).strip())


if __name__ == "__main__":
    unittest.main()
