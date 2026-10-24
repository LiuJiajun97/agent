"""LangGraph 原生版测试：不调用真实 API，不需要 API Key。"""

from __future__ import annotations

import json
import unittest
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import agent.tools
import agent_lg


class FakeToolCallingModel(BaseChatModel):
    """按预设顺序返回消息的假模型，不发起任何网络请求。"""

    responses: list[AIMessage] = Field(default_factory=list)
    tool_calls_log: list[Any] = Field(default_factory=list)
    # 假模型默认也会逐字输出，模拟真实模型的流式行为。
    stream_chunks: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        self.tool_calls_log.append(list(tools))
        return self

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.stream_chunks:
            for piece in self.stream_chunks:
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            return

        # 没有预设流式片段时，把预设回复整段输出。
        message = self.responses.pop(0) if self.responses else AIMessage(content="")
        if message.tool_calls:
            # 流式下的工具调用要用 tool_call_chunks 表达。
            call = message.tool_calls[0]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"], ensure_ascii=False),
                            "id": call["id"],
                            "index": 0,
                        }
                    ],
                )
            )
            return
        yield ChatGenerationChunk(message=AIMessageChunk(content=message.content))


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """构造一条带工具调用的模型回复。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


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
        model = FakeToolCallingModel(
            responses=[AIMessage(content="第一轮"), AIMessage(content="第二轮")]
        )
        graph = agent_lg.build_graph(model=model)

        agent_lg.run_turn(graph, "第一问", thread_id="lg0")
        agent_lg.run_turn(graph, "第二问", thread_id="lg0")

        state = graph.get_state({"configurable": {"thread_id": "lg0"}})
        messages = state.values.get("messages", [])
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "第一问")
        self.assertEqual(messages[2].content, "第二问")

    def test_tool_names_match_handwritten_version(self) -> None:
        """三套实现的工具集必须等价，否则对照实验不成立。"""
        handwritten = [t["name"] for t in agent.tools.TOOLS]
        self.assertEqual([t.name for t in agent_lg.build_tools()], handwritten)

    def test_build_model_uses_responses_api(self) -> None:
        """框架版必须走 Responses API，才能和手写版对照同一端点。"""
        model = agent_lg.build_model()
        self.assertTrue(getattr(model, "use_responses_api", False))
        self.assertEqual(getattr(model, "output_version", None), "responses/v1")


class EndToEndTests(unittest.TestCase):
    """B. 假模型端到端。"""

    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        model = FakeToolCallingModel(
            responses=[
                tool_call_message("calculate", {"expression": "(128 * 37 + 9) / 5"}),
                AIMessage(content="计算结果是 949"),
            ]
        )
        graph = agent_lg.build_graph(model=model)

        answer = agent_lg.run_turn(graph, "请计算", thread_id="lg1")

        self.assertEqual(answer, "计算结果是 949")
        self.assertTrue(model.tool_calls_log)

    def test_tool_result_is_json_and_matches_handwritten_format(self) -> None:
        """工具结果回传给模型时是 JSON 文本，与手写版 json.dumps 保持同构。"""
        model = FakeToolCallingModel(
            responses=[
                tool_call_message("calculate", {"expression": "(128 * 37 + 9) / 5"}),
                AIMessage(content="done"),
            ]
        )
        graph = agent_lg.build_graph(model=model)

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "请计算"}]},
            config={"configurable": {"thread_id": "lg-json"}},
        )
        tool_messages = [m for m in result["messages"] if m.type == "tool"]

        self.assertEqual(len(tool_messages), 1)
        self.assertIn('"ok": true', tool_messages[0].content)
        self.assertIn("949.0", tool_messages[0].content)

    def test_tool_error_is_returned_to_model(self) -> None:
        """工具出错时错误要回传给模型，循环继续，而不是直接崩掉。"""
        model = FakeToolCallingModel(
            responses=[
                tool_call_message("calculate", {"expression": "a + 1"}),
                AIMessage(content="表达式不合法"),
            ]
        )
        graph = agent_lg.build_graph(model=model)

        answer = agent_lg.run_turn(graph, "请计算", thread_id="lg2")

        self.assertEqual(answer, "表达式不合法")

    def test_system_prompt_is_not_stored_in_state(self) -> None:
        """系统提示词只在调用模型时拼进去，不应污染对话状态。"""
        model = FakeToolCallingModel(responses=[AIMessage(content="ok")])
        graph = agent_lg.build_graph(model=model)

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config={"configurable": {"thread_id": "lg-sys"}},
        )
        roles = [m.type for m in result["messages"]]

        self.assertNotIn("system", roles)

    def test_multi_turn_uses_checkpointer(self) -> None:
        """第二轮提问时，历史应当已被 checkpointer 保留。"""
        model = FakeToolCallingModel(
            responses=[AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答")]
        )
        checkpointer = agent_lg.build_checkpointer()
        graph = agent_lg.build_graph(model=model, checkpointer=checkpointer)

        first = agent_lg.run_turn(graph, "第一问", thread_id="lg3")
        second = agent_lg.run_turn(graph, "第二问", thread_id="lg3")

        self.assertEqual(first, "第一轮回答")
        self.assertEqual(second, "第二轮回答")

    def test_clear_thread_resets_history(self) -> None:
        checkpointer = agent_lg.build_checkpointer()
        agent_lg.clear_thread(checkpointer, "lg4")
        graph = agent_lg.build_graph(
            model=FakeToolCallingModel(responses=[AIMessage(content="ok")]),
            checkpointer=checkpointer,
        )
        answer = agent_lg.run_turn(graph, "你好", thread_id="lg4")
        self.assertEqual(answer, "ok")


class StreamingTests(unittest.TestCase):
    """C. 流式输出。"""

    def test_stream_yields_text_incrementally(self) -> None:
        model = FakeToolCallingModel(
            responses=[AIMessage(content="你好，世界")],
            stream_chunks=["你", "好", "，", "世界"],
        )
        graph = agent_lg.build_graph(model=model)

        chunks = list(agent_lg.run_turn_stream(graph, "你好", thread_id="lg-s1"))

        self.assertEqual(chunks, ["你", "好", "，", "世", "界"])

    def test_stream_does_not_leak_tool_output(self) -> None:
        """工具结果不应混进流式回答里。"""
        model = FakeToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate",
                            "args": {"expression": "(128 * 37 + 9) / 5"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="结果是 949"),
            ]
        )
        graph = agent_lg.build_graph(model=model)

        chunks = list(agent_lg.run_turn_stream(graph, "请计算", thread_id="lg-s2"))

        self.assertEqual("".join(chunks), "结果是 949")
        self.assertNotIn("ok", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
