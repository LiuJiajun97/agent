"""LangChain 版测试：不调用真实 API，不需要 API Key。"""

from __future__ import annotations

import json
import unittest
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import agent.tools
import agent_lc


class FakeToolCallingModel(BaseChatModel):
    """按预设顺序返回消息的假模型，不发起任何网络请求。

    模型每次被调用就消耗一条预设消息；
    bind_tools 记录模型拿到的工具清单，用来验证工具确实被绑定。
    """

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


class GraphTests(unittest.TestCase):
    """B. 图构建。"""

    def test_build_agent_returns_compiled_graph(self) -> None:
        agent = agent_lc.build_agent()
        self.assertTrue(hasattr(agent, "invoke"))
        self.assertTrue(hasattr(agent, "get_graph"))

    def test_build_agent_accepts_injected_model(self) -> None:
        model = FakeToolCallingModel(responses=[AIMessage(content="hi")])
        agent = agent_lc.build_agent(model=model)
        self.assertTrue(hasattr(agent, "invoke"))


class EndToEndTests(unittest.TestCase):
    """C. 假模型端到端。"""

    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "(128 * 37 + 9) / 5"},
                    "id": "call_1",
                }
            ],
        )
        model = FakeToolCallingModel(responses=[tool_call, AIMessage(content="计算结果是 949")])
        agent = agent_lc.build_agent(model=model)

        answer = agent_lc.run_turn(agent, "请计算", thread_id="t1")

        self.assertEqual(answer, "计算结果是 949")
        # 模型确实拿到了工具清单。
        self.assertTrue(model.tool_calls_log)

    def test_tool_error_is_returned_to_model(self) -> None:
        """工具出错时错误要回传给模型，循环继续，而不是直接崩掉。"""
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "a + 1"},
                    "id": "call_1",
                }
            ],
        )
        model = FakeToolCallingModel(responses=[tool_call, AIMessage(content="表达式不合法")])
        agent = agent_lc.build_agent(model=model)

        answer = agent_lc.run_turn(agent, "请计算", thread_id="t2")

        self.assertEqual(answer, "表达式不合法")

    def test_multi_turn_uses_checkpointer(self) -> None:
        """第二轮提问时，历史应当已被 checkpointer 保留。"""
        model = FakeToolCallingModel(responses=[AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答")])
        agent = agent_lc.build_agent(model=model)

        first = agent_lc.run_turn(agent, "第一问", thread_id="t3")
        second = agent_lc.run_turn(agent, "第二问", thread_id="t3")

        self.assertEqual(first, "第一轮回答")
        self.assertEqual(second, "第二轮回答")

    def test_clear_thread_resets_history(self) -> None:
        checkpointer = agent_lc.build_checkpointer()
        agent_lc.clear_thread(checkpointer, "t4")
        # 清空后不应抛异常，且线程可以继续使用。
        agent = agent_lc.build_agent(
            model=FakeToolCallingModel(responses=[AIMessage(content="ok")]),
            checkpointer=checkpointer,
        )
        answer = agent_lc.run_turn(agent, "你好", thread_id="t4")
        self.assertEqual(answer, "ok")


class StreamingTests(unittest.TestCase):
    """D. 流式输出。"""

    def test_stream_yields_text_incrementally(self) -> None:
        model = FakeToolCallingModel(
            responses=[AIMessage(content="你好，世界")],
            stream_chunks=["你", "好", "，", "世界"],
        )
        agent = agent_lc.build_agent(model=model)

        chunks = list(agent_lc.run_turn_stream(agent, "你好", thread_id="lc-s1"))

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
        agent = agent_lc.build_agent(model=model)

        chunks = list(agent_lc.run_turn_stream(agent, "请计算", thread_id="lc-s4"))

        self.assertEqual("".join(chunks), "结果是 949")
        self.assertNotIn("ok", "".join(chunks))

    def test_stream_skips_empty_trailing_chunk(self) -> None:
        """收尾的空内容块不应产出空字符串。"""
        model = FakeToolCallingModel(
            responses=[AIMessage(content="好")],
            stream_chunks=["好"],
        )
        agent = agent_lc.build_agent(model=model)

        chunks = list(agent_lc.run_turn_stream(agent, "你好", thread_id="lc-s3"))

        self.assertTrue(all(chunks))
        self.assertEqual("".join(chunks), "好")


if __name__ == "__main__":
    unittest.main()
