"""手写版测试：不调用真实 API，不需要 API Key。

覆盖三部分：
- 工具实现（与 API 无关）
- 规范化历史 + 循环（注入 Turn）
- 两个 API 后端的端到端封装（Responses / Chat Completions）
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

import agent


def function_call_item(
    call_id: str = "call_test",
    name: str = "calculate",
    arguments: str = '{"expression": "(128 * 37 + 9) / 5"}',
) -> SimpleNamespace:
    """模拟 Responses API 的 function_call 输出项。"""
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def responses_response(
    output: list | None = None,
    output_text: str = "",
    status: str = "completed",
    error: Any = None,
) -> SimpleNamespace:
    """模拟 Responses API 的 response 对象。"""
    return SimpleNamespace(
        output=output or [],
        output_text=output_text,
        status=status,
        error=error,
    )


def responses_event(kind: str, **kwargs: Any) -> SimpleNamespace:
    """模拟 Responses API 的流式事件。"""
    return SimpleNamespace(type=kind, **kwargs)


def chat_tool_call(
    call_id: str = "call_1",
    name: str = "calculate",
    arguments: str = '{"expression": "(128 * 37 + 9) / 5"}',
) -> SimpleNamespace:
    """模拟 Chat Completions 的 tool_call。"""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def chat_response(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """模拟 Chat Completions 的非流式响应。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def chat_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """模拟 Chat Completions 的流式文本分片。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeResponses:
    """模拟 client.responses。"""

    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses[len(self.calls) - 1]
        return iter(response) if kwargs.get("stream") else response


class FakeResponsesClient:
    def __init__(self, responses: list) -> None:
        self.responses = FakeResponses(responses)


class FakeChatCompletions:
    """模拟 client.chat.completions。"""

    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses[len(self.calls) - 1]
        return iter(response) if kwargs.get("stream") else response


class FakeChatClient:
    def __init__(self, responses: list) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(responses))


class ToolTests(unittest.TestCase):
    def test_calculate(self) -> None:
        self.assertEqual(agent.calculate("(128 * 37 + 9) / 5"), 949.0)
        self.assertEqual(agent.calculate("17 ** 3"), 4913)

    def test_calculate_rejects_unknown_syntax(self) -> None:
        with self.assertRaises(ValueError):
            agent.calculate("a + 1")

    def test_execute_unknown_tool(self) -> None:
        result = agent.execute_tool("missing", {})
        self.assertFalse(result["ok"])
        self.assertIn("未知工具", result["error"])

    def test_execute_tool_wraps_errors(self) -> None:
        result = agent.execute_tool("calculate", {"expression": "1 / 0"})
        self.assertFalse(result["ok"])
        self.assertIn("ZeroDivisionError", result["error"])

    def test_current_time_shape(self) -> None:
        result = agent.execute_tool("get_current_time", {})
        self.assertTrue(result["ok"])
        self.assertIn("local_time", result)
        self.assertIn("utc_offset_seconds", result)


class AgentTurnTests(unittest.TestCase):
    """A. run_agent_turn：规范化历史 + 两种 API 后端。"""

    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        call = function_call_item()
        client = FakeResponsesClient(
            [
                responses_response([call], ""),
                responses_response([], "计算结果是 949"),
            ]
        )
        history: list[Any] = []

        answer = agent.run_agent_turn(client, history, "请计算", style="responses")

        self.assertEqual(answer, "计算结果是 949")
        self.assertEqual(len(client.responses.calls), 2)

        # 历史是规范格式：assistant 带 tool_calls，工具结果单独一条。
        self.assertEqual(history[0], {"role": "user", "content": "请计算"})
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["tool_calls"][0]["id"], "call_test")
        self.assertEqual(history[2]["role"], "tool")
        self.assertEqual(history[2]["tool_call_id"], "call_test")
        self.assertEqual(json.loads(history[2]["content"])["result"], 949.0)

        # 第二次请求把工具调用和结果都回传给了模型。
        second_input = client.responses.calls[1]["input"]
        self.assertIn(
            {
                "type": "function_call",
                "call_id": "call_test",
                "name": "calculate",
                "arguments": '{"expression": "(128 * 37 + 9) / 5"}',
            },
            second_input,
        )
        self.assertIn(
            {
                "type": "function_call_output",
                "call_id": "call_test",
                "output": history[2]["content"],
            },
            second_input,
        )

    def test_history_is_reused_across_turns(self) -> None:
        client = FakeResponsesClient(
            [
                responses_response([], "第一轮回答"),
                responses_response([], "第二轮回答"),
            ]
        )
        history: list[Any] = []

        first = agent.run_agent_turn(client, history, "第一问", style="responses")
        second = agent.run_agent_turn(client, history, "第二问", style="responses")

        self.assertEqual(first, "第一轮回答")
        self.assertEqual(second, "第二轮回答")

        second_input = client.responses.calls[1]["input"]
        self.assertIn({"role": "user", "content": "第一问"}, second_input)
        self.assertIn({"role": "assistant", "content": "第一轮回答"}, second_input)
        self.assertIn({"role": "user", "content": "第二问"}, second_input)

    def test_malformed_tool_arguments_are_sent_back_to_model(self) -> None:
        """参数不是合法 JSON 时，错误要作为工具结果回传，而不是直接崩掉。"""
        bad_call = function_call_item(arguments="not json")
        client = FakeResponsesClient(
            [
                responses_response([bad_call], ""),
                responses_response([], "参数格式不对"),
            ]
        )
        history: list[Any] = []

        answer = agent.run_agent_turn(client, history, "算一下", style="responses")

        self.assertEqual(answer, "参数格式不对")
        tool_message = [m for m in history if m["role"] == "tool"][0]
        payload = json.loads(tool_message["content"])
        self.assertFalse(payload["ok"])
        self.assertIn("参数解析失败", payload["error"])

    def test_chat_style_end_to_end(self) -> None:
        """同一个循环换成 Chat Completions 后端，行为保持一致。"""
        client = FakeChatClient(
            [
                chat_response(tool_calls=[chat_tool_call()]),
                chat_response(content="计算结果是 949"),
            ]
        )
        history: list[Any] = []

        answer = agent.run_agent_turn(client, history, "请计算", style="chat")

        self.assertEqual(answer, "计算结果是 949")

        second_messages = client.chat.completions.calls[1]["messages"]
        self.assertEqual(second_messages[0]["role"], "system")
        self.assertEqual(second_messages[1], {"role": "user", "content": "请计算"})
        self.assertTrue(
            any(
                m.get("role") == "assistant" and m.get("tool_calls")
                for m in second_messages
            )
        )
        self.assertTrue(
            any(
                m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
                for m in second_messages
            )
        )

    def test_failed_turn_rolls_back_history(self) -> None:
        failed = responses_response(
            status="failed",
            error=SimpleNamespace(code="server_error", message="上游挂了"),
        )
        client = FakeResponsesClient([failed])
        history: list[Any] = []

        with self.assertRaises(RuntimeError):
            agent.run_agent_turn(client, history, "你好", style="responses")

        self.assertEqual(history, [])


class CoreLoopTests(unittest.TestCase):
    """B. agent_loop：只依赖注入的 request / run_tool_call。"""

    def test_agent_loop_with_injected_dependencies(self) -> None:
        turns = [
            agent.Turn(tool_calls=[agent.ToolCall("c1", "custom", '{"x": "a"}')]),
            agent.Turn("完成"),
        ]

        def request() -> agent.Turn:
            return turns.pop(0)

        def run_tool_call(call: agent.ToolCall) -> dict[str, Any]:
            return {"ok": True, "result": 1}

        history: list[Any] = []
        answer = agent.agent_loop(request, run_tool_call, history)

        self.assertEqual(answer, "完成")
        self.assertEqual(history[0]["role"], "assistant")
        self.assertEqual(history[0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(history[1]["role"], "tool")
        self.assertEqual(history[1]["tool_call_id"], "c1")

    def test_agent_loop_returns_text_without_tools(self) -> None:
        history: list[Any] = []

        answer = agent.agent_loop(
            lambda: agent.Turn("直接回答"), lambda call: {}, history
        )

        self.assertEqual(answer, "直接回答")
        self.assertEqual(history, [{"role": "assistant", "content": "直接回答"}])

    def test_agent_loop_raises_after_max_steps(self) -> None:
        def request() -> agent.Turn:
            return agent.Turn(tool_calls=[agent.ToolCall("x", "t", "{}")])

        with self.assertRaises(RuntimeError):
            agent.agent_loop(request, lambda call: {"ok": True}, [], max_steps=2)


class StreamingLoopTests(unittest.TestCase):
    """C. agent_loop_stream：消费规范化流式事件。"""

    def test_stream_yields_text_incrementally(self) -> None:
        events = [
            agent.TextDelta("你"),
            agent.TextDelta("好"),
            agent.Completed(agent.Turn("你好")),
        ]
        history: list[Any] = []

        chunks = list(
            agent.agent_loop_stream(
                lambda: iter(events),
                lambda call: {},
                history,
            )
        )

        self.assertEqual(chunks, ["你", "好"])
        self.assertEqual(history, [{"role": "assistant", "content": "你好"}])

    def test_stream_executes_tool_then_continues(self) -> None:
        rounds = [
            [
                agent.Completed(
                    agent.Turn(
                        tool_calls=[
                            agent.ToolCall("c1", "calculate", '{"expression": "1 + 1"}')
                        ]
                    )
                )
            ],
            [
                agent.TextDelta("结果是 2"),
                agent.Completed(agent.Turn("结果是 2")),
            ],
        ]
        round_idx = [0]

        def request_stream() -> Any:
            events = rounds[round_idx[0]]
            round_idx[0] += 1
            return iter(events)

        history: list[Any] = []
        chunks = list(
            agent.agent_loop_stream(
                request_stream, lambda call: {"ok": True, "result": 2}, history
            )
        )

        self.assertEqual("".join(chunks), "结果是 2")
        self.assertEqual(history[0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(history[1]["role"], "tool")
        self.assertEqual(history[1]["tool_call_id"], "c1")

    def test_stream_raises_on_failure_event(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            list(
                agent.agent_loop_stream(
                    lambda: iter([agent.Failed("限流了")]), lambda call: {}, []
                )
            )

        self.assertIn("限流了", str(ctx.exception))

    def test_stream_raises_when_completed_event_missing(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            list(
                agent.agent_loop_stream(
                    lambda: iter([agent.TextDelta("只有文字")]), lambda call: {}, []
                )
            )

        self.assertIn("完成事件", str(ctx.exception))

    def test_stream_raises_after_max_steps(self) -> None:
        def request_stream() -> Any:
            return iter(
                [
                    agent.Completed(
                        agent.Turn(tool_calls=[agent.ToolCall("x", "t", "{}")])
                    )
                ]
            )

        with self.assertRaises(RuntimeError):
            list(agent.agent_loop_stream(request_stream, lambda call: {}, [], max_steps=2))

    def test_chat_style_stream_end_to_end(self) -> None:
        client = FakeChatClient(
            [[chat_chunk(content="结果"), chat_chunk(content="是 2", finish_reason="stop")]]
        )
        history: list[Any] = []

        text = "".join(
            agent.run_agent_turn_stream(client, history, "1+1", style="chat")
        )

        self.assertEqual(text, "结果是 2")
        self.assertEqual(history[0], {"role": "user", "content": "1+1"})
        self.assertEqual(history[1], {"role": "assistant", "content": "结果是 2"})

    def test_failed_stream_rolls_back_history(self) -> None:
        failed = responses_response(
            status="failed",
            error=SimpleNamespace(code="server_error", message="上游挂了"),
        )
        events = [
            responses_event("response.output_text.delta", delta="开头"),
            responses_event("response.failed", response=failed),
        ]
        client = FakeResponsesClient([events])
        history: list[Any] = []

        with self.assertRaises(RuntimeError):
            list(agent.run_agent_turn_stream(client, history, "你好", style="responses"))

        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
