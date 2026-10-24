"""API 适配层测试：不调用真实 API，不需要 API Key。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import agent
from agent.backends import (
    ChatCompletionsBackend,
    Completed,
    Failed,
    ResponsesBackend,
    TextDelta,
    build_backend,
    to_chat_messages,
    to_chat_tools,
    to_responses_input,
)


def function_call_item(
    call_id: str = "call_1",
    name: str = "calculate",
    arguments: str = '{"expression": "1 + 1"}',
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
    arguments: str = '{"expression": "1 + 1"}',
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
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """模拟 Chat Completions 的流式分片。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def chat_fragment(
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """模拟 Chat Completions 的 tool_call 流式分片。"""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeResponses:
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


SAMPLE_HISTORY = [
    {"role": "user", "content": "帮我算一下"},
    {
        "role": "assistant",
        "content": "我来计算",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "calculate",
                "arguments": '{"expression": "1 + 1"}',
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
]


class ConversionTests(unittest.TestCase):
    """A. 规范历史与工具 schema 的转换。"""

    def test_responses_input_is_flat_items(self) -> None:
        items = to_responses_input(SAMPLE_HISTORY)

        self.assertEqual(items[0], {"role": "user", "content": "帮我算一下"})
        self.assertEqual(items[1], {"role": "assistant", "content": "我来计算"})
        self.assertEqual(
            items[2],
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "calculate",
                "arguments": '{"expression": "1 + 1"}',
            },
        )
        self.assertEqual(
            items[3],
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"ok": true}',
            },
        )

    def test_responses_input_skips_empty_assistant_text(self) -> None:
        """中间轮只有工具调用，不应产生空的 assistant 消息项。"""
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "t", "arguments": "{}"},
                ],
            }
        ]

        items = to_responses_input(history)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "function_call")

    def test_chat_messages_nest_tool_calls(self) -> None:
        messages = to_chat_messages(SAMPLE_HISTORY)

        self.assertEqual(messages[0], {"role": "user", "content": "帮我算一下"})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "我来计算")
        self.assertEqual(
            messages[1]["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "1 + 1"}',
                    },
                }
            ],
        )
        self.assertEqual(
            messages[2],
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
        )

    def test_chat_tools_nest_function_schema(self) -> None:
        tools = to_chat_tools(agent.TOOLS)

        calculate = tools[0]
        self.assertEqual(calculate["type"], "function")
        self.assertEqual(calculate["function"]["name"], "calculate")
        self.assertIn("数学表达式", calculate["function"]["description"])
        self.assertIn("expression", calculate["function"]["parameters"]["properties"])


class ResponsesBackendTests(unittest.TestCase):
    """B. Responses API 适配器。"""

    def test_request_returns_turn_and_passes_responses_payload(self) -> None:
        client = FakeResponsesClient([responses_response([function_call_item()], "")])
        backend = ResponsesBackend(client)

        turn = backend.request([{"role": "user", "content": "算一下"}])

        self.assertEqual(turn.text, "")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call_1")
        self.assertEqual(turn.tool_calls[0].name, "calculate")

        kwargs = client.responses.calls[0]
        self.assertEqual(kwargs["instructions"], agent.SYSTEM_PROMPT)
        self.assertEqual(kwargs["input"], [{"role": "user", "content": "算一下"}])
        self.assertEqual(kwargs["tools"], agent.TOOLS)
        self.assertNotIn("stream", kwargs)

    def test_request_raises_with_real_reason_on_failed_status(self) -> None:
        failed = responses_response(
            status="failed",
            error=SimpleNamespace(code="server_error", message="上游挂了"),
        )
        backend = ResponsesBackend(FakeResponsesClient([failed]))

        with self.assertRaises(RuntimeError) as ctx:
            backend.request([])

        self.assertIn("server_error", str(ctx.exception))
        self.assertIn("上游挂了", str(ctx.exception))

    def test_stream_yields_text_then_completed(self) -> None:
        events = [
            responses_event("response.output_text.delta", delta="你"),
            responses_event("response.output_text.delta", delta="好"),
            responses_event(
                "response.completed",
                response=responses_response([], "你好"),
            ),
        ]
        backend = ResponsesBackend(FakeResponsesClient([events]))

        produced = list(backend.request_stream([{"role": "user", "content": "你好"}]))

        self.assertEqual(produced[0], TextDelta("你"))
        self.assertEqual(produced[1], TextDelta("好"))
        self.assertIsInstance(produced[2], Completed)
        self.assertEqual(produced[2].turn.text, "你好")

    def test_stream_carries_tool_calls(self) -> None:
        events = [
            responses_event(
                "response.completed",
                response=responses_response([function_call_item()], ""),
            ),
        ]
        backend = ResponsesBackend(FakeResponsesClient([events]))

        produced = list(backend.request_stream([]))

        self.assertIsInstance(produced[-1], Completed)
        self.assertEqual(produced[-1].turn.tool_calls[0].name, "calculate")

    def test_stream_reports_failed_and_incomplete_reasons(self) -> None:
        cases = {
            "response.failed": SimpleNamespace(code="server_error", message="上游过载"),
            "response.incomplete": SimpleNamespace(
                code="max_output_tokens", message="输出被截断"
            ),
            "error": SimpleNamespace(code="rate_limit_exceeded", message="限流了"),
        }

        for kind, error in cases.items():
            with self.subTest(kind=kind):
                failed_response = responses_response(status="failed", error=error)
                events = [responses_event(kind, response=failed_response)]
                if kind == "error":
                    # 网关错误事件的信息就在事件自身。
                    events = [responses_event(kind, code=error.code, message=error.message)]

                backend = ResponsesBackend(FakeResponsesClient([events]))
                produced = list(backend.request_stream([]))

                self.assertEqual(len(produced), 1)
                self.assertIsInstance(produced[0], Failed)
                self.assertIn(error.message, produced[0].reason)

    def test_stream_reports_missing_completed_event(self) -> None:
        events = [responses_event("response.output_text.delta", delta="只有文字")]
        backend = ResponsesBackend(FakeResponsesClient([events]))

        produced = list(backend.request_stream([]))

        self.assertIsInstance(produced[-1], Failed)
        self.assertIn("response.completed", produced[-1].reason)


class ChatBackendTests(unittest.TestCase):
    """C. Chat Completions 适配器。"""

    def test_request_converts_history_and_tools(self) -> None:
        client = FakeChatClient([chat_response(content="结果是 2")])
        backend = ChatCompletionsBackend(client)

        turn = backend.request([{"role": "user", "content": "算一下"}])

        self.assertEqual(turn.text, "结果是 2")
        self.assertEqual(turn.tool_calls, [])

        kwargs = client.chat.completions.calls[0]
        self.assertEqual(
            kwargs["messages"][0],
            {"role": "system", "content": agent.SYSTEM_PROMPT},
        )
        self.assertEqual(kwargs["messages"][1], {"role": "user", "content": "算一下"})
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "calculate")
        self.assertNotIn("stream", kwargs)

    def test_request_reads_tool_calls(self) -> None:
        client = FakeChatClient([chat_response(tool_calls=[chat_tool_call()])])
        backend = ChatCompletionsBackend(client)

        turn = backend.request([])

        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call_1")
        self.assertEqual(turn.tool_calls[0].arguments, '{"expression": "1 + 1"}')

    def test_request_raises_when_output_is_truncated(self) -> None:
        client = FakeChatClient([chat_response(finish_reason="length")])
        backend = ChatCompletionsBackend(client)

        with self.assertRaises(RuntimeError) as ctx:
            backend.request([])

        self.assertIn("length", str(ctx.exception))

    def test_stream_accumulates_text_and_tool_fragments(self) -> None:
        chunks = [
            chat_chunk(content="我"),
            chat_chunk(content="来算"),
            chat_chunk(tool_calls=[chat_fragment(0, "call_1", "calculate", "")]),
            chat_chunk(tool_calls=[chat_fragment(0, arguments='{"expression":')]),
            chat_chunk(
                tool_calls=[chat_fragment(0, arguments='"1 + 1"}')],
                finish_reason="tool_calls",
            ),
            SimpleNamespace(choices=[]),  # usage-only 收尾块
        ]
        backend = ChatCompletionsBackend(FakeChatClient([chunks]))

        produced = list(backend.request_stream([]))

        text = "".join(event.text for event in produced if isinstance(event, TextDelta))
        self.assertEqual(text, "我来算")

        completed = produced[-1]
        self.assertIsInstance(completed, Completed)
        self.assertEqual(completed.turn.text, "我来算")
        self.assertEqual(len(completed.turn.tool_calls), 1)
        self.assertEqual(completed.turn.tool_calls[0].id, "call_1")
        self.assertEqual(completed.turn.tool_calls[0].name, "calculate")
        self.assertEqual(
            completed.turn.tool_calls[0].arguments, '{"expression":"1 + 1"}'
        )

    def test_stream_reports_truncated_output(self) -> None:
        chunks = [
            chat_chunk(content="说到一半"),
            chat_chunk(finish_reason="length"),
        ]
        backend = ChatCompletionsBackend(FakeChatClient([chunks]))

        produced = list(backend.request_stream([]))

        self.assertIsInstance(produced[-1], Failed)
        self.assertIn("length", produced[-1].reason)


class FactoryTests(unittest.TestCase):
    """D. 后端选择。"""

    def test_build_backend_returns_matching_adapter(self) -> None:
        responses_client = FakeResponsesClient([])
        chat_client = FakeChatClient([])

        self.assertIsInstance(
            build_backend(responses_client, "responses"), ResponsesBackend
        )
        self.assertIsInstance(build_backend(chat_client, "chat"), ChatCompletionsBackend)

    def test_build_backend_rejects_unknown_style(self) -> None:
        with self.assertRaises(ValueError):
            build_backend(FakeResponsesClient([]), "grpc")


if __name__ == "__main__":
    unittest.main()
