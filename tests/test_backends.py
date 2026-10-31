"""API 适配层测试。

覆盖三部分：
- 规范历史与工具 schema 的转换（纯逻辑）
- 两个后端的端到端调用（真实网关，需要 .env 里配好 OPENAI_API_KEY / OPENAI_BASE_URL）
- 失败 / 截断等只能靠构造响应形态触发的分支（unittest.mock 打桩 SDK 方法）
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

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
from tests.live import real_openai_client, tool_call_arguments


def function_call_item(
    call_id: str = "call_1",
    name: str = "calculate",
    arguments: str = '{"expression": "1 + 1"}',
) -> SimpleNamespace:
    """构造 Responses API 的 function_call 输出项。"""
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def chat_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """构造 Chat Completions 的流式分片。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


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
        """真实调用：模型发起工具调用，参数是合法 JSON。"""
        backend = ResponsesBackend(real_openai_client())

        turn = backend.request(
            [{"role": "user", "content": "请使用 calculate 工具计算 123 * 45"}]
        )

        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "calculate")
        args = tool_call_arguments(turn.tool_calls[0].arguments)
        self.assertIn("expression", args)

    def test_request_raises_with_real_reason_on_failed_status(self) -> None:
        client = mock.MagicMock()
        client.responses.create.return_value = mock.Mock(
            status="failed",
            error=mock.Mock(code="server_error", message="上游挂了"),
            output=[],
            output_text="",
        )
        backend = ResponsesBackend(client)

        with self.assertRaises(RuntimeError) as ctx:
            backend.request([])

        self.assertIn("server_error", str(ctx.exception))
        self.assertIn("上游挂了", str(ctx.exception))

    def test_stream_yields_text_then_completed(self) -> None:
        """真实流式：先有文本增量，最后收尾是 Completed。"""
        backend = ResponsesBackend(real_openai_client())

        produced = list(
            backend.request_stream(
                [{"role": "user", "content": "用一句话说明什么是二分查找"}]
            )
        )

        deltas = [event for event in produced if isinstance(event, TextDelta)]
        self.assertTrue(deltas)
        self.assertIsInstance(produced[-1], Completed)
        self.assertEqual(
            produced[-1].turn.text, "".join(event.text for event in deltas)
        )

    def test_stream_carries_tool_calls(self) -> None:
        """真实流式触发工具调用：Completed 里带得上工具调用。"""
        backend = ResponsesBackend(real_openai_client())

        produced = list(
            backend.request_stream(
                [{"role": "user", "content": "请使用 calculate 工具计算 123 * 45"}]
            )
        )

        self.assertIsInstance(produced[-1], Completed)
        self.assertEqual(produced[-1].turn.tool_calls[0].name, "calculate")
        tool_call_arguments(produced[-1].turn.tool_calls[0].arguments)

    def test_stream_reports_failed_and_incomplete_reasons(self) -> None:
        cases = {
            "response.failed": mock.Mock(code="server_error", message="上游过载"),
            "response.incomplete": mock.Mock(
                code="max_output_tokens", message="输出被截断"
            ),
            "error": mock.Mock(code="rate_limit_exceeded", message="限流了"),
        }

        for kind, error in cases.items():
            with self.subTest(kind=kind):
                client = mock.MagicMock()
                if kind == "error":
                    # 网关错误事件的信息就在事件自身；
                    # response 必须显式置 None，否则 Mock 会自动补一个假属性出来。
                    client.responses.create.return_value = iter(
                        [
                            mock.Mock(
                                type=kind,
                                response=None,
                                code=error.code,
                                message=error.message,
                            )
                        ]
                    )
                else:
                    failed_response = mock.Mock(
                        status="failed", error=error, output=[], output_text=""
                    )
                    client.responses.create.return_value = iter(
                        [mock.Mock(type=kind, response=failed_response)]
                    )

                backend = ResponsesBackend(client)
                produced = list(backend.request_stream([]))

                self.assertEqual(len(produced), 1)
                self.assertIsInstance(produced[0], Failed)
                self.assertIn(error.message, produced[0].reason)

    def test_stream_reports_missing_completed_event(self) -> None:
        client = mock.MagicMock()
        client.responses.create.return_value = iter(
            [mock.Mock(type="response.output_text.delta", delta="只有文字")]
        )
        backend = ResponsesBackend(client)

        produced = list(backend.request_stream([]))

        self.assertIsInstance(produced[-1], Failed)
        self.assertIn("response.completed", produced[-1].reason)


class ChatBackendTests(unittest.TestCase):
    """C. Chat Completions 适配器。"""

    def test_request_converts_history_and_tools(self) -> None:
        """真实请求：历史与工具都带上了，模型给出回答。"""
        backend = ChatCompletionsBackend(real_openai_client())

        turn = backend.request(
            [{"role": "user", "content": "用一句话说明什么是哈希表"}]
        )

        self.assertTrue(turn.text.strip())

    def test_request_reads_tool_calls(self) -> None:
        backend = ChatCompletionsBackend(real_openai_client())

        turn = backend.request(
            [{"role": "user", "content": "请使用 calculate 工具计算 123 * 45"}]
        )

        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "calculate")
        args = tool_call_arguments(turn.tool_calls[0].arguments)
        self.assertIn("expression", args)

    def test_request_raises_when_output_is_truncated(self) -> None:
        client = mock.MagicMock()
        client.chat.completions.create.return_value = mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content="说到一半", tool_calls=None),
                    finish_reason="length",
                )
            ]
        )
        backend = ChatCompletionsBackend(client)

        with self.assertRaises(RuntimeError) as ctx:
            backend.request([])

        self.assertIn("length", str(ctx.exception))

    def test_stream_accumulates_text_and_tool_fragments(self) -> None:
        """真实流式：文本分片拼成完整回答，工具调用合并成一个。"""
        backend = ChatCompletionsBackend(real_openai_client())

        produced = list(
            backend.request_stream(
                [{"role": "user", "content": "请使用 calculate 工具计算 123 * 45"}]
            )
        )

        text = "".join(event.text for event in produced if isinstance(event, TextDelta))
        completed = produced[-1]
        self.assertIsInstance(completed, Completed)
        self.assertEqual(completed.turn.text, text)
        self.assertEqual(len(completed.turn.tool_calls), 1)
        self.assertEqual(completed.turn.tool_calls[0].name, "calculate")
        tool_call_arguments(completed.turn.tool_calls[0].arguments)

    def test_stream_reports_truncated_output(self) -> None:
        client = mock.MagicMock()
        client.chat.completions.create.return_value = iter(
            [
                chat_chunk(content="说到一半"),
                chat_chunk(finish_reason="length"),
            ]
        )
        backend = ChatCompletionsBackend(client)

        produced = list(backend.request_stream([]))

        self.assertIsInstance(produced[-1], Failed)
        self.assertIn("length", produced[-1].reason)


class FactoryTests(unittest.TestCase):
    """D. 后端选择。"""

    def test_build_backend_returns_matching_adapter(self) -> None:
        # 工厂只负责挑适配器，客户端传什么都不影响这里。
        self.assertIsInstance(build_backend(object(), "responses"), ResponsesBackend)
        self.assertIsInstance(build_backend(object(), "chat"), ChatCompletionsBackend)

    def test_build_backend_rejects_unknown_style(self) -> None:
        with self.assertRaises(ValueError):
            build_backend(object(), "grpc")


if __name__ == "__main__":
    unittest.main()
