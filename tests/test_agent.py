from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Iterator

import agent


def make_tool_call(id: str, name: str, arguments: str) -> SimpleNamespace:
    """模拟 Chat Completions 的 tool_call 对象。"""
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_message(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """模拟 Chat Completions 的 message 对象。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_response(message: SimpleNamespace) -> SimpleNamespace:
    """模拟 Chat Completions 的 response 对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )


def make_chunk(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """模拟 Chat Completions 的 stream chunk。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeCompletions:
    """模拟 client.chat.completions。"""

    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        resp = self._responses[self.calls - 1]
        if kwargs.get("stream"):
            # 流式：resp 应该是一个 list of chunks
            return iter(resp)
        return resp


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


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


class AgentLoopTests(unittest.TestCase):
    def test_tool_call_is_executed_and_returned_to_model(self) -> None:
        call = make_tool_call("call_test", "calculate", json.dumps({"expression": "(128 * 37 + 9) / 5"}))
        client = FakeClient(
            [
                make_response(make_message(tool_calls=[call])),
                make_response(make_message(content="计算结果是 949")),
            ]
        )
        history: list[Any] = []

        answer = agent.run_agent_turn(client, history, "请计算")

        self.assertEqual(answer, "计算结果是 949")
        self.assertEqual(client.chat.completions.calls, 2)
        outputs = [
            item
            for item in history
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        self.assertEqual(outputs[0]["tool_call_id"], "call_test")
        self.assertEqual(json.loads(outputs[0]["content"])["result"], 949.0)


class CoreLoopTests(unittest.TestCase):
    def test_agent_loop_with_injected_dependencies(self) -> None:
        """agent_loop 不依赖全局配置：模型请求和工具执行都可以随意替换。"""
        call = agent._make_tool_call("c1", "custom", json.dumps({"x": "a"}))
        rounds = [
            make_response(make_message(tool_calls=[call])),
            make_response(make_message(content="完成")),
        ]
        round_idx = [0]

        def request_model() -> Any:
            r = rounds[round_idx[0]]
            round_idx[0] += 1
            return r

        def run_tool_call(call: Any) -> dict[str, Any]:
            return {"ok": True, "result": 1}

        history: list[Any] = []
        answer = agent.agent_loop(request_model, run_tool_call, history)

        self.assertEqual(answer, "完成")
        outputs = [
            item
            for item in history
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        self.assertEqual(outputs[0]["tool_call_id"], "c1")

    def test_agent_loop_raises_after_max_steps(self) -> None:
        def request_model() -> Any:
            call = make_tool_call("x", "t", "{}")
            return make_response(make_message(tool_calls=[call]))

        with self.assertRaises(RuntimeError):
            agent.agent_loop(request_model, lambda call: {"ok": True}, [], max_steps=2)


class StreamingLoopTests(unittest.TestCase):
    """流式核心循环。"""

    def test_stream_yields_text_incrementally(self) -> None:
        """文本增量应该逐个 yield 出去，而不是攒成一段。"""
        chunks = [
            make_chunk(content="你"),
            make_chunk(content="好"),
        ]
        history: list[Any] = []

        result = list(
            agent.agent_loop_stream(
                lambda: iter(chunks),
                lambda c: {},
                history,
            )
        )

        self.assertEqual(result, ["你", "好"])

    def test_stream_executes_tool_then_continues(self) -> None:
        """流式模式下工具照常执行，结果照常回传。"""
        call_chunk = make_chunk(
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="c1",
                    function=SimpleNamespace(name="calculate", arguments='{"expression": "1 + 1"}'),
                )
            ]
        )
        rounds = [
            [call_chunk],
            [make_chunk(content="结果是 2")],
        ]
        round_idx = [0]

        def request_stream() -> Any:
            r = rounds[round_idx[0]]
            round_idx[0] += 1
            return iter(r)

        history: list[Any] = []
        chunks = list(
            agent.agent_loop_stream(
                request_stream, lambda c: {"ok": True, "result": 2}, history
            )
        )

        self.assertEqual("".join(chunks), "结果是 2")
        outputs = [
            item
            for item in history
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        self.assertEqual(outputs[0]["tool_call_id"], "c1")

    def test_stream_raises_after_max_steps(self) -> None:
        def request_stream() -> Any:
            call_chunk = make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="x",
                        function=SimpleNamespace(name="t", arguments="{}"),
                    )
                ]
            )
            return iter([call_chunk])

        with self.assertRaises(RuntimeError):
            list(agent.agent_loop_stream(request_stream, lambda c: {}, [], max_steps=2))


if __name__ == "__main__":
    unittest.main()
