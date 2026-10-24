from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Iterator

import agent


def make_function_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """模拟 Responses API 的 function_call 输出项。"""
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def make_response(output: list | None = None, output_text: str = "") -> SimpleNamespace:
    """模拟 Responses API 的 response 对象。"""
    return SimpleNamespace(output=output or [], output_text=output_text)


def make_event(kind: str, **kwargs: Any) -> SimpleNamespace:
    """模拟 Responses API 的流式事件。"""
    obj = SimpleNamespace(type=kind)
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj


def make_error(code: str, message: str) -> SimpleNamespace:
    """模拟 Responses API 的 ResponseError。"""
    return SimpleNamespace(code=code, message=message)


class FakeResponses:
    """模拟 client.responses。"""

    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        resp = self._responses[self.calls - 1]
        if kwargs.get("stream"):
            # 流式：resp 应该是一个 list of events
            return iter(resp)
        return resp


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.responses = FakeResponses(responses)


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
        call = make_function_call("call_test", "calculate", json.dumps({"expression": "(128 * 37 + 9) / 5"}))
        client = FakeClient(
            [
                make_response([call], ""),
                make_response([], "计算结果是 949"),
            ]
        )
        history: list[Any] = []

        answer = agent.run_agent_turn(client, history, "请计算")

        self.assertEqual(answer, "计算结果是 949")
        self.assertEqual(client.responses.calls, 2)
        outputs = [
            item
            for item in history
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual(outputs[0]["call_id"], "call_test")
        self.assertEqual(json.loads(outputs[0]["output"])["result"], 949.0)


class CoreLoopTests(unittest.TestCase):
    def test_agent_loop_with_injected_dependencies(self) -> None:
        """agent_loop 不依赖全局配置：模型请求和工具执行都可以随意替换。"""
        call = agent._make_tool_call("c1", "custom", json.dumps({"x": "a"}))
        rounds = [
            make_response([call], ""),
            make_response([], "完成"),
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
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual(outputs[0]["call_id"], "c1")

    def test_agent_loop_raises_after_max_steps(self) -> None:
        def request_model() -> Any:
            call = make_function_call("x", "t", "{}")
            return make_response([call], "")

        with self.assertRaises(RuntimeError):
            agent.agent_loop(request_model, lambda call: {"ok": True}, [], max_steps=2)

    def test_agent_loop_raises_on_failed_response(self) -> None:
        """HTTP 200 但 status=failed 时，SDK 不抛异常，这里要主动报错。

        不检查的话会拿到空 output，看起来像"模型什么都没说"。
        """
        failed = SimpleNamespace(
            output=[],
            output_text="",
            status="failed",
            error=SimpleNamespace(code="server_error", message="上游挂了"),
        )

        def request_model() -> Any:
            return failed

        with self.assertRaises(RuntimeError) as ctx:
            agent.agent_loop(request_model, lambda call: {"ok": True}, [])

        message = str(ctx.exception)
        self.assertIn("server_error", message)
        self.assertIn("上游挂了", message)


class StreamingLoopTests(unittest.TestCase):
    """流式核心循环。"""

    def test_stream_yields_text_incrementally(self) -> None:
        events = [
            make_event("response.output_text.delta", delta="你"),
            make_event("response.output_text.delta", delta="好"),
            make_event("response.completed", response=make_response([], "你好")),
        ]
        history: list[Any] = []

        result = list(
            agent.agent_loop_stream(
                lambda: iter(events),
                lambda c: {},
                history,
            )
        )

        self.assertEqual(result, ["你", "好"])

    def test_stream_executes_tool_then_continues(self) -> None:
        call = make_function_call("c1", "calculate", '{"expression": "1 + 1"}')
        round1 = [
            make_event("response.completed", response=make_response([call], "")),
        ]
        round2 = [
            make_event("response.output_text.delta", delta="结果是 2"),
            make_event("response.completed", response=make_response([], "结果是 2")),
        ]
        rounds = [round1, round2]
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
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual(outputs[0]["call_id"], "c1")

    def test_stream_raises_after_max_steps(self) -> None:
        def request_stream() -> Any:
            call = make_function_call("x", "t", "{}")
            return iter([make_event("response.completed", response=make_response([call], ""))])

        with self.assertRaises(RuntimeError):
            list(agent.agent_loop_stream(request_stream, lambda c: {}, [], max_steps=2))

    def test_stream_raises_with_real_reason_on_failed_response(self) -> None:
        """response.failed 必须报出真实原因，而不是"未返回最终结果"。"""
        failed_response = SimpleNamespace(
            output=[],
            output_text="",
            status="failed",
            error=make_error("server_error", "上游模型过载"),
        )

        def request_stream() -> Any:
            return iter(
                [
                    make_event("response.output_text.delta", delta="开头"),
                    make_event("response.failed", response=failed_response),
                ]
            )

        with self.assertRaises(RuntimeError) as ctx:
            list(agent.agent_loop_stream(request_stream, lambda c: {}, []))

        message = str(ctx.exception)
        self.assertIn("server_error", message)
        self.assertIn("上游模型过载", message)

    def test_stream_raises_on_incomplete_response(self) -> None:
        """response.incomplete（比如输出被 max_output_tokens 截断）也要报错。"""
        incomplete = SimpleNamespace(
            output=[],
            output_text="",
            status="incomplete",
            error=make_error("max_output_tokens", "输出被截断"),
        )

        def request_stream() -> Any:
            return iter([make_event("response.incomplete", response=incomplete)])

        with self.assertRaises(RuntimeError) as ctx:
            list(agent.agent_loop_stream(request_stream, lambda c: {}, []))

        self.assertIn("输出被截断", str(ctx.exception))

    def test_stream_raises_on_gateway_error_event(self) -> None:
        """网关直接下发 error 事件时也要报出信息。"""
        def request_stream() -> Any:
            return iter(
                [make_event("error", code="rate_limit_exceeded", message="限流了")]
            )

        with self.assertRaises(RuntimeError) as ctx:
            list(agent.agent_loop_stream(request_stream, lambda c: {}, []))

        self.assertIn("限流了", str(ctx.exception))

    def test_stream_raises_when_completed_event_missing(self) -> None:
        """整条流没有 response.completed 事件时，错误信息要说清是缺事件。"""
        def request_stream() -> Any:
            return iter([make_event("response.output_text.delta", delta="只有文字")])

        with self.assertRaises(RuntimeError) as ctx:
            list(agent.agent_loop_stream(request_stream, lambda c: {}, []))

        self.assertIn("response.completed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
