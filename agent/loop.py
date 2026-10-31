"""智能体循环模块：核心循环 + 单轮封装。

阅读顺序：
1. _tool_result_for - 处理单个工具调用（解析参数 + 分发 + 错误包装）
2. agent_loop - 核心循环，反复请求模型并执行工具
3. run_agent_turn - 选择 API 后端、注入配置，并负责出错回滚
4. agent_loop_stream / run_agent_turn_stream - 上面两者的流式版本

两个循环变体共用同一段工具执行逻辑（_execute_tool_calls），
单轮封装的回滚逻辑也共用（_begin_turn / _rollback_on_error）。

循环本身不关心底层是 Responses API 还是 Chat Completions：
两种 API 的差异由 agent/backends.py 翻译成 Turn 和流式事件。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from agent.backends import (
    Completed,
    Failed,
    TextDelta,
    ToolCall,
    Turn,
    build_backend,
)
from agent.config import MAX_STEPS
from agent.tools import execute_tool


def _tool_result_for(call: ToolCall) -> dict[str, Any]:
    """执行单个工具调用，返回可回传给模型的结果字典。

    call.arguments 是模型生成的 JSON 字符串，格式错误也要回传给模型，
    让它有机会修正参数。
    """
    try:
        arguments = json.loads(call.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": f"参数解析失败: {exc}"}

    return execute_tool(call.name, arguments)


def _append_assistant(history: list[dict[str, Any]], turn: Turn) -> None:
    """把模型这一轮输出写进规范历史。"""
    message: dict[str, Any] = {"role": "assistant", "content": turn.text}
    if turn.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in turn.tool_calls
        ]
    history.append(message)


def _append_tool_result(
    history: list[dict[str, Any]], call: ToolCall, result: dict[str, Any]
) -> None:
    """把工具结果写进规范历史，供下一次请求回传给模型。"""
    history.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def _execute_tool_calls(
    history: list[dict[str, Any]], turn: Turn, run_tool_call: Any
) -> None:
    """执行本轮模型要的工具调用，并把结果回传进历史。

    非流式与流式两个循环变体在这一步完全一致，
    抽出来避免同一段逻辑在两处各写一遍、改一处忘另一处。
    """
    for call in turn.tool_calls:
        result = run_tool_call(call)
        _append_tool_result(history, call, result)


def _begin_turn(history: list[dict[str, Any]], user_text: str) -> int:
    """开始一轮对话：追加用户消息，返回回滚时要用的起始下标。"""
    history_start = len(history)
    history.append({"role": "user", "content": user_text})
    return history_start


@contextmanager
def _rollback_on_error(
    history: list[dict[str, Any]], history_start: int
) -> Iterator[None]:
    """出错时把历史回滚到本轮开始之前，避免留下"有问无答"的残缺历史。"""
    try:
        yield
    except Exception:
        # 先回滚，再用裸 raise 保留原始异常信息。
        del history[history_start:]
        raise


def _max_steps_exceeded(max_steps: int) -> RuntimeError:
    """超过最大模型请求次数时抛出的异常。

    两个循环变体共用这一处，保证报错文案不会走偏。
    """
    return RuntimeError(f"超过最大工具调用轮数 ({max_steps})")


def agent_loop(
    request: Any,
    run_tool_call: Any,
    history: list[dict[str, Any]],
    max_steps: int = MAX_STEPS,
) -> str:
    """核心智能体循环：请求模型 -> 执行它要的工具 -> 回传结果，直到模型给出最终回答。

    不依赖任何全局配置，全部通过参数注入：

    - request：无参可调用对象，返回规范化的 Turn。
    - run_tool_call：接收单个 ToolCall，返回结果字典。
    - history：规范格式的对话历史，会被原地追加。
    - max_steps：最大模型请求次数，防止无限循环。

    达到 max_steps 仍有工具调用时抛出 RuntimeError。
    """
    for _ in range(max_steps):
        turn = request()
        _append_assistant(history, turn)

        if not turn.tool_calls:
            return turn.text

        _execute_tool_calls(history, turn, run_tool_call)

    raise _max_steps_exceeded(max_steps)


def run_agent_turn(
    client: Any,
    history: list[dict[str, Any]],
    user_text: str,
    style: str | None = None,
) -> str:
    """执行一轮用户请求并返回最终回答。

    "一轮"可能包含多次模型请求：
    用户消息 -> 模型请求工具 -> 执行工具 -> 结果回传 -> 最终回答

    style 不传时用 OPENAI_API_STYLE 的值（responses / chat）。

    这是 agent_loop 的薄封装：选择 API 后端，并负责出错时回滚历史。
    """
    history_start = _begin_turn(history, user_text)

    with _rollback_on_error(history, history_start):
        backend = build_backend(client, style)
        return agent_loop(
            lambda: backend.request(history), _tool_result_for, history, MAX_STEPS
        )


# ---- 流式版本 ----


def agent_loop_stream(
    request_stream: Any,
    run_tool_call: Any,
    history: list[dict[str, Any]],
    max_steps: int = MAX_STEPS,
) -> Iterator[str]:
    """流式核心循环：逐段产出模型文本，工具调用仍在内部处理完再继续。

    与 agent_loop 的区别只有两点：
    1. 模型响应是逐段拿到的，所以边拿边 yield，不等全部生成完。
    2. 工具调用的执行时机不变——仍然等本轮模型的输出收齐后再执行。
    """
    for _ in range(max_steps):
        turn: Turn | None = None
        failure: str | None = None

        for event in request_stream():
            if isinstance(event, TextDelta):
                if event.text:
                    yield event.text
            elif isinstance(event, Completed):
                turn = event.turn
            elif isinstance(event, Failed):
                failure = event.reason

        if failure is not None:
            raise RuntimeError(failure)

        if turn is None:
            raise RuntimeError("流式响应未返回最终结果（缺少完成事件）")

        _append_assistant(history, turn)

        if not turn.tool_calls:
            return

        _execute_tool_calls(history, turn, run_tool_call)

    raise _max_steps_exceeded(max_steps)


def run_agent_turn_stream(
    client: Any,
    history: list[dict[str, Any]],
    user_text: str,
    style: str | None = None,
) -> Iterator[str]:
    """流式执行一轮用户请求，逐段产出模型文本。

    style 不传时用 OPENAI_API_STYLE 的值（responses / chat）。

    注意这是一个惰性生成器：用户消息在调用时立刻写入历史，
    但出错回滚要等到生成器被消费时才发生——调用方一旦开始迭代就按流式语义走。
    """
    history_start = _begin_turn(history, user_text)

    def _generate() -> Iterator[str]:
        with _rollback_on_error(history, history_start):
            backend = build_backend(client, style)
            yield from agent_loop_stream(
                lambda: backend.request_stream(history),
                _tool_result_for,
                history,
                MAX_STEPS,
            )

    return _generate()
