"""智能体循环模块：核心循环 + 单轮封装。

阅读顺序：
1. _tool_result_for - 处理单个工具调用（解析参数 + 分发 + 错误包装）
2. agent_loop - 核心循环，反复请求模型并执行工具
3. run_agent_turn - 把全局配置注入核心循环，并负责出错回滚
4. agent_loop_stream / run_agent_turn_stream - 上面两者的流式版本
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from agent.config import MAX_STEPS, MODEL, SYSTEM_PROMPT
from agent.tools import TOOLS, execute_tool


def _raise_if_failed(response: Any) -> None:
    """非流式响应如果状态是 failed / incomplete，抛出带真实原因的异常。

    SDK 只在 HTTP 错误码时抛异常；HTTP 200 + status=failed 时会正常返回，
    不检查的话会拿到一个空 output，看起来像"模型什么都没说"。
    """
    status = getattr(response, "status", None)
    if status not in ("failed", "incomplete"):
        return

    error = getattr(response, "error", None)
    code = getattr(error, "code", None) if error is not None else None
    message = getattr(error, "message", None) if error is not None else None

    parts = [f"模型响应未成功完成（status={status}）"]
    if code:
        parts.append(f"code={code}")
    if message:
        parts.append(message)
    raise RuntimeError("，".join(parts))


def _describe_failure(event: Any) -> str:
    """把失败事件翻译成一句人能看懂的错误信息。

    Responses API 有两种失败信号：
    - response.failed / response.incomplete：事件里带 .response，
      真正的错误在 response.error（.code / .message）。
    - error：网关直接下发的错误事件，错误信息就在事件自身。
    """
    kind = getattr(event, "type", "unknown")

    response = getattr(event, "response", None)
    error = getattr(response, "error", None) if response is not None else None

    if error is None:
        error = event if kind == "error" else None

    if error is None:
        return f"流式响应失败（事件类型: {kind}）"

    code = getattr(error, "code", None)
    message = getattr(error, "message", None) or "（无详细信息）"

    parts = [f"流式响应失败（{kind}）"]
    if code:
        parts.append(f"code={code}")
    parts.append(message)
    return "，".join(parts)


def _tool_result_for(call: Any) -> dict[str, Any]:
    """执行单个工具调用，返回可回传给模型的结果字典。

    call 是 Responses API 的 function_call 输出项，
    有 .name / .arguments（JSON 字符串）/ .call_id。
    """
    try:
        arguments = json.loads(getattr(call, "arguments", None) or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        # 参数格式错误也要回传给模型，让它有机会修正。
        return {"ok": False, "error": f"参数解析失败: {exc}"}

    return execute_tool(call.name, arguments)


def agent_loop(
    request_model: Any,
    run_tool_call: Any,
    history: list[Any],
    max_steps: int = MAX_STEPS,
) -> str:
    """核心智能体循环：请求模型 -> 执行它要的工具 -> 回传结果，直到模型给出最终回答。

    不依赖任何全局配置，全部通过参数注入：

    - request_model：无参可调用对象，返回一次 Responses API 响应。
    - run_tool_call：接收单个 function_call 项，返回结果字典。
    - history：对话历史，会被原地追加。
    - max_steps：最大模型请求次数，防止无限循环。

    达到 max_steps 仍有工具调用时抛出 RuntimeError。
    """
    for _ in range(max_steps):
        response = request_model()

        # HTTP 200 但 status=failed/incomplete 的情况 SDK 不会抛异常，
        # 这里主动检查，否则会把空回答当成正常结果返回。
        _raise_if_failed(response)

        # Responses API 的输出是 item 列表，直接追加到历史。
        history.extend(response.output)

        tool_calls = [i for i in response.output if getattr(i, "type", None) == "function_call"]
        if not tool_calls:
            return response.output_text

        for call in tool_calls:
            result = run_tool_call(call)

            # Responses API 的工具结果格式：function_call_output 项。
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        # 回到循环顶部再次请求模型，此时历史里已有工具结果。

    raise RuntimeError(f"超过最大工具调用轮数 ({max_steps})")


def run_agent_turn(client: Any, history: list[Any], user_text: str) -> str:
    """执行一轮用户请求并返回最终回答。

    "一轮"可能包含多次模型请求：
    用户消息 -> 模型请求工具 -> 执行工具 -> 结果回传 -> 最终回答

    这是 agent_loop 的薄封装：注入配置和客户端，并负责出错时回滚历史。
    """
    # 记下起始位置，出错时回滚，避免留下"有问无答"的残缺历史。
    history_start = len(history)
    history.append({"role": "user", "content": user_text})

    def request_model() -> Any:
        # 每次请求都是独立的，所以消息历史、工具清单都要完整传入。
        return client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=history,
            tools=TOOLS,
        )

    try:
        return agent_loop(request_model, _tool_result_for, history, MAX_STEPS)
    except Exception:
        # 先回滚，再用裸 raise 保留原始异常信息。
        del history[history_start:]
        raise


# ---- 流式版本 ----


def agent_loop_stream(
    request_stream: Any,
    run_tool_call: Any,
    history: list[Any],
    max_steps: int = MAX_STEPS,
) -> Iterator[str]:
    """流式核心循环：逐段产出模型文本，工具调用仍在内部处理完再继续。

    与 agent_loop 的区别只有两点：
    1. 模型响应是逐段拿到的，所以边拿边 yield，不等全部生成完。
    2. 工具调用的执行时机不变——仍然等本轮模型的输出收齐后再执行。
    """
    for _ in range(max_steps):
        final_response: Any = None
        failure: str | None = None

        for event in request_stream():
            kind = getattr(event, "type", None)

            if kind == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield delta

            elif kind == "response.completed":
                # 正常收尾事件，带完整 response 对象。
                final_response = getattr(event, "response", None)

            elif kind in ("response.failed", "response.incomplete"):
                # 模型侧失败或输出被截断，要把真实原因告诉调用方，
                # 不能只报"流式响应未返回最终结果"。
                failure = _describe_failure(event)

            elif kind == "error":
                # 网关级错误事件。
                failure = _describe_failure(event)

        if failure is not None:
            raise RuntimeError(failure)

        if final_response is None:
            raise RuntimeError("流式响应未返回最终结果（缺少 response.completed 事件）")

        history.extend(final_response.output)

        tool_calls = [
            i for i in final_response.output
            if getattr(i, "type", None) == "function_call"
        ]

        if not tool_calls:
            return

        for call in tool_calls:
            result = run_tool_call(call)
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        # 回到循环顶部再次请求模型，此时历史里已有工具结果。

    raise RuntimeError(f"超过最大工具调用轮数 ({max_steps})")


class _ToolCall:
    """模拟 Responses API 的 function_call 输出项，供 _tool_result_for 使用。"""

    def __init__(self, call_id: str, name: str, arguments: str):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.type = "function_call"


def _make_tool_call(call_id: str, name: str, arguments: str) -> _ToolCall:
    return _ToolCall(call_id, name, arguments)


def run_agent_turn_stream(
    client: Any,
    history: list[Any],
    user_text: str,
) -> Iterator[str]:
    """流式执行一轮用户请求，逐段产出模型文本。

    用的是 Responses API 的 stream 模式（client.responses.create(stream=True)）。
    """
    history_start = len(history)
    history.append({"role": "user", "content": user_text})

    def request_stream() -> Any:
        return client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=history,
            tools=TOOLS,
            stream=True,
        )

    def _generate() -> Iterator[str]:
        try:
            yield from agent_loop_stream(
                request_stream, _tool_result_for, history, MAX_STEPS
            )
        except Exception:
            del history[history_start:]
            raise

    return _generate()
