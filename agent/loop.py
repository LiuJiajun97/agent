"""智能体循环模块：核心循环 + 单轮封装。

阅读顺序：
1. _tool_result_for - 处理单个工具调用（解析参数 + 分发 + 错误包装）
2. agent_loop - 核心循环，反复请求模型并执行工具
3. run_agent_turn - 把全局配置注入核心循环，并负责出错回滚
4. agent_loop_stream / run_agent_turn_stream - 上面两者的流式版本
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from agent.config import MAX_STEPS, MODEL, SYSTEM_PROMPT
from agent.tools import TOOLS, execute_tool


def _tool_result_for(call: Any) -> dict[str, Any]:
    """执行单个工具调用，返回可回传给模型的结果字典。

    call 是 Chat Completions 的 tool_call 对象，
    有 .id / .function.name / .function.arguments（JSON 字符串）。
    """
    try:
        arguments = json.loads(call.function.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        # 参数格式错误也要回传给模型，让它有机会修正。
        return {"ok": False, "error": f"参数解析失败: {exc}"}

    return execute_tool(call.function.name, arguments)


def agent_loop(
    request_model: Any,
    run_tool_call: Any,
    history: list[Any],
    max_steps: int = MAX_STEPS,
) -> str:
    """核心智能体循环：请求模型 -> 执行它要的工具 -> 回传结果，直到模型给出最终回答。

    不依赖任何全局配置，全部通过参数注入：

    - request_model：无参可调用对象，返回一次 Chat Completions 响应。
    - run_tool_call：接收单个 tool_call 项，返回结果字典。
    - history：对话历史，会被原地追加。
    - max_steps：最大模型请求次数，防止无限循环。

    达到 max_steps 仍有工具调用时抛出 RuntimeError。
    """
    for _ in range(max_steps):
        response = request_model()
        message = response.choices[0].message

        # 追加 assistant 消息到历史。
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        history.append(assistant_msg)

        # 没有工具调用 = 模型已经答完，直接返回。
        if not message.tool_calls:
            return message.content or "（模型没有返回文本）"

        for tc in message.tool_calls:
            result = run_tool_call(tc)

            # Chat Completions 的工具结果格式：role="tool" + tool_call_id。
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
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
        return client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
        )

    try:
        return agent_loop(request_model, _tool_result_for, history, MAX_STEPS)
    except Exception:
        # 先回滚，再用裸 raise 保留原始异常信息。
        del history[history_start:]
        raise


# ---- 流式版本 ----


@dataclass
class StreamEvent:
    """归一化后的流式事件。

    Chat Completions 的流式 chunk 里，文本增量在 delta.content，
    工具调用增量在 delta.tool_calls（分片，需要累积）。
    """

    # "text" 文本增量 / "tool_calls" 工具调用完成 / "final" 整轮响应结束
    kind: str
    # kind="text" 时的文本片段
    text: str = ""
    # kind="tool_calls" 时的 tool_call 列表（已累积完整）
    tool_calls: list = field(default_factory=list)


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
        # 累积器：index -> {id, name, arguments}
        tool_calls_acc: dict[int, dict[str, str]] = {}
        text_chunks: list[str] = []

        for raw in request_stream():
            delta = raw.choices[0].delta
            if delta.content:
                text_chunks.append(delta.content)
                for ch in delta.content:
                    yield ch
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name or "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

        # 构造 tool_calls 列表（模拟 Chat Completions 的 tool_call 对象）
        accumulated_tool_calls = []
        for idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[idx]
            accumulated_tool_calls.append(
                _make_tool_call(acc["id"], acc["name"], acc["arguments"])
            )

        # 追加 assistant 消息到历史。
        full_text = "".join(text_chunks)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_text or None}
        if accumulated_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in accumulated_tool_calls
            ]
        history.append(assistant_msg)

        # 没有工具调用 = 模型答完了。
        if not accumulated_tool_calls:
            return

        for tc in accumulated_tool_calls:
            result = run_tool_call(tc)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        # 回到循环顶部再次请求模型，此时历史里已有工具结果。

    raise RuntimeError(f"超过最大工具调用轮数 ({max_steps})")


class _ToolCallFunction:
    """模拟 Chat Completions tool_call.function 的命名空间。"""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """模拟 Chat Completions 的 tool_call 对象，供 _tool_result_for 使用。"""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _ToolCallFunction(name, arguments)


def _make_tool_call(id: str, name: str, arguments: str) -> _ToolCall:
    return _ToolCall(id, name, arguments)


def run_agent_turn_stream(
    client: Any,
    history: list[Any],
    user_text: str,
) -> Iterator[str]:
    """流式执行一轮用户请求，逐段产出模型文本。

    用的是 Chat Completions 的 stream 模式（client.chat.completions.create(stream=True)），
    粒度比 Responses API 更细。
    """
    # 与 run_agent_turn 一样：先记下起始位置，出错时回滚。
    history_start = len(history)
    history.append({"role": "user", "content": user_text})

    def request_stream() -> Any:
        return client.chat.completions.create(
            model=MODEL,
            messages=history,
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
