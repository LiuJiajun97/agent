"""API 适配层：把 Responses API 和 Chat Completions 的差异收敛到两个后端。

循环（agent/loop.py）只认识三样东西：

- Turn：模型这一轮的文本和工具调用。
- TextDelta / Completed / Failed：流式事件。
- 规范历史：user / assistant（可带 tool_calls）/ tool 三种角色。

两种 API 的端点、消息格式、工具 schema、流式事件都不一样，
但都在这里翻译成统一形态，循环本身不需要知道底层用的是哪种 API。

阅读顺序：
1. ToolCall / Turn / 三种事件 - 循环与后端之间的契约
2. 历史与工具格式转换 - 规范格式 -> 各 API 的请求格式
3. ResponsesBackend / ChatCompletionsBackend - 两个适配器
4. build_backend - 按配置选择适配器
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from agent.config import API_STYLE, MODEL, SYSTEM_PROMPT
from agent.tools import TOOLS

Message = dict[str, Any]


@dataclass
class ToolCall:
    """一次工具调用，arguments 是模型生成的 JSON 字符串。"""

    id: str
    name: str
    arguments: str


@dataclass
class Turn:
    """模型一轮输出的规范化结果。

    text 是本轮文本（中间轮可能为空），tool_calls 是本轮要执行的工具。
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class TextDelta:
    """流式文本增量。"""

    text: str


@dataclass(frozen=True)
class Completed:
    """流式正常收尾，携带完整的一轮结果。"""

    turn: Turn


@dataclass(frozen=True)
class Failed:
    """流式失败，reason 是给人看的完整错误信息。"""

    reason: str


Event = TextDelta | Completed | Failed


class Backend(Protocol):
    """循环依赖的最小后端接口。"""

    def request(self, history: list[Message]) -> Turn: ...

    def request_stream(self, history: list[Message]) -> Iterator[Event]: ...


# ---- 格式转换 ----


def to_responses_input(history: list[Message]) -> list[dict[str, Any]]:
    """把规范历史转换成 Responses API 的 input item 列表。

    Responses API 的工具交互是平铺的 item：
    助手发起的 function_call 和工具返回的 function_call_output 都各占一项。
    """
    items: list[dict[str, Any]] = []

    for message in history:
        role = message.get("role")

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
            continue

        if role == "assistant":
            text = message.get("content") or ""
            # 空文本的中间轮不产生消息项，避免出现空 assistant 消息。
            if text:
                items.append({"role": "assistant", "content": text})
            for call in message.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )
            continue

        items.append({"role": role, "content": message.get("content", "")})

    return items


def to_chat_messages(history: list[Message]) -> list[dict[str, Any]]:
    """把规范历史转换成 Chat Completions 的 messages 列表。

    工具调用挂在 assistant 消息的 tool_calls 字段上，
    工具结果则是一条 role=tool 的消息，通过 tool_call_id 关联。
    """
    messages: list[dict[str, Any]] = []

    for message in history:
        role = message.get("role")

        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
            continue

        if role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            calls = message.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for call in calls
                ]
            messages.append(entry)
            continue

        messages.append({"role": role, "content": message.get("content", "")})

    return messages


def to_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把扁平的工具 schema 转成 Chat Completions 的嵌套格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        }
        for tool in tools
    ]


# ---- Responses API ----


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
    """把 Responses API 的失败事件翻译成一句人能看懂的错误信息。

    有两种失败信号：
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


def _turn_from_response(response: Any) -> Turn:
    """把一次 Responses API 响应转成 Turn。"""
    tool_calls: list[ToolCall] = []

    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        tool_calls.append(
            ToolCall(
                id=getattr(item, "call_id", None) or f"call_{len(tool_calls)}",
                name=getattr(item, "name", "") or "",
                arguments=getattr(item, "arguments", None) or "{}",
            )
        )

    return Turn(
        text=getattr(response, "output_text", None) or "",
        tool_calls=tool_calls,
    )


class ResponsesBackend:
    """OpenAI Responses API（/v1/responses）适配器。"""

    def __init__(
        self,
        client: Any,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        tools: list[dict[str, Any]] = TOOLS,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools

    def _create(self, history: list[Message], stream: bool) -> Any:
        return self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=to_responses_input(history),
            tools=self.tools,
            **({"stream": True} if stream else {}),
        )

    def request(self, history: list[Message]) -> Turn:
        response = self._create(history, stream=False)
        _raise_if_failed(response)
        return _turn_from_response(response)

    def request_stream(self, history: list[Message]) -> Iterator[Event]:
        final_response: Any = None
        failure: str | None = None

        for event in self._create(history, stream=True):
            kind = getattr(event, "type", None)

            if kind == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield TextDelta(delta)

            elif kind == "response.completed":
                # 正常收尾事件，带完整 response 对象。
                final_response = getattr(event, "response", None)

            elif kind in ("response.failed", "response.incomplete", "error"):
                # 模型侧失败、输出被截断，或网关级错误。
                failure = _describe_failure(event)

        if failure is not None:
            yield Failed(failure)
            return

        if final_response is None:
            yield Failed("流式响应未返回最终结果（缺少 response.completed 事件）")
            return

        yield Completed(_turn_from_response(final_response))


# ---- Chat Completions API ----


class ChatCompletionsBackend:
    """Chat Completions API（/v1/chat/completions）适配器。"""

    def __init__(
        self,
        client: Any,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        tools: list[dict[str, Any]] = TOOLS,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools

    def _create(self, history: list[Message], stream: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *to_chat_messages(history),
            ],
        }
        if self.tools:
            kwargs["tools"] = to_chat_tools(self.tools)
        if stream:
            kwargs["stream"] = True
        return self.client.chat.completions.create(**kwargs)

    def request(self, history: list[Message]) -> Turn:
        response = self._create(history, stream=False)
        choice = response.choices[0]

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise RuntimeError("模型输出被截断（finish_reason=length）")

        message = choice.message
        tool_calls: list[ToolCall] = []

        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{len(tool_calls)}",
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", None) or "{}",
                )
            )

        return Turn(
            text=getattr(message, "content", None) or "",
            tool_calls=tool_calls,
        )

    def request_stream(self, history: list[Message]) -> Iterator[Event]:
        text_parts: list[str] = []
        # 工具调用在流里是分片下发的，按 index 累积。
        call_buffers: dict[int, dict[str, Any]] = {}

        for chunk in self._create(history, stream=True):
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # 有些网关会额外发一个只带 usage 的空 choices 收尾块。
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)

            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    text_parts.append(content)
                    yield TextDelta(content)

                for fragment in getattr(delta, "tool_calls", None) or []:
                    index = getattr(fragment, "index", 0)
                    if index is None:
                        index = 0

                    buffer = call_buffers.setdefault(
                        index, {"id": None, "name": "", "arguments": ""}
                    )
                    fragment_id = getattr(fragment, "id", None)
                    if fragment_id:
                        buffer["id"] = fragment_id

                    function = getattr(fragment, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            buffer["name"] = name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            buffer["arguments"] += arguments

            if getattr(choice, "finish_reason", None) == "length":
                yield Failed("模型输出被截断（finish_reason=length）")
                return

        tool_calls: list[ToolCall] = []
        for index in sorted(call_buffers):
            buffer = call_buffers[index]
            if not buffer["name"]:
                # 分片不完整，没法确定工具名，跳过这个调用。
                continue
            tool_calls.append(
                ToolCall(
                    id=buffer["id"] or f"call_{index}",
                    name=buffer["name"],
                    arguments=buffer["arguments"] or "{}",
                )
            )

        yield Completed(Turn(text="".join(text_parts), tool_calls=tool_calls))


def build_backend(client: Any, style: str | None = None) -> Backend:
    """按配置构造后端；style 显式传入时优先于环境变量。"""
    resolved = (style or API_STYLE).strip().lower()

    if resolved == "responses":
        return ResponsesBackend(client)
    if resolved == "chat":
        return ChatCompletionsBackend(client)

    raise ValueError(f"未知的 API 风格: {resolved!r}（可选 responses / chat）")
