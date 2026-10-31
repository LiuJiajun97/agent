"""LangChain 版 Agent 组装：create_agent 建图 + checkpointer 多轮记忆 + 单轮调用。

与手写版 agent/loop.py 的对应关系：
- agent/loop.py 的 agent_loop()  <-  这里由 create_agent() 生成的图承担
- agent/loop.py 的 history 列表  <-  这里由 InMemorySaver checkpointer 承担
- agent/loop.py 的 MAX_STEPS     <-  这里由 LangGraph 的 recursion_limit 承担
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from agent.config import API_STYLE, MAX_STEPS, MODEL, SYSTEM_PROMPT
from agent_lc.tools import build_tools

# LangGraph 的 recursion_limit 计数的是图超级步：
# 一次"模型请求 + 工具执行"约等于 2 步。
# 手写版的 MAX_STEPS 语义是"最多 8 次模型请求"，乘 2 后两边大致对齐。
RECURSION_LIMIT = MAX_STEPS * 2

# 交互式会话使用的固定线程 ID，历史按它隔离。
DEFAULT_THREAD_ID = "default"


def build_model(streaming: bool = True, api_style: str | None = None) -> Any:
    """构造 ChatOpenAI 模型客户端。

    显式传 api_key 和 base_url，而不是依赖 SDK 隐式读环境变量，
    这样即使换网关也只需改 .env，行为可预期。

    api_style 不传时用 OPENAI_API_STYLE：
    - responses：走 /v1/responses，和手写版默认行为一致。
    - chat：走 /v1/chat/completions，兼容只支持 Chat Completions 的网关。

    output_version="responses/v1" 只在 Responses 模式下设置：
    它让 reasoning 摘要、内置工具调用等响应项落进 AIMessage.content
    （内容块列表）而不是 additional_kwargs。Chat 模式下 content 本来就是字符串。

    streaming=True 表示走流式接口：即使调用 invoke()，
    底层也是流式请求再聚合，对应 graph.stream() 能拿到逐字增量。
    """
    from langchain_openai import ChatOpenAI

    style = (api_style or API_STYLE).strip().lower()
    if style not in ("responses", "chat"):
        raise ValueError(f"未知的 API 风格: {style!r}（可选 responses / chat）")

    options: dict[str, Any] = {"use_responses_api": style == "responses"}
    if style == "responses":
        options["output_version"] = "responses/v1"

    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        streaming=streaming,
        **options,
    )


def build_checkpointer() -> Any:
    """构造内存版 checkpointer，用于保存多轮对话历史。"""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def build_agent(
    model: Any = None,
    tools: Any = None,
    checkpointer: Any = None,
) -> Any:
    """组装出可执行的 Agent 图。

    三个参数都可注入，方便测试时换成假模型而不发起网络请求。
    """
    from langchain.agents import create_agent

    return create_agent(
        model=model if model is not None else build_model(),
        tools=tools if tools is not None else build_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer if checkpointer is not None else build_checkpointer(),
    )


def message_text(message: Any) -> str:
    """从一条消息里提取纯文本。

    Responses API（output_version="responses/v1"）下 AIMessage.content
    是**内容块列表**，形如 [{"type": "text", "text": "..."}, ...]，
    而不是 Chat Completions 那种单个字符串。

    AIMessage.text 属性对两种形态都正确（字符串 content 原样返回，
    块列表则拼接其中的 text 块），所以优先用它；
    假模型 / 旧版消息对象没有 .text 时再回退到 content。
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # 内容块列表：只取 type=="text" 的块，跳过 reasoning / tool_call 等。
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)

    return ""


def run_turn_stream(
    agent: Any,
    user_text: str,
    thread_id: str = DEFAULT_THREAD_ID,
) -> Iterator[str]:
    """流式执行一轮对话，逐段产出模型文本。

    graph.stream(stream_mode="messages") 产出的是 (消息块, 元数据) 二元组。
    消息块里既有模型回复（AIMessageChunk），也有工具结果（ToolMessageChunk），
    这里只取模型回复的文本增量，工具结果不混进回答里。
    """
    from langchain_core.messages import AIMessageChunk

    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        },
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk):
            # 工具结果等非模型文本，跳过。
            continue
        text = message_text(chunk)
        # 空内容的收尾块跳过。
        if text:
            # 逐字吐出，避免终端一次性显示一整段。
            for ch in text:
                yield ch


def run_turn(agent: Any, user_text: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """执行一轮对话并返回模型的最终文本回答。

    历史由 checkpointer 按 thread_id 自动保存与恢复，
    这里只需要传入本轮的用户消息。

    走的是非流式接口（invoke），拿整轮结果里的最后一条消息。
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        },
    )

    messages = result.get("messages") or []
    if not messages:
        return "（模型没有返回文本）"

    last = messages[-1]
    return message_text(last) or "（模型没有返回文本）"


def clear_thread(checkpointer: Any, thread_id: str = DEFAULT_THREAD_ID) -> None:
    """清空某个线程的对话历史。

    对应手写版的 history.clear()。
    """
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if delete_thread is None:
        return
    try:
        delete_thread(thread_id)
    except Exception:
        # 清空失败不该中断交互，静默忽略即可。
        pass
