"""LangChain 版 Agent：与 agent/ 平级并列，用于和手写版做对照。

模块划分：
- agent_lc.tools：@tool 包装层，委托给 agent.tools（不复制实现）
- agent_lc.graph：create_agent 建图 + InMemorySaver 多轮记忆 + 单轮调用
- agent_lc.cli：命令行入口
"""

from agent_lc.graph import (
    DEFAULT_THREAD_ID,
    RECURSION_LIMIT,
    build_agent,
    build_checkpointer,
    build_model,
    build_tools,
    clear_thread,
    message_text,
    run_turn,
    run_turn_stream,
)
from agent_lc.tools import calculate, get_current_time

__all__ = [
    "DEFAULT_THREAD_ID",
    "RECURSION_LIMIT",
    "build_agent",
    "build_checkpointer",
    "build_model",
    "build_tools",
    "calculate",
    "clear_thread",
    "get_current_time",
    "message_text",
    "run_turn",
    "run_turn_stream",
]
