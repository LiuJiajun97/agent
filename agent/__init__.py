"""最小可运行 Agent：Responses API + Function Calling + 工具执行循环。

模块划分：
- agent.config：环境变量、模型名、最大步数、系统指令
- agent.tools：工具清单 + 工具实现
- agent.loop：核心循环 agent_loop + 单轮封装 run_agent_turn（含流式版本）
- agent.cli：命令行入口
"""

from agent.config import MAX_STEPS, MODEL, SYSTEM_PROMPT
from agent.loop import (
    _ToolCall,
    _make_tool_call,
    agent_loop,
    agent_loop_stream,
    run_agent_turn,
    run_agent_turn_stream,
)
from agent.tools import TOOLS, calculate, execute_tool, get_current_time

__all__ = [
    "MAX_STEPS",
    "MODEL",
    "SYSTEM_PROMPT",
    "TOOLS",
    "agent_loop",
    "agent_loop_stream",
    "calculate",
    "execute_tool",
    "get_current_time",
    "run_agent_turn",
    "run_agent_turn_stream",
]
