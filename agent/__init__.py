"""最小可运行 Agent：Responses API / Chat Completions + Function Calling + 工具执行循环。

模块划分：
- agent.config：环境变量、模型名、API 风格、最大步数、系统指令
- agent.tools：工具清单 + 工具实现
- agent.backends：两种 API 的适配器，把差异收敛成 Turn / 流式事件
- agent.loop：核心循环 agent_loop + 单轮封装 run_agent_turn（含流式版本）
- agent.cli：命令行入口
"""

from agent.backends import (
    Backend,
    ChatCompletionsBackend,
    Completed,
    Failed,
    ResponsesBackend,
    TextDelta,
    ToolCall,
    Turn,
    build_backend,
    to_chat_messages,
    to_chat_tools,
    to_responses_input,
)
from agent.config import API_STYLE, MAX_STEPS, MODEL, SYSTEM_PROMPT
from agent.loop import (
    agent_loop,
    agent_loop_stream,
    run_agent_turn,
    run_agent_turn_stream,
)
from agent.tools import TOOLS, calculate, execute_tool, get_current_time

__all__ = [
    "API_STYLE",
    "Backend",
    "ChatCompletionsBackend",
    "Completed",
    "Failed",
    "MAX_STEPS",
    "MODEL",
    "ResponsesBackend",
    "SYSTEM_PROMPT",
    "TOOLS",
    "TextDelta",
    "ToolCall",
    "Turn",
    "agent_loop",
    "agent_loop_stream",
    "build_backend",
    "calculate",
    "execute_tool",
    "get_current_time",
    "run_agent_turn",
    "run_agent_turn_stream",
    "to_chat_messages",
    "to_chat_tools",
    "to_responses_input",
]
