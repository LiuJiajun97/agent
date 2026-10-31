"""配置模块：环境变量、模型名、最大循环步数、系统指令。

模型名、API 风格、最大步数等配置在这里统一定义，.env 也只在这里加载一次；
OPENAI_API_KEY / OPENAI_BASE_URL 由 agent/cli.py 与 agent_lc/graph.py 直接读取，
因为它们要显式传给 SDK。
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:
    # 没装 dotenv 也能导入模块，方便测试。
    load_dotenv = None

if load_dotenv is not None:
    # override=False：系统环境变量优先于 .env。
    load_dotenv()

# 可在 .env 中用 OPENAI_MODEL 覆盖。
MODEL = os.getenv("OPENAI_MODEL", "MiniMax-M3.1-Flash")

# API 风格：responses（默认）或 chat。
API_STYLE = os.getenv("OPENAI_API_STYLE", "responses").strip().lower()
if API_STYLE not in ("responses", "chat"):
    raise ValueError(
        f"OPENAI_API_STYLE 只能是 responses 或 chat，当前为 {API_STYLE!r}"
    )

# 模型请求次数上限，防止无限调用工具和持续产生费用。
MAX_STEPS = 8

# 规定 Agent 的身份、工具使用方式和回答风格。
SYSTEM_PROMPT = """你是一个简洁、可靠的助手。
需要算术时调用 calculate 工具；需要当前时间时调用 get_current_time 工具。
不要编造工具结果。工具报错时，根据错误信息决定是否换一种方式调用。
默认使用用户当前使用的语言回答。"""
