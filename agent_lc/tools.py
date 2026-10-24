"""LangChain 版工具层：把 agent.tools 的实现包装成 @tool。

与手写版的差别：手写版在 agent/tools.py 里手写 JSON Schema 描述参数；
这里由 @tool 从函数签名和 docstring 自动生成 schema，代码量明显更少。

工具的计算逻辑仍然来自 agent.tools，本模块只做包装，不复制实现，
保证两套实现的工具行为完全一致（便于横向对照）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from agent.tools import calculate as _calculate
from agent.tools import get_current_time as _get_current_time


@tool
def calculate(expression: str) -> dict[str, Any]:
    """安全地计算一个数学表达式，例如 (128 * 37 + 9) / 5。

    只支持数字、括号和 + - * / // % ** 运算符。
    """
    try:
        return {"ok": True, "result": _calculate(expression)}
    except Exception as exc:
        # 显式包装成结构化错误，和手写版 agent.tools.execute_tool 保持同构输出，
        # 这样两套实现喂给模型的工具结果格式完全一致，便于对照。
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@tool
def get_current_time() -> dict[str, Any]:
    """获取运行该程序的计算机当前本地日期和时间。"""
    try:
        return {"ok": True, **_get_current_time()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_tools() -> list[BaseTool]:
    """返回 LangChain 版工具清单。"""
    return [calculate, get_current_time]
