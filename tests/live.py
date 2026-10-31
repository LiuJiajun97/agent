"""真实调用网关的测试辅助：检查环境、建客户端、解析工具参数。

这个仓库里没有假模型也没有假客户端——
需要模拟特定响应形态的用例，直接用标准库 unittest.mock 打桩 SDK 方法，
或者临时替换工具函数的实现。
"""

from __future__ import annotations

import json
import os
from typing import Any


def require_gateway() -> None:
    """确认网关配置就绪。

    真实调用用例缺配置时必须直接失败，不能 skip——
    否则会出现"跳过了 20 个测试"却显示 OK 的假绿。
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise AssertionError(
            "缺少 OPENAI_API_KEY：真实调用用例需要网关令牌。"
            "用 OpenAI 兼容网关时还要设置 OPENAI_BASE_URL。"
        )


def real_openai_client() -> Any:
    """构造真实 OpenAI 客户端（base_url / key 从环境变量读）。"""
    require_gateway()
    from openai import OpenAI

    return OpenAI()


def tool_call_arguments(raw: str) -> dict[str, Any]:
    """把模型给的工具参数字符串解析成字典。"""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AssertionError(f"工具参数不是 JSON 对象: {raw!r}")
    return parsed


__all__ = ["real_openai_client", "require_gateway", "tool_call_arguments"]
