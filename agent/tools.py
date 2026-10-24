"""工具模块：工具清单（发给模型的定义）+ 真正执行工具的 Python 函数。

阅读顺序：
1. TOOLS - 告诉模型"可以调用哪些函数"
2. calculate / get_current_time - 真正执行工具
3. execute_tool - 把模型请求分发到对应函数
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from typing import Any

# 工具清单：Chat Completions API 格式。
# 模型只能看到名称、描述和参数结构，真正的代码由我们自己执行。
# Responses API 的工具格式是"扁平"的：name / description / parameters 直接挂在顶层，
# 而不是嵌套在 function 字段下（那是 Chat Completions 的格式）。
TOOLS = [
    {
        "type": "function",
        "name": "calculate",
        "description": "安全地计算一个数学表达式，例如 (128 * 37 + 9) / 5。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "只包含数字、括号和 + - * / // % ** 运算符的表达式。",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "获取运行该程序的计算机当前本地日期和时间。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]

# 只允许这些运算，函数调用、属性访问、变量名等一律拒绝。
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_ast(node: ast.AST) -> int | float:
    """递归计算一个受限的 AST 节点。

    不用 eval 是因为它能执行任意代码；这里只认白名单节点，
    例如 __import__('os').system(...) 会被下面的 else 分支拒绝。
    """
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body)

    # 只接受数字常量。用 type() 判断而非 isinstance，排除 True / False。
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    # 一元运算：+10、-3
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _evaluate_ast(node.operand)
        result = _UNARY_OPS[type(node.op)](value)

    # 二元运算：1 + 2、10 / 5、2 ** 8
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)

        # 限制指数，避免 9 ** 999999999 耗尽 CPU 和内存。
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数绝对值不能超过 100")

        result = _BINARY_OPS[type(node.op)](left, right)

    else:
        raise ValueError("表达式中包含不支持的语法")

    # 限制超大整数，避免 10000 ** 100 耗尽资源。
    if isinstance(result, int) and result.bit_length() > 4096:
        raise ValueError("计算结果过大")

    # 排除 NaN 和 Infinity，例如 1e308 * 1e308。
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("计算结果不是有限数值")

    # 只返回普通数字，保证后面能安全地 json.dumps。
    if not isinstance(result, (int, float)):
        raise ValueError("计算结果必须是实数")

    return result


def calculate(expression: str) -> int | float:
    """安全地计算一个算术表达式。

    ast.parse 只负责把文本解析成树，不负责执行；
    真正的执行在 _evaluate_ast 中，只跑白名单节点。
    """
    expression = expression.strip()
    if not expression:
        raise ValueError("表达式不能为空")
    if len(expression) > 200:
        raise ValueError("表达式太长")

    # mode="eval" 只接受表达式，不接受 import / for / def 等语句。
    tree = ast.parse(expression, mode="eval")
    return _evaluate_ast(tree)


def get_current_time() -> dict[str, str | int]:
    """返回运行程序这台电脑的当前本地时间。"""
    now = datetime.now().astimezone()
    offset = now.utcoffset()

    return {
        # ISO 8601 格式，含日期、时间和时区偏移。
        "local_time": now.isoformat(timespec="seconds"),
        # 用秒表示 UTC 偏移，便于模型计算。
        "utc_offset_seconds": int(offset.total_seconds()) if offset else 0,
        "weekday": now.strftime("%A"),
    }


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """按工具名和参数执行对应函数。

    异常不向外抛，而是包装成 {"ok": False, "error": ...}，
    这样模型能看到失败原因并有机会修正参数。
    """
    try:
        if name == "calculate":
            result = calculate(str(arguments["expression"]))
            return {"ok": True, "result": result}

        if name == "get_current_time":
            # ** 把 get_current_time() 的返回字段展开到当前字典。
            return {"ok": True, **get_current_time()}

        # 模型理论上只能调用 TOOLS 中声明的工具，但这里仍做防御性检查。
        return {"ok": False, "error": f"未知工具: {name}"}

    except Exception as exc:
        # 捕获所有异常，让 Agent 能把错误作为工具结果回传给模型。
        # 生产环境通常还会在这里记录日志。
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
