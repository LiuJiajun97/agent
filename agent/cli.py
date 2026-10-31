"""命令行入口：支持单次问答和交互式多轮对话。

本文件与 agent_lc/cli.py、agent_lg/cli.py 的 REPL 骨架刻意保持逐行相似，
方便横向对照三套实现；改交互行为时三处需要同步改。
"""

from __future__ import annotations

import os
import sys
from typing import Any

from agent.config import MODEL
from agent.loop import run_agent_turn_stream


def main() -> None:
    """命令行入口：单次问答 + 交互式 REPL。"""
    # Windows 控制台默认编码可能不是 UTF-8，统一重配置避免中文输出报错。
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    # 放在函数内导入：依赖缺失时在这里给中文提示，而不是让 ImportError 直接抛给用户。
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("缺少依赖，请先运行: pip install -r requirements.txt")

    # 只检查是否存在，不打印内容，避免密钥泄露。
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY")

    # OpenAI() 自动读取 OPENAI_API_KEY 和 OPENAI_BASE_URL，
    # 所以既能连官方接口，也能连 OpenAI-compatible 网关。
    client = OpenAI()

    # 单次模式用完即退出，交互模式在多轮之间保留历史。
    history: list[Any] = []

    # 带参数时执行单次问答，例如 python -m agent.cli "计算 1 + 2"
    if len(sys.argv) > 1:
        text = "".join(run_agent_turn_stream(client, history, " ".join(sys.argv[1:])))
        print(text)
        return

    print(f"Agent 原型（模型: {MODEL}）。输入 exit 退出，clear 清空上下文。")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C 或输入流结束时友好退出。
            print("\n再见。")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "q"} or user_text == "退出":
            break

        # 清空只重置对话历史，不重置客户端。
        if user_text.lower() in {"clear", "reset"} or user_text == "清空":
            history.clear()
            print("上下文已清空。")
            continue

        try:
            # 流式输出：模型吐一段就显示一段，不用等整段回答生成完。
            print("\nAgent> ", end="", flush=True)
            for chunk in run_agent_turn_stream(client, history, user_text):
                print(chunk, end="", flush=True)
            print()
        except Exception as exc:
            # 单轮失败不影响继续提问。
            print(f"\n[错误] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
