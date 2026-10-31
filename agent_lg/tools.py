"""LangGraph 原生版的工具层。

@tool 装饰器属于 LangChain 生态层（LangGraph 原生用法同样基于
langchain_core 的工具原语），所以直接复用 agent_lc 的实现，不重复定义。

真正的差别在 agent_lg/graph.py：那里的图是手动画出来的，
而不是由 create_agent 封装生成。

本模块是纯转发层：工具定义全部来自 agent_lc.tools，
而 agent_lc.tools 又委托给 agent.tools——三套实现的工具行为完全一致。
"""

from agent_lc.tools import build_tools, calculate, get_current_time

__all__ = ["build_tools", "calculate", "get_current_time"]
