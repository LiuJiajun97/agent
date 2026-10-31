"""LangGraph 原生版：手写 StateGraph，把 create_agent 封装的图显式画出来。

create_agent 生成的图长这样：

    __start__ -> model
    model     -> __end__    (条件边)
    model     -> tools      (条件边)
    tools     -> model      (回边)

本模块用 LangGraph 的 StateGraph API 把 model / tools 两个节点
及它们之间的边逐条写出来（__start__ / __end__ 由框架隐式处理），
方便看清 create_agent 到底替你做了什么。

与另外两套实现的对应关系：
- agent/loop.py 的 agent_loop()  <-  本文件的 model / tools 节点 + 条件边
- agent/loop.py 的 history 列表  <-  本文件的 AgentState.messages
- agent/loop.py 的 MAX_STEPS     <-  RECURSION_LIMIT（转发自 agent_lc.graph）
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.config import SYSTEM_PROMPT
from agent_lc.graph import (
    DEFAULT_THREAD_ID,
    RECURSION_LIMIT,
    build_checkpointer,
    build_model,
    clear_thread,
    run_turn,
    run_turn_stream,
)
from agent_lg.tools import build_tools


class AgentState(TypedDict):
    """图的状态，只有一条消息列表。

    add_messages 是 LangGraph 的 reducer：
    节点返回 {"messages": [新消息]} 时，它负责把新消息追加到旧列表后面，
    而不是覆盖。这正是手写版 history.append 的对应物。
    """

    messages: Annotated[list, add_messages]


def build_graph(
    model: Any = None,
    tools: Any = None,
    checkpointer: Any = None,
    streaming: bool = True,
) -> Any:
    """手动画出 Agent 图并编译。

    三个参数都可注入，方便测试时换成假模型而不发起网络请求。
    checkpointer 缺省为 InMemorySaver，与 agent_lc.build_agent 保持一致。
    streaming 缺省为 True，模型走流式接口，图才能逐段产出。
    """
    model = model if model is not None else build_model(streaming=streaming)
    tools = tools if tools is not None else build_tools()
    checkpointer = checkpointer if checkpointer is not None else build_checkpointer()

    # 绑定工具只需做一次，模型节点每次调用直接用绑定好的版本。
    bound_model = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict[str, Any]:
        """模型节点：带系统提示词调用模型，把模型回复追加到状态。"""
        # 系统提示词只在调用模型时拼进去，不写进状态，
        # 这样多轮对话的状态里不会堆重复的系统消息。
        response = bound_model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    def route_after_model(state: AgentState) -> str:
        """条件边：模型要调工具就转给 tools 节点，否则直接结束。"""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", route_after_model, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "model")
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "DEFAULT_THREAD_ID",
    "RECURSION_LIMIT",
    "AgentState",
    "build_checkpointer",
    "build_graph",
    "build_model",
    "build_tools",
    "clear_thread",
    "run_turn",
    "run_turn_stream",
]
