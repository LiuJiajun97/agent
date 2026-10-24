# 最小 Python Agent

一个不使用 Agent 框架的最小可运行原型：

```text
用户消息
   ↓
模型判断是否需要工具
   ↓
Python 执行工具
   ↓
工具结果回传模型
   ↓
最终回答
```

项目使用 OpenAI Chat Completions API 和 Function Calling，包含安全的算术工具、当前时间工具、多轮对话与自动化测试。

同时提供两套框架实现与手写版**三方并列**：

| 目录 | 实现方式 | 用途 |
|---|---|---|
| `agent/` | 手写 OpenAI SDK 循环 | 理解 Agent 循环到底怎么跑 |
| `agent_lc/` | LangChain `create_agent` | 理解框架封装后的形态 |
| `agent_lg/` | 原生 LangGraph `StateGraph` | 看清 `create_agent` 封装了什么 |

## 项目结构

```text
agent/
├── .venv/                  # 本地虚拟环境，不提交
├── .idea/                  # PyCharm 配置，不提交
├── .env.example            # 配置模板
├── .gitignore
├── agent/                  # 手写版（不使用 Agent 框架）
│   ├── __init__.py         # 包说明 + 统一导出公共 API
│   ├── config.py           # 环境变量加载、模型名、最大步数、系统指令
│   ├── tools.py            # TOOLS 清单 + calculate / get_current_time / execute_tool
│   ├── loop.py             # agent_loop 核心循环 + run_agent_turn 单轮封装（含流式版）
│   └── cli.py              # main 命令行入口（单次问答 + 交互 REPL）
├── agent_lc/               # LangChain 版（create_agent 封装）
│   ├── __init__.py         # 包说明 + 统一导出公共 API
│   ├── tools.py            # @tool 包装层，委托给 agent.tools
│   ├── graph.py            # create_agent 建图 + checkpointer 多轮记忆 + 流式调用
│   └── cli.py              # main 命令行入口（交互体验与手写版一致）
├── agent_lg/               # LangGraph 原生版（手写 StateGraph）
│   ├── __init__.py         # 包说明 + 统一导出公共 API
│   ├── tools.py            # 复用 agent_lc 的 @tool 工具
│   ├── graph.py            # 手写节点 + 条件边 + StateGraph + 流式调用
│   └── cli.py              # main 命令行入口（交互体验与手写版一致）
├── pyproject.toml          # 项目元数据和依赖
├── README.md
├── requirements.txt        # 便于直接安装的依赖清单
└── tests/
    ├── __init__.py
    ├── test_agent.py       # 手写版测试
    ├── test_lc.py          # LangChain 版测试
    └── test_lg.py          # LangGraph 原生版测试
```

## 1. 安装依赖

建议使用 Python 3.10 或更高版本。

```powershell
cd C:\Users\LJJ\Desktop\agent
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果还没有虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不要向 uv 管理的全局 Python 安装依赖，也不要使用 `--break-system-packages`。

依赖包含两套实现的全部需要：`openai`（手写版）、`langchain` / `langchain-openai` / `langgraph`（LangChain 版）。

## 2. 配置 API Key

复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
OPENAI_API_KEY=你的 API Key
OPENAI_MODEL=MiniMax-M3.1-Flash
```

使用兼容 OpenAI 的网关时，取消注释并填写：

```dotenv
OPENAI_BASE_URL=https://你的网关地址/v1
```

`.env` 已被 `.gitignore` 忽略，不要提交真实密钥。

也可以只在当前 PowerShell 会话设置：

```powershell
$env:OPENAI_API_KEY = "你的 API Key"
```

## 3. PyCharm 配置

将项目解释器指向：

```text
C:\Users\LJJ\Desktop\agent\.venv\Scripts\python.exe
```

路径：`Settings` → `Project: agent` → `Python Interpreter` → `Add Interpreter` → `Add Local Interpreter` → `Existing`。

如果出现 `externally-managed-environment`，说明 PyCharm 正在使用 uv 管理的全局解释器，切换到项目 `.venv` 即可。

## 4. 运行

### 手写版

交互模式：

```powershell
.\.venv\Scripts\python.exe -m agent.cli
```

单次模式：

```powershell
.\.venv\Scripts\python.exe -m agent.cli "请计算 (128 * 37 + 9) / 5"
.\.venv\Scripts\python.exe -m agent.cli "现在几点？再计算 17 ** 3"
```

示例对话：

```text
你好，请先计算 (128 * 37 + 9) / 5，再告诉我现在几点。
```

### LangChain 版

命令和交互方式与手写版完全一致：

```powershell
.\.venv\Scripts\python.exe -m agent_lc.cli
.\.venv\Scripts\python.exe -m agent_lc.cli "请计算 (128 * 37 + 9) / 5"
```

### LangGraph 原生版

```powershell
.\.venv\Scripts\python.exe -m agent_lg.cli
.\.venv\Scripts\python.exe -m agent_lg.cli "请计算 (128 * 37 + 9) / 5"
```

### 安装为命令

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
minimal-agent "你好"        # 手写版
minimal-agent-lc "你好"      # LangChain 版
minimal-agent-lg "你好"      # LangGraph 原生版
```

## 5. 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

共 39 个测试：`test_agent.py` 覆盖手写版，`test_lc.py` 覆盖 LangChain 版，`test_lg.py` 覆盖 LangGraph 原生版。
测试均不调用真实 API，不需要 API Key。

## 6. 核心循环

核心循环在 `agent/loop.py` 的 `agent_loop()`，它不依赖任何全局配置，模型请求和工具执行都靠参数注入：

```python
for _ in range(max_steps):
    response = request_model()      # 请求模型
    history.extend(response.output)  # 记录本轮输出

    tool_calls = [i for i in response.output if i.type == "function_call"]
    if not tool_calls:               # 没有工具调用 = 模型答完了
        return response.output_text

    for call in tool_calls:          # 执行工具并回传结果
        result = run_tool_call(call)
        history.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, ensure_ascii=False),
        })
```

`run_agent_turn()` 只是薄封装：把 `MODEL` / `SYSTEM_PROMPT` / `TOOLS` 和 API 客户端包装成上面两个可调用对象，并负责出错时回滚历史。

如果没有工具调用，`response.output_text` 就是最终回答；否则把工具结果追加到历史，再次请求模型，直到模型给出最终回答或达到最大轮数。

## 7. 添加工具

需要同时修改两处：

1. 在 `TOOLS` 中增加 JSON Schema。
2. 在 `execute_tool()` 中增加同名处理分支。

生产环境还应增加权限控制、超时、重试、日志、人工确认和成本上限。

## 8. 框架对照版

项目里有三套实现，跑同一个模型、同一套工具、同一份系统提示词，可以直接横向对比。

### 8.1 LangChain 版（`agent_lc/`）

`create_agent` 一行就能建出完整的 Agent 图：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,              # ChatOpenAI 实例
    tools=tools,              # @tool 包装的两个工具
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,  # InMemorySaver，保存多轮历史
)
```

多轮记忆由 checkpointer 按 `thread_id` 自动维护，每次只需传入本轮消息：

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": user_text}]},
    config={
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    },
)
answer = result["messages"][-1].content
```

### 8.2 LangGraph 原生版（`agent_lg/`）

`create_agent` 内部其实就是构造一张 `StateGraph` 再 `.compile()`。
`agent_lg/graph.py` 把这张图的四条边逐条写出来：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(AgentState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile(checkpointer=checkpointer)
```

其中 `add_messages` 是 LangGraph 的 reducer，作用是把节点返回的新消息**追加**到旧列表后面，
对应手写版的 `history.append`。

`route_after_model` 就是手写版里那句 `if not tool_calls: return`：

```python
def route_after_model(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
```

### 8.3 三套实现对照

| 维度 | 手写版 `agent/` | LangChain 版 `agent_lc/` | LangGraph 原生版 `agent_lg/` |
|---|---|---|---|
| Agent 循环 | 自己写 `agent_loop()` | `create_agent` 封装生成 | 手写 `StateGraph` 节点与边 |
| 工具参数 schema | 手写 JSON Schema | `@tool` 自动生成 | 同 LangChain 版（复用） |
| 工具实现 | `agent/tools.py` | 包装 `agent/tools.py` | 同 LangChain 版（复用） |
| 多轮记忆 | 手工维护 `history` 列表 | `InMemorySaver` + `thread_id` | 同 LangChain 版 |
| 步数上限 | `MAX_STEPS = 8`（模型请求数） | `recursion_limit = 16`（图超级步） | 同 LangChain 版 |
| 工具错误回传 | 显式包装成 `{"ok": False, ...}` | 同左 | 同左 |
| 走的 API 路径 | Responses API `/v1/responses` | Chat Completions `/v1/chat/completions` | 同 LangChain 版 |
| 命令 | `minimal-agent` | `minimal-agent-lc` | `minimal-agent-lg` |

三套实现生成的图结构完全一致，都是：

```text
__start__ -> model
model     -> __end__    (条件边)
model     -> tools      (条件边)
tools     -> model      (回边)
```

### 8.4 两处有意的行为差异

1. **步数上限是近似对齐**：LangGraph 的 `recursion_limit` 计数的是「图超级步」（一次模型请求 + 一次工具执行约 2 步），而 `MAX_STEPS` 语义是「模型请求次数」，所以取 `MAX_STEPS * 2 = 16`，两边大致相当但非严格等价。另外 `create_agent` 自己默认把 `recursion_limit` 设成 9999，所以框架版必须显式传值才能保住成本上限。
2. **工具结果格式**：手写版用 `json.dumps(result, ensure_ascii=False)`，框架版由 `ToolNode` 序列化。两者都是 JSON 文本，结构一致。

三套实现现在走**同一条** Chat Completions API 路径，对照更公平——差异只剩「循环谁来写」。

### 8.5 错误处理说明

LangChain 原生提供 `handle_tool_error=True`，会把异常文本直接回传给模型。
本项目**故意不用**，而是在工具体内显式 `try/except` 包成 `{"ok": False, "error": ...}`——
这样三套实现喂给模型的工具结果格式完全一致，横向对照时输出形态不会干扰判断。

## 9. 流式输出

三套实现都提供 `run_turn_stream()`，返回一个**文本增量生成器**：

```python
for chunk in run_turn_stream(client, history, "讲个笑话"):
    print(chunk, end="", flush=True)
```

命令行入口默认就走流式，模型吐一段就显示一段。

### 9.1 手写版：Chat Completions 的 stream 模式

```python
stream = client.chat.completions.create(
    model=MODEL, messages=history, tools=TOOLS, stream=True,
)
for raw in stream:
    delta = raw.choices[0].delta
    if delta.content:
        for ch in delta.content:
            yield ch
    if delta.tool_calls:
        # 累积工具调用参数（分片），收齐后执行
        ...
```

Chat Completions 的流式粒度比 Responses API 更细（千问有 13 个 delta vs 2 个），
所以拆字后的显示更平滑。`agent/loop.py` 用累积器把分片的 `delta.tool_calls` 拼成完整调用。

注意：**工具调用的执行时机没变**——仍然等本轮模型输出收齐后才执行工具，再继续下一轮。
所以流式只会加速“文字出现”的速度，不会让工具提前执行。

### 9.2 框架版：LangGraph 的 stream_mode

```python
for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": user_text}]},
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="messages",
):
    yield chunk.content
```

`stream_mode="messages"` 产出 `(消息块, 元数据)` 二元组，消息块是 `AIMessageChunk`。
**注意工具结果也是消息块**（`ToolMessageChunk`），所以要先按类型过滤：

```python
if not isinstance(chunk, AIMessageChunk):
    continue   # 工具结果不混进回答里
```

LangGraph 还有其他模式：`"updates"`（按节点汇报）、`"values"`（每步完整状态）。

### 9.3 三套实现对照

| 维度 | 手写版 | LangChain 版 | LangGraph 原生版 |
|---|---|---|---|
| 底层接口 | `client.responses.stream()` | `graph.stream()` | `graph.stream()` |
| 事件归一化 | 自己写 `StreamEvent` | 框架已归一化 | 框架已归一化 |
| 过滤工具结果 | 事件类型天然区分 | `isinstance` 判断 | `isinstance` 判断 |
| 模型需流式 | 不需要（stream 模式自带） | `ChatOpenAI(streaming=True)` | `ChatOpenAI(streaming=True)` |
| 工具执行时机 | 本轮收齐后 | 本轮收齐后 | 本轮收齐后 |

### 9.4 一个容易踩的坑

框架版的 `run_turn()`（非流式）**没有**改成流式的收集版，仍走 `invoke()` 取最后一条消息。
原因是流式会把中间轮的模型文本也吐出来，与“只要最终回答”的语义不一致。

## 官方文档

- [OpenAI API Quickstart](https://developers.openai.com/api/docs/quickstart)
- [Function Calling Guide](https://developers.openai.com/api/docs/guides/function-calling)
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)