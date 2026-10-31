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

项目默认使用 OpenAI **Responses API**（`/v1/responses`）与 Function Calling，
也可以通过 `OPENAI_API_STYLE=chat` 切到 **Chat Completions**（`/v1/chat/completions`）。
两种 API 的差异收敛在 `agent/backends.py`，Agent 循环本身不做区分。
包含安全的算术工具、当前时间工具、多轮对话与自动化测试。

三套实现（手写版、LangChain 版、LangGraph 原生版）默认走**同一条** API 路径，
切到 chat 模式时也会同步切换，可以直接横向对比。

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
│   ├── config.py           # 环境变量加载、模型名、API 风格、最大步数、系统指令
│   ├── tools.py            # TOOLS 清单 + calculate / get_current_time / execute_tool
│   ├── backends.py         # API 适配层：Responses / Chat Completions 两种后端
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
    ├── live.py             # 真实调用网关的公共辅助
    ├── test_agent.py       # 手写版循环测试
    ├── test_backends.py    # 两种 API 适配层测试
    ├── test_lc.py          # LangChain 版测试
    └── test_lg.py          # LangGraph 原生版测试
```

## 1. 安装依赖

建议使用 Python 3.10 或更高版本。

```powershell
cd <仓库目录>
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果还没有虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不要向 uv 管理的全局 Python 安装依赖，也不要使用 `--break-system-packages`。

依赖包含三套实现的全部需要：`openai`（手写版）、`langchain` / `langchain-openai` / `langgraph`（两个框架版）。

## 2. 配置 API Key

复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
OPENAI_API_KEY=你的 API Key
OPENAI_MODEL=MiniMax-M3.1-Flash

# API 风格：responses（默认）或 chat
OPENAI_API_STYLE=responses
```

如果网关只支持 Chat Completions，把 `OPENAI_API_STYLE` 改成 `chat`；
手写版和两个框架版会一起切换。

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
<仓库目录>\.venv\Scripts\python.exe
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

共 72 个测试：`test_agent.py` 覆盖手写版循环，`test_backends.py` 覆盖两种 API 适配器，
`test_lc.py` 覆盖 LangChain 版，`test_lg.py` 覆盖 LangGraph 原生版。
其中 25 个会**真实调用**网关：需先在 `.env` 里配好 `OPENAI_API_KEY` 与
`OPENAI_BASE_URL`（指向提供该模型的 OpenAI 兼容网关），缺配置会直接失败而不是跳过。
需要构造特定响应形态（失败、截断）的用例用标准库 `unittest.mock` 打桩，不引入额外依赖。

## 6. 核心循环

核心循环在 `agent/loop.py` 的 `agent_loop()`，它不依赖任何全局配置，模型请求和工具执行都靠参数注入：

```python
for _ in range(max_steps):
    turn = request()                 # 请求模型，返回 Turn
    history.append(...)              # 记录 assistant 输出
    if not turn.tool_calls:          # 没有工具调用 = 模型答完了
        return turn.text

    for call in turn.tool_calls:     # 执行工具并回传结果
        result = run_tool_call(call)
        history.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
```

`run_agent_turn()` 只是薄封装：根据 `OPENAI_API_STYLE` 选择后端（`build_backend`），
把 `MODEL` / `SYSTEM_PROMPT` / `TOOLS` 交给后端，并负责出错时回滚历史。

历史使用厂商无关的规范格式：`user`、`assistant`（可带 `tool_calls`）、`tool`。
发给模型前由 `agent/backends.py` 转换成对应 API 的消息结构。

如果没有工具调用，`turn.text` 就是最终回答；否则把工具结果追加到历史，再次请求模型，直到模型给出最终回答或达到最大轮数。

## 7. 添加工具

需要同时修改两处：

1. 在 `TOOLS` 中增加 JSON Schema。
2. 在 `execute_tool()` 中增加同名处理分支。

`TOOLS` 用的是 Responses API 的**扁平**格式，`name` / `description` / `parameters` 直接挂在顶层。
Chat Completions 需要的嵌套格式由 `to_chat_tools()` 自动转换，新增工具时不用写两份：

```python
{
    "type": "function",
    "name": "calculate",
    "description": "安全地计算一个数学表达式……",
    "parameters": {"type": "object", "properties": {...}, "required": [...]},
}
```

框架版不手写 schema——`@tool` 装饰器会从函数签名和 docstring 自动生成，
只需在 `agent_lc/tools.py` 加一个包装函数即可。

生产环境还应增加权限控制、超时、重试、日志、人工确认和成本上限。

## 8. 框架对照版

项目里有三套实现，跑同一个模型、同一套工具、同一份系统提示词和同一个 API 风格，可以直接横向对比。

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

默认走 Responses API 时，模型客户端这样构造：

```python
model = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    use_responses_api=True,        # chat 模式下为 False
    output_version="responses/v1", # 仅 Responses 模式设置
    streaming=True,
)
```

`agent_lc/graph.py` 会把这两项都交给 `OPENAI_API_STYLE` 决定：
chat 模式下 `use_responses_api=False`，也不再设置 `output_version`。

多轮记忆由 checkpointer 按 `thread_id` 自动维护，每次只需传入本轮消息：

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": user_text}]},
    config={
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    },
)
answer = message_text(result["messages"][-1])   # 见下方说明
```

**注意 `content` 的格式差异**：Responses API 下（`output_version="responses/v1"`）
`AIMessage.content` 是**内容块列表**而不是字符串：

```python
[{"type": "reasoning", "summary": [], "id": "rs_..."},
 {"type": "text", "text": "答案是 949", "index": 1}]
```

所以取纯文本要用 `message.text`（对两种格式都兼容），
或者像本项目那样写个 `message_text()` 辅助函数自己抽。

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
| 走的 API 路径 | 默认 `/v1/responses`，可切 `/v1/chat/completions` | 同左 | 同左 |
| 命令 | `minimal-agent` | `minimal-agent-lc` | `minimal-agent-lg` |

三套实现生成的图结构完全一致，都是：

```text
__start__ -> model
model     -> __end__    (条件边)
model     -> tools      (条件边)
tools     -> model      (回边)
```

### 8.4 三处有意的行为差异

1. **步数上限是近似对齐**：LangGraph 的 `recursion_limit` 计数的是「图超级步」（一次模型请求 + 一次工具执行约 2 步），而 `MAX_STEPS` 语义是「模型请求次数」，所以取 `MAX_STEPS * 2 = 16`，两边大致相当但非严格等价。另外 `create_agent` 自己默认把 `recursion_limit` 设成 9999，所以框架版必须显式传值才能保住成本上限。
2. **工具结果格式**：手写版用 `json.dumps(result, ensure_ascii=False)`，框架版由 `ToolNode` 序列化。两者都是 JSON 文本，结构一致。
3. **工具 schema 格式**：手写版直接写 Responses API 的扁平 schema；框架版由 `@tool` 装饰器从函数签名自动生成，再由 LangChain 转成 API 需要的格式。底层结果一致，但生成方式不同。

三套实现的**循环结构**和**API 端点**完全一致（请求模型 → 执行工具 → 回传结果 → 再请求），
差异只剩「循环谁来写」。

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

### 9.1 手写版：Responses API 的 stream 模式

```python
stream = client.responses.create(
    model=MODEL, instructions=SYSTEM_PROMPT,
    input=to_responses_input(history), tools=TOOLS, stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        yield event.delta
    elif event.type == "response.completed":
        final_response = event.response   # 完整响应，含 output
```

以上逻辑现在位于 `agent/backends.py` 的 `ResponsesBackend.request_stream()`。

Responses API 的流式事件是**结构化**的：文本增量和工具调用分属不同事件类型，
不需要像 Chat Completions 那样自己累积 `delta.tool_calls` 分片。
常用事件：

| 事件类型 | 含义 |
|---|---|
| `response.output_text.delta` | 文本增量，带 `.delta` |
| `response.completed` | 正常收尾，带完整 `.response` |
| `response.failed` | 失败，带 `.response.error`（`.code` / `.message`） |
| `response.incomplete` | 输出被截断，同样带 `.response.error` |
| `error` | 网关级错误，错误信息直接在事件上 |

**失败事件必须单独处理**：SDK 只在 HTTP 错误码时抛异常，
`HTTP 200 + status=failed` 会正常返回一个 `output` 为空的响应。
不检查的话会拿到空回答，看起来像"模型什么都没说"。
`agent/backends.py` 的 `_raise_if_failed()`（非流式）和 `_describe_failure()`（流式）负责这件事。

注意：**工具调用的执行时机没变**——仍然等本轮模型输出收齐后才执行工具，再继续下一轮。
所以流式只会加速"文字出现"的速度，不会让工具提前执行。

### 9.2 框架版：LangGraph 的 stream_mode

```python
for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": user_text}]},
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="messages",
):
    yield message_text(chunk)
```

`stream_mode="messages"` 产出 `(消息块, 元数据)` 二元组，消息块是 `AIMessageChunk`。
**注意工具结果也是消息块**（`ToolMessageChunk`），所以要先按类型过滤：

```python
if not isinstance(chunk, AIMessageChunk):
    continue   # 工具结果不混进回答里
```

另外 Responses API 下 `chunk.content` 同样是内容块列表，
所以取文本要走 `message_text(chunk)` 而不是直接用 `chunk.content`。

LangGraph 还有其他模式：`"updates"`（按节点汇报）、`"values"`（每步完整状态）。

切到 Chat Completions 时，手写版由 `ChatCompletionsBackend` 累积 `delta.content`
和分片的 `delta.tool_calls`；框架版仍由 LangChain 内部处理，
`message_text()` 已经同时兼容字符串和内容块列表两种 content。

### 9.3 三套实现对照

| 维度 | 手写版 | LangChain 版 | LangGraph 原生版 |
|---|---|---|---|
| 底层接口 | 后端内 `responses` / `chat.completions` 的流式接口 | `graph.stream()` | `graph.stream()` |
| API 端点 | 默认 `/v1/responses`，可切 `/v1/chat/completions` | 同左 | 同左 |
| 事件来源 | Responses 结构化事件 / Chat 的 `delta` 分片 | `AIMessageChunk` | `AIMessageChunk` |
| 提取纯文本 | 事件自带 `.delta` | `message_text()`（content 是块列表） | `message_text()`（content 是块列表） |
| 过滤工具结果 | 只 yield 文本增量事件，工具结果天然不进来 | `isinstance` 判断 | `isinstance` 判断 |
| 模型需流式 | 不需要（stream 模式自带） | `ChatOpenAI(streaming=True)` | `ChatOpenAI(streaming=True)` |
| 工具执行时机 | 本轮收齐后 | 本轮收齐后 | 本轮收齐后 |

### 9.4 一个容易踩的坑

框架版的 `run_turn()`（非流式）**没有**改成流式的收集版，仍走 `invoke()` 取最后一条消息。
原因是流式会把中间轮的模型文本也吐出来，与“只要最终回答”的语义不一致。

## 10. 双 API 支持

### 10.1 切换方式

在 `.env` 里设置：

```dotenv
OPENAI_API_STYLE=responses   # 默认，走 /v1/responses
# OPENAI_API_STYLE=chat      # 走 /v1/chat/completions
```

三套实现共用这个开关：手写版通过 `build_backend()` 选择适配器，
两个框架版把它映射成 `ChatOpenAI(use_responses_api=...)`。

### 10.2 适配层结构

`agent/backends.py` 把两种 API 的差异收敛成统一契约：

- `Turn` / `ToolCall`：模型一轮输出（文本 + 工具调用）。
- `TextDelta` / `Completed` / `Failed`：流式事件。
- 规范历史：`user` / `assistant`（可带 `tool_calls`）/ `tool`。

循环只消费这些统一结构，代码里不出现任何 API 专属字段。

| 差异点 | Responses 适配器 | Chat Completions 适配器 |
|---|---|---|
| 端点 | `client.responses.create` | `client.chat.completions.create` |
| 系统提示 | `instructions=` | 消息列表首条 `system` |
| 工具 schema | 扁平 `name` / `parameters` | 嵌套 `function: {name, parameters}` |
| 工具调用 | `function_call` item + `call_id` | `message.tool_calls[].function` |
| 工具结果 | `function_call_output` item | `role=tool` + `tool_call_id` |
| 流式文本 | `response.output_text.delta` | `delta.content` |
| 流式工具调用 | 结构化 item，直接可取 | 按 `index` 累积 `delta.tool_calls` |
| 失败信号 | `response.failed` / `incomplete` | HTTP 异常 + `finish_reason=length` |

### 10.3 约束

- 不做自动探测：配置错误直接报错，而不是偷偷回退到另一种 API。
- 新增工具仍然只改 `TOOLS` 和 `execute_tool()`，Chat 格式由适配层转换。
- 两种模式使用同一套规范历史，切换 API 不需要清空已有上下文。

## 官方文档

- [OpenAI API Quickstart](https://developers.openai.com/api/docs/quickstart)
- [Function Calling Guide](https://developers.openai.com/api/docs/guides/function-calling)
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)