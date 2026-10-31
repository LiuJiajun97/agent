# AGENTS.md

最小 Python Agent 原型：同一套 Agent 循环的**三份并列实现**（手写 OpenAI SDK / LangChain `create_agent` / LangGraph 原生 `StateGraph`），支持 Responses API 与 Chat Completions 两种 Function Calling 风格。

## Setup commands

- Install deps: `pip install -r requirements.txt`（或 `pip install -e .` 安装为包）
- Run:         `python -m agent.cli "你好"` / `python -m agent_lc.cli` / `python -m agent_lg.cli`
- 单次问答:     `python -m agent_lg.cli "计算 123 * 45"`
- 交互 REPL:    `python -m agent_lg.cli`（`exit` 退出，`clear` 清空上下文）
- Test:        `python -m unittest discover -s tests -v`

无 lint / typecheck 工具链（未安装 ruff、mypy、pytest），质量靠 unittest + 代码评审保障。

## Project layout

- `agent/` — 手写版：不用 Agent 框架，直接用 OpenAI SDK 跑循环。`config.py` 集中定义模型名、API 风格、步数等配置
- `agent/backends.py` — API 适配层：Responses / Chat Completions 两种后端，循环本身不区分风格
- `agent/loop.py` — `agent_loop` 核心循环与流式变体
- `agent_lc/` — LangChain 版：`create_agent` 封装 + checkpointer 多轮记忆
- `agent_lg/` — LangGraph 原生版：手写 `StateGraph` 节点与条件边，把 `create_agent` 内部摊开
- `tests/` — unittest 用例，按实现分文件（`test_agent.py` / `test_backends.py` / `test_lc.py` / `test_lg.py`）
- `tests/live.py` — 真实调用网关的公共辅助（检查配置、建客户端、解析工具参数）
- `.env.example` — 配置模板；`.env` 已被 `.gitignore` 忽略

## 三套实现的对应关系

改行为时**三处要同步**，这是本项目的核心目的：

| 概念 | `agent/` | `agent_lc/` | `agent_lg/` |
|---|---|---|---|
| 核心循环 | `loop.agent_loop` | `create_agent` 生成 | `graph.py` 的 `model`/`tools` 节点 + 条件边 |
| 对话历史 | `history` 列表 | checkpointer | `AgentState.messages` + `add_messages` reducer |
| 步数上限 | `MAX_STEPS` | `RECURSION_LIMIT` | `RECURSION_LIMIT` |
| 工具 | `agent/tools.py` | `agent_lc/tools.py`（`@tool` 包装） | 复用 `agent_lc` 的工具 |

## 配置与环境变量

- 变量只有四个：`OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_API_STYLE` / `OPENAI_BASE_URL`
- 模型名 / API 风格 / 步数等由 `agent/config.py` 统一定义；`OPENAI_API_KEY` / `OPENAI_BASE_URL` 由 `agent/cli.py` 与 `agent_lc/graph.py` 直接读取（需显式传给 SDK）
- `OPENAI_API_STYLE` 只允许 `responses` 或 `chat`，切换时三套实现同步生效
- **`OPENAI_BASE_URL` 必须显式设为实际提供该模型的 OpenAI 兼容网关**；不设会回落到 `https://api.openai.com/v1`，用网关令牌请求必然 401，且报错文案指向 OpenAI 官网，极易误判成"key 失效"
- 网关地址与令牌只写在本地 `.env`，不要提交

## Code style

- 模块顶部中文 docstring 讲清"为什么这么写"，而不是复述代码
- `from __future__ import annotations` + 现代类型标注（`dict[str, Any]`、`str | None`）
- 注释解释 why，不解释 what；关键取舍（如"为什么不用 eval"）写清原因
- 每个包用 `__all__` 显式导出公共 API
- 函数/类用中文 docstring，必要时列出参数含义
- 保持 Windows 兼容：中文输出路径要处理 stdout 编码

## Testing instructions

- 单元测试：`python -m unittest discover -s tests -v`（unittest，不是 pytest；当前 72 个用例，其中 25 个会真实打网关，整轮约 1–2 分钟）
- 测试会**真实调用**网关：需 `.env` 里配好 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`（指向提供该模型的 OpenAI 兼容网关）；缺配置直接失败，**不 skip**
- 需要构造特定响应形态（失败、截断、非法参数）的分支用标准库 `unittest.mock` 打桩 SDK 方法，不引入额外依赖；纯逻辑用例不需要网络
- 新增行为必须补测试，放进对应实现的 `tests/test_<模块>.py`
- 三套实现的行为变更要在三处都补，避免横向对比失衡
- 提交前测试必须全绿

## PR & commit conventions

- 从 `master` 分支切出，不要直接推 `master`
- 提交信息用 conventional commits：`feat:` / `fix:` / `docs:` / `refactor:`（现有历史均为 `feat:`）
- 一条提交只做一件事；跨实现的同步改动放同一提交，便于对照
- 远程仓库：`github.com/LiuJiajun97/agent`（`origin` 走 ghfast.top 镜像）

## Security

- **绝不提交密钥**：`.env` 已在 `.gitignore`，仓库里只保留 `.env.example` 模板
- 本地令牌是自建网关的令牌，不是 OpenAI 官方 key；不要写进代码、注释、测试断言或日志
- `calculate` 工具用 AST 白名单求值，**禁止改用 `eval` / `exec`**；已有限制：表达式长度、指数绝对值、结果位数、非有限浮点
- 工具执行异常要包装成 `{"ok": False, "error": ...}` 回传给模型，而不是直接抛出中断 Agent
