<div align="center">

# What-if Studio

### 对结局意难平？让 AI 虚拟剧组替你重拍一个平行宇宙。

<p>
  <a href="#快速开始">快速开始</a> ·
  <a href="#核心体验">核心体验</a> ·
  <a href="#架构">架构</a> ·
  <a href="#路线图">路线图</a>
</p>

![Python](https://img.shields.io/badge/Python-3.10%2B-111111?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-111111?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-111111?style=for-the-badge)
![Vite](https://img.shields.io/badge/Vite-Frontend-111111?style=for-the-badge&logo=vite&logoColor=white)
![D3](https://img.shields.io/badge/D3-Network%20Stage-111111?style=for-the-badge&logo=d3.js&logoColor=white)

</div>

---

## 一句话

你输入一句话：

> "《哈利波特与凤凰社》我不接受这个结局，小天狼星必须活下来。"

剩下的交给剧组：导演们开会、争吵、提案、分工，最后给你一段可播放的 HE 平行结局短片。

---

## 核心体验

- **围观一个活的剧组**——导演 Agent 依次发言讨论，不是同时机械开工
- **你随时可以插话**——通过 WebSocket 实时介入讨论，你的意见会被最高优先级响应
- **讨论过程实时可见**——SSE 流式输出每一位导演的发言
- **系统主动问你**——讨论到关键节点时，系统会暂停并向你提问
- **越用越懂你**——跨会话记忆你的风格偏好，自动注入历史剧本参考
- **多种输出可选**——仅剧本（Markdown 下载）/ 剧本+分镜预览 / 剧本+视频

---

## 产品亮点

### 人机共创，而非单模型直出

你不是旁观者。通过底部的干预输入栏，你可以随时注入意见。导演们会**优先回应你的想法**，然后再继续讨论。系统在第二轮讨论后还会主动暂停，向你征求反馈。

### 长期记忆与自我进化

系统在后台自动记录你的风格偏好和历史剧本。每次新对话开始时，会注入你的历史偏好和相似剧本作为参考。使用 TF-IDF 向量检索（无需外部 API），你用得越多，系统越懂你。

### 多导演协作 + LangGraph 编排

守门人负责原著精神，叙事/视觉/声音/素材四位执行导演各司其职。LangGraph StateGraph 编排讨论流程，支持暂停/恢复，分歧时自动抛出问题。

### 分镜预览省钱模式

选"剧本+分镜预览"：先用文本 LLM 生成低成本分镜描述（帧级画面+时长），确认后再调视频模型生成。避免直接调视频 API 浪费额度。

### 可配置模型路由器

文本 LLM 和视频生成都通过统一的 ModelRouter 调用，支持多提供商：DeepSeek / OpenAI / Claude / GLM 等文本模型，Doubao / HappyHorse / Kling / Wan / Grok / 本地 VACE 等视频模型。所有 API Key 通过 `.env` 配置。

---

## 典型流程

1. 输入作品名 + 结局方向 + 风格偏好 + 上传素材视频
2. 剧组开机，导演们依次发言讨论（实时 SSE 流）
3. **随时**通过底部输入栏注入你的意见，或点击暂停
4. 约 12 秒后系统主动提问，征求你的反馈
5. 讨论收敛后选择输出格式：仅剧本 / 剧本+分镜 / 剧本+视频
6. 如果选了分镜：先看帧级预览 → 确认 → 再生成视频
7. 成片弹出播放器，完成分享或二次制作

---

## 架构

```
┌──────────────────────────────────────────────────┐
│  Frontend: Vite + Vanilla JS + D3                │
│  - 网络舞台可视化导演状态                          │
│  - SSE 讨论流实时渲染                              │
│  - WebSocket 干预输入栏                            │
│  - 输出格式选择 + 分镜预览                         │
└──────────────┬───────────────────────────────────┘
               │ /api  (SSE)  │ /ws  (WebSocket)
┌──────────────▼───────────────────────────────────┐
│  Backend: FastAPI + LangGraph + SQLAlchemy        │
│                                                   │
│  src/agents/langgraph_service.py                  │
│  - StateGraph 编排 5 个导演 + 评论家              │
│  - MemorySaver  checkpoint 暂停/恢复              │
│  - 实时用户干预 + 定期主动提问                     │
│                                                   │
│  src/core/model_router.py                         │
│  - TextModelRouter: DeepSeek/OpenAI/Claude/GLM    │
│  - VideoModelRouter: Doubao/Kling/Wan/Grok/本地   │
│                                                   │
│  src/core/memory_service.py                       │
│  - TF-IDF 向量检索 (Chromadb 备选)                │
│  - Mem0 偏好记忆 (JSON 备选)                      │
│  - 跨会话自动注入历史偏好 + 相似剧本               │
└──────────────────────────────────────────────────┘
```

---

## 快速开始

### 1) 安装后端依赖

```bash
uv sync
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

至少填写以下之一（按优先级：DeepSeek → OpenAI → SiliconFlow → Zhipu）：

- `DEEPSEEK_API_KEY`
- 或 `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL`

视频生成（可选）：

- `HAPPYHORSE_API_KEY` / `KLING_API_KEY` / `OPENAI_NEXT_API_KEY` 等

### 3) 构建前端

```bash
cd web
npm install
npm run build
cd ..
```

### 4) 启动服务

```bash
# 后端 (生产模式)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# 前端 (开发模式，含 HMR)
cd web && npx vite --host 0.0.0.0 --port 5180
```

访问：

- App: `http://127.0.0.1:5180`
- API Docs: `http://127.0.0.1:8000/docs`

---

## API 端点一览

### 核心流程
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/projects` | 创建项目 |
| POST | `/api/projects/{id}/script/stream` | SSE 讨论流（`?user_id=` 启用记忆） |
| PUT | `/api/projects/{id}/output/select` | 选择输出格式 |
| POST | `/api/projects/{id}/storyboard/generate` | 生成分镜预览 |
| POST | `/api/projects/{id}/storyboard/confirm` | 确认分镜并创建视频任务 |
| GET | `/api/projects/{id}/script/export` | 导出剧本（markdown/json/txt） |
| GET | `/api/video-jobs/{id}/events` | 视频任务 SSE 进度流 |

### 人机交互
| 方法 | 路径 | 说明 |
|------|------|------|
| WS | `/ws/{session_id}` | WebSocket 干预通道 |
| POST | `/api/sessions/{id}/intervene` | REST 干预（WS 备选） |

### 记忆
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/feedback` | 提交评分/反馈 |

### WebSocket 动作
| action | 说明 |
|--------|------|
| `intervene` | 注入意见到讨论 |
| `pause_now` | 暂停讨论 |
| `resume_now` | 恢复讨论 |

---

## 当前已实现

- LangGraph 多导演讨论编排（替代 AutoGen）
- WebSocket 人机共创（随时插话 + 暂停 + 系统主动提问）
- 跨会话记忆系统（偏好记忆 + TF-IDF 剧本检索）
- 可配置多提供商模型路由器（文本 + 视频）
- SSE 讨论流 + 视频任务进度流
- 输出格式选择：仅剧本 / 剧本+分镜预览 / 剧本+视频
- 分镜预览（低成本文本 LLM）+ 确认后生成视频
- Alembic 数据库迁移（SQLite → PostgreSQL 可选）
- 前端 D3 网络舞台 + Agent 状态可视化
- ANet 网关（可选启用）

---

## 路线图

- [x] LangGraph 替换 AutoGen，支持暂停/恢复
- [x] WebSocket 人机共创（随时插话）
- [x] 跨会话记忆与偏好学习
- [x] 模型路由器（多文本 + 多视频提供商）
- [x] 分镜预览流水线（低成本先看效果）
- [x] 输出格式选择（剧本/分镜/视频）
- [ ] 导演讨论完整回放与检索
- [ ] 社区功能
- [ ] PostgreSQL 生产部署

---

## 适合谁

- 想做"剧情重写 / 平行结局"产品的团队
- 需要"多 Agent 协作 + 人机共创"示例的开发者
- 研究 LangGraph Human-in-the-Loop 模式的工程师
- 想把 AI 从"聊天框"升级到"共创系统"的创作者

---

## 结尾

如果你也有一个"这个结局我不认"的故事，欢迎把它丢给 What-if Studio，让剧组替你拍出来。你不是观众——你是导演。
