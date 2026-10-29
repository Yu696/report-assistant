# 智能报告助手

基于 **LangChain + 在线大模型** 的智能文档处理与知识库问答应用。面向学生和文字工作者，解决「报告/作业要写、要改、要总结，还想针对一堆资料提问」的需求。

前后端一体：FastAPI 提供接口，浏览器打开即用，无需安装客户端。

## 功能特性

### 文本处理
- **总结**：长文提炼成要点
- **润色**：让表达更通顺、更书面化
- **扩写**：把内容写得更充实、更具体
- **校对**：找错别字、语病、标点错误
- **文件上传**：支持 `txt / md / pdf / docx`，自动提取文字
- **流式输出**：结果逐字返回（打字机效果）

### 知识库问答（RAG）
- **多文件入库**：上传多份资料，自动切片并向量化存储
- **检索式问答**：答案基于资料内容，尽量给出处
- **多轮对话**：记住上下文，可以连续追问
- **流式输出**：回答逐字返回

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| AI 编排 | LangChain（Prompt / Loader / TextSplitter / VectorStore） |
| 大模型 | DeepSeek（`deepseek-chat`） |
| 嵌入模型 | SiliconFlow `BAAI/bge-m3` |
| 向量数据库 | Chroma |
| 前端 | 原生 HTML + CSS + JavaScript（单页，SSE 流式） |

## 项目结构

```
report-assistant/
├── main.py            # 后端：FastAPI 路由 + LangChain 逻辑
├── static/
│   └── index.html     # 前端：双标签页 + 流式渲染
├── requirements.txt   # 依赖清单
├── .env               # 环境变量（密钥，勿提交）
├── .env.example       # 环境变量模板
└── README.md
```

## 快速开始

### 1. 获取 API Key

- **DeepSeek**（必填）：https://platform.deepseek.com/ ，创建 API Key
- **SiliconFlow**（选填，仅知识库问答需要）：https://siliconflow.cn/ ，创建 API Key

### 2. 配置环境变量

复制模板并填入密钥：

```bash
cp .env.example .env
```

`.env` 内容：

```
DEEPSEEK_API_KEY=sk-你的deepseek密钥
SILICONFLOW_API_KEY=sk-你的siliconflow密钥   # 不填则知识库功能不可用
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 国内建议使用镜像源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 4. 启动服务

```bash
uvicorn main:app --reload
```

### 5. 打开网页

浏览器访问 http://127.0.0.1:8000

## 使用说明

1. **文本处理**：粘贴文字或上传文件 → 选择模式（总结/润色/扩写/校对）→ 点「开始处理」
2. **知识库问答**：上传一份或多份资料 → 等入库完成 → 在输入框提问（可追问）

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/process` | 流式文本处理，参数 `mode` + `text` |
| POST | `/api/upload` | 单文件上传，返回提取的文字 |
| POST | `/api/rag/ingest` | 多文件上传并建立知识库 |
| POST | `/api/chat` | 流式 RAG 问答，参数 `question` |
| POST | `/api/reset` | 清空对话与知识库 |
| GET | `/api/status` | 查询知识库是否可用 |

## 常见问题

**Q：提示「未配置 SILICONFLOW_API_KEY」？**
A：知识库问答需要嵌入模型。去 siliconflow.cn 注册拿 key，填入 `.env` 后**重启服务**（`.env` 只在启动时读取）。

**Q：上传文件报「不是有效 JSON」？**
A：这是后端返回 500 错误导致的误报，与文件格式无关。本项目已修复嵌入模型的 tiktoken 境外下载问题，升级到最新代码即可。

**Q：改了 `.env` 为什么不生效？**
A：`uvicorn --reload` 只监听 `.py` 文件，不会重读 `.env`。改完 `.env` 需手动重启服务。

## 说明

- 对话与知识库保存在内存中，重启服务后清空
- 文本处理只需要 DeepSeek，知识库问答额外需要 SiliconFlow
