import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()

if not DEEPSEEK_API_KEY:
    raise RuntimeError("未设置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入你的密钥")

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.7,
    max_tokens=4096,
)

# RAG 嵌入模型（可选）：配置了 SILICONFLOW_API_KEY 才启用知识库功能
embeddings = None
if SILICONFLOW_API_KEY:
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

TEMPLATES = {
    "summarize": (
        "你是中文文档总结助手。请把下面的内容总结成简洁的要点，"
        "保留关键信息，分点列出，用中文输出：\n\n{text}"
    ),
    "polish": (
        "你是中文写作润色助手。请润色下面的文字，让表达更通顺、更书面化、更专业，"
        "保持原意不变，直接输出润色后的文字：\n\n{text}"
    ),
    "expand": (
        "你是中文写作助手。请把下面的内容扩写得更充实、更具体，"
        "补充细节和说明，保持逻辑清晰，直接输出扩写后的文字：\n\n{text}"
    ),
    "proofread": (
        "你是中文校对助手。请检查下面的文字，找出错别字、语病和标点错误，"
        "先简要列出发现的问题，再输出修改后的完整版本：\n\n{text}"
    ),
}

# 单用户本地演示用的全局状态
chat_history = []
vector_store = None
ingested_files = []


def chunk_text(chunk) -> str:
    c = chunk.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def extract_text(file: UploadFile) -> str:
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
            return "\n\n".join(d.page_content for d in docs)
        if suffix == ".docx":
            docs = Docx2txtLoader(tmp_path).load()
            return "\n\n".join(d.page_content for d in docs)
        raw = open(tmp_path, "rb").read()
        for enc in ("utf-8", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")
    finally:
        os.unlink(tmp_path)


def build_chat_messages(question: str):
    messages = []
    if vector_store is not None:
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)
        system = (
            "你是知识库问答助手。请优先根据下面【资料】的内容回答，回答时尽量引用资料；"
            "如果资料中没有相关信息，请明确说明“资料中未找到”。\n\n【资料】\n" + context
        )
    else:
        system = "你是智能问答助手，请结合对话历史回答用户的问题。"

    messages.append(SystemMessage(content=system))
    for m in chat_history[-12:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))
    return messages


class ProcessRequest(BaseModel):
    mode: str
    text: str


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="智能报告助手")


@app.post("/api/process")
async def process(req: ProcessRequest):
    if req.mode not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"不支持的 mode: {req.mode}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    chain = ChatPromptTemplate.from_template(TEMPLATES[req.mode]) | llm

    async def gen():
        async for chunk in chain.astream({"text": req.text}):
            yield f"data: {json.dumps({'text': chunk_text(chunk)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析文件失败: {e}")
    return {"filename": file.filename, "text": text}


@app.post("/api/rag/ingest")
async def rag_ingest(files: list[UploadFile] = File(...)):
    if embeddings is None:
        raise HTTPException(status_code=400, detail="未配置 SILICONFLOW_API_KEY，无法使用知识库功能")

    from langchain_chroma import Chroma

    docs = []
    names = []
    for f in files:
        text = extract_text(f)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": f.filename}))
        names.append(f.filename)

    if not docs:
        raise HTTPException(status_code=400, detail="未能从文件中提取到内容")

    global vector_store, ingested_files
    vector_store = Chroma.from_documents(docs, embedding=embeddings)
    ingested_files = names
    return {"files": names, "chunks": len(docs)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    messages = build_chat_messages(question)

    async def gen():
        full = ""
        async for chunk in llm.astream(messages):
            t = chunk_text(chunk)
            full += t
            yield f"data: {json.dumps({'text': t}, ensure_ascii=False)}\n\n"
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": full})
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset():
    global chat_history, vector_store, ingested_files
    chat_history = []
    vector_store = None
    ingested_files = []
    return {"ok": True}


@app.get("/api/status")
async def status():
    return {"rag_available": embeddings is not None, "files": ingested_files}


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
