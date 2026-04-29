import json
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .dataload import DataLoad
from .make_chunk import MakeChunk
from .embedding import Embedding
from .vector_store import ChromaStore
from .llm import OllamaLLM
from .retriever import Retriever

app = FastAPI(title="RAG API")

# 環境変数から設定を読み込む
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs_storage")

# 起動時にコンポーネントを初期化（重いモデルを一度だけロード）
embedding = Embedding(model_name=EMBED_MODEL)
vector_store = ChromaStore(persist_directory=CHROMA_DIR)
llm = OllamaLLM(model=LLM_MODEL)
retriever = Retriever(embedding=embedding, vector_store=vector_store, llm=llm)


class QueryRequest(BaseModel):
    question: str
    k: int = 3
    threshold: float = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問が空です。")

    retriever.k = req.k
    retriever.threshold = req.threshold

    answer, docs = retriever.query_with_sources(req.question)
    sources = list({doc.metadata.get("source", "不明") for doc in docs})

    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問が空です。")

    retriever.k = req.k
    retriever.threshold = req.threshold

    token_gen, docs = retriever.query_with_sources_stream(req.question)
    sources = list({doc.metadata.get("source", "不明") for doc in docs})

    def generate():
        for token in token_gen:
            yield token
        yield f"__SOURCES__{json.dumps(sources, ensure_ascii=False)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/documents")
def list_documents():
    result = vector_store.collection.get(include=["metadatas"])
    sources = sorted(set(m.get("source", "不明") for m in result["metadatas"]))
    return {"documents": sources, "count": len(sources)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed_extensions = {".txt", ".pdf"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"対応していないファイル形式です: {ext}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    save_path = os.path.join(DOCS_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ドキュメントをロード → チャンク化 → Embedding → DB保存
    dataload = DataLoad(source="local")
    documents = dataload.load(DOCS_DIR)

    chunker = MakeChunk()
    chunks = chunker.text_split(documents)

    embeddings = embedding.embed_documents(chunks)
    vector_store.add(chunks, embeddings)

    return {"message": f"{file.filename} を処理しました。"}
