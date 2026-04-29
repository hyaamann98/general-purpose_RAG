import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .dataload import DataLoad
from .make_chunk import MakeChunk
from .embedding import Embedding
from .vector_store import ChromaStore
from .llm import OllamaLLM
from .retriever import Retriever

app = FastAPI(title="RAG API")

# 環境変数から設定を読み込む
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs_storage")

# 起動時にコンポーネントを初期化（重いモデルを一度だけロード）
embedding = Embedding(model_name=EMBED_MODEL)
vector_store = ChromaStore(persist_directory=CHROMA_DIR)
llm = OllamaLLM(model=LLM_MODEL)
retriever = Retriever(embedding=embedding, vector_store=vector_store, llm=llm)


class QueryRequest(BaseModel):
    question: str
    k: int = 5
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
