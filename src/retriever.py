from langchain_core.documents import Document
from .embedding import Embedding
from .vector_store import ChromaStore
from .llm import OllamaLLM


class Retriever:
    def __init__(
        self,
        embedding: Embedding,
        vector_store: ChromaStore,
        llm: OllamaLLM,
        k: int = 5,
        threshold: float = None,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.threshold = threshold

    def _build_context(self, docs: list[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"以下のコンテキストを参考に質問に答えてください。\n\n"
            f"コンテキスト:\n{context}\n\n"
            f"質問: {question}\n"
            f"回答:"
        )

    def query(self, question: str) -> str:
        answer, _ = self.query_with_sources(question)
        return answer

    def query_with_sources(self, question: str) -> tuple[str, list[Document]]:
        query_embedding = self.embedding.embed_query(question)
        docs = self.vector_store.search(query_embedding, k=self.k, threshold=self.threshold)
        if not docs:
            return "関連するドキュメントが見つかりませんでした。", []
        context = self._build_context(docs)
        prompt = self._build_prompt(question, context)
        answer = self.llm.generate(prompt)
        return answer, docs