from abc import ABC, abstractmethod
from langchain_core.documents import Document
import chromadb
import faiss
import numpy as np
import pickle
import hashlib
import os


class VectorStoreBase(ABC):
    @abstractmethod
    def add(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], k: int = 5) -> list[Document]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def get_stored_ids(self) -> set[str]:
        pass


class ChromaStore(VectorStoreBase):
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Client saving datas into designed dir
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        ids = [
            hashlib.md5((chunk.metadata.get("source", "") + chunk.page_content).encode()).hexdigest()
            for chunk in chunks
        ]
        existing_ids = self.get_stored_ids()
        new_items = [
            (id_, chunk, emb)
            for id_, chunk, emb in zip(ids, chunks, embeddings)
            if id_ not in existing_ids
        ]
        if not new_items:
            print("  -> 新規チャンクなし。スキップします。")
            return
        new_ids, new_chunks, new_embeddings = zip(*new_items)
        self.collection.add(
            ids=list(new_ids),
            embeddings=list(new_embeddings),
            documents=[chunk.page_content for chunk in new_chunks],
            metadatas=[chunk.metadata for chunk in new_chunks],
        )
        print(f"  -> {len(new_ids)} 件追加（{len(chunks) - len(new_ids)} 件スキップ）")

    def search(self, query_embedding: list[float], k: int = 5, threshold: float = None) -> list[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if threshold is not None and distance > threshold:
                continue
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def save(self, path: str) -> None:
        # PersistentClient は自動で永続化するため不要だが、インターフェース維持のため残す
        pass

    def load(self, path: str) -> None:
        # PersistentClient は起動時に自動ロードするため不要だが、インターフェース維持のため残す
        pass

    def get_stored_ids(self) -> set[str]:
        result = self.collection.get()
        return set(result["ids"])


class FAISSStore(VectorStoreBase):
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = index_path
        self.index = None        # FAISSインデックス本体
        self.documents: list[Document] = []  # ベクトルに対応するドキュメントを保持

    def add(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.documents.extend(chunks)

    def search(self, query_embedding: list[float], k: int = 5) -> list[Document]:
        query_vec = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(query_vec, k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

    def get_stored_ids(self) -> set[str]:
        return {doc.metadata.get("id", str(i)) for i, doc in enumerate(self.documents)}
