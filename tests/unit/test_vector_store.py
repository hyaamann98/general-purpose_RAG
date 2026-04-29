"""
Unit tests for src/vector_store.py (ChromaStore)
"""
import numpy as np
import pytest
from langchain_core.documents import Document


def _make_docs_and_embeddings(n: int = 5, dim: int = 768):
    """Generate n random unit-norm Documents + embeddings."""
    docs = [Document(page_content=f"テストドキュメント {i}", metadata={"source": f"doc_{i}.txt"}) for i in range(n)]
    raw = np.random.rand(n, dim).astype(np.float32)
    # Normalize to unit norm so cosine = dot product
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = (raw / norms).tolist()
    return docs, embeddings


def test_add_and_search_returns_k_docs(temp_chroma_store, embedding_model):
    docs, embeddings = _make_docs_and_embeddings(10)
    temp_chroma_store.add(docs, embeddings)

    query_embedding = embedding_model.embed_query("テスト")
    results = temp_chroma_store.search(query_embedding, k=3)
    assert len(results) == 3


def test_threshold_filters_all_unrelated_docs(temp_chroma_store, embedding_model):
    """With an extremely tight threshold (0.0 = cosine distance), nothing passes."""
    docs, embeddings = _make_docs_and_embeddings(5)
    temp_chroma_store.add(docs, embeddings)

    query_embedding = embedding_model.embed_query("全く関係のない質問")
    results = temp_chroma_store.search(query_embedding, k=5, threshold=0.0)
    # cosine distance > 0.0 for any non-identical pair, so all should be filtered
    assert results == []


def test_duplicate_detection_via_md5(temp_chroma_store, embedding_model):
    """Adding the same chunk twice should not increase the collection count."""
    doc = Document(page_content="重複テスト用ドキュメント", metadata={"source": "dup.txt"})
    embedding = embedding_model.embed_query(doc.page_content)

    temp_chroma_store.add([doc], [embedding])
    temp_chroma_store.add([doc], [embedding])  # second add should be skipped

    ids = temp_chroma_store.get_stored_ids()
    assert len(ids) == 1


def test_get_stored_ids_returns_set(temp_chroma_store, embedding_model):
    docs, embeddings = _make_docs_and_embeddings(3)
    temp_chroma_store.add(docs, embeddings)
    ids = temp_chroma_store.get_stored_ids()
    assert isinstance(ids, set)
    assert len(ids) == 3


def test_search_with_k_larger_than_collection(temp_chroma_store, embedding_model):
    """Requesting k=10 when only 3 docs exist should return all 3."""
    docs, embeddings = _make_docs_and_embeddings(3)
    temp_chroma_store.add(docs, embeddings)

    query_embedding = embedding_model.embed_query("テスト")
    results = temp_chroma_store.search(query_embedding, k=10)
    assert len(results) == 3
