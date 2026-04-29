"""
Unit tests for src/retriever.py (Retriever)
Uses unittest.mock to avoid requiring Ollama to be running.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.retriever import Retriever
from src.embedding import Embedding
from src.vector_store import ChromaStore


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "モックの回答です。"
    return llm


@pytest.fixture
def retriever_with_docs(temp_chroma_store, embedding_model, mock_llm):
    """Retriever pre-loaded with a few sample documents."""
    docs = [
        Document(page_content="東京は日本の首都です。", metadata={"source": "tokyo.txt"}),
        Document(page_content="大阪は関西の中心都市です。", metadata={"source": "osaka.txt"}),
    ]
    embeddings = embedding_model.embed_documents(docs)
    temp_chroma_store.add(docs, embeddings)
    return Retriever(
        embedding=embedding_model,
        vector_store=temp_chroma_store,
        llm=mock_llm,
        k=3,
    )


def test_query_returns_string(retriever_with_docs):
    answer = retriever_with_docs.query("東京について教えてください。")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_query_with_sources_returns_tuple(retriever_with_docs):
    answer, docs = retriever_with_docs.query_with_sources("東京について教えてください。")
    assert isinstance(answer, str)
    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)


def test_query_empty_store_returns_fallback(temp_chroma_store, embedding_model, mock_llm):
    """If vector store is empty, query should return the Japanese fallback message."""
    retriever = Retriever(
        embedding=embedding_model,
        vector_store=temp_chroma_store,
        llm=mock_llm,
    )
    answer = retriever.query("何か教えてください。")
    assert "関連するドキュメントが見つかりませんでした" in answer
    mock_llm.generate.assert_not_called()


def test_query_with_sources_empty_store_returns_empty_list(temp_chroma_store, embedding_model, mock_llm):
    retriever = Retriever(
        embedding=embedding_model,
        vector_store=temp_chroma_store,
        llm=mock_llm,
    )
    _, docs = retriever.query_with_sources("何か教えてください。")
    assert docs == []


def test_prompt_contains_question(retriever_with_docs, mock_llm):
    """The prompt sent to the LLM should contain the original question."""
    question = "東京タワーの高さは何メートルですか？"
    retriever_with_docs.query(question)
    call_args = mock_llm.generate.call_args
    prompt = call_args[0][0]
    assert question in prompt


def test_context_contains_retrieved_docs(retriever_with_docs, mock_llm):
    """The prompt should include the page_content of retrieved documents."""
    retriever_with_docs.query("東京について")
    call_args = mock_llm.generate.call_args
    prompt = call_args[0][0]
    assert "東京" in prompt
