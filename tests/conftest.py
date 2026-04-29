"""
Shared pytest fixtures for all tests.
"""
import sys
import os
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from src.embedding import Embedding
from src.vector_store import ChromaStore
from src.make_chunk import MakeChunk


@pytest.fixture(scope="session")
def embedding_model():
    """Load the embedding model once per test session (slow to load)."""
    return Embedding(model_name="intfloat/multilingual-e5-base")


@pytest.fixture
def temp_chroma_store(tmp_path):
    """Temporary ChromaStore that does not pollute the production chroma_db."""
    return ChromaStore(persist_directory=str(tmp_path / "chroma_test"))


@pytest.fixture
def sample_documents():
    """Small, stable Japanese/English documents for deterministic tests."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures", "sample_docs")
    docs = []
    if os.path.isdir(fixtures_dir):
        for fname in sorted(os.listdir(fixtures_dir)):
            fpath = os.path.join(fixtures_dir, fname)
            if fname.endswith(".txt"):
                with open(fpath, encoding="utf-8") as f:
                    docs.append(Document(page_content=f.read(), metadata={"source": fname}))
    return docs


@pytest.fixture
def chunker():
    return MakeChunk(chunk_size=200, chunk_overlap=20)
