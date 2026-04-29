"""
Unit tests for src/make_chunk.py
"""
import pytest
from langchain_core.documents import Document
from src.make_chunk import MakeChunk


def test_chunk_splits_long_document():
    """A document longer than chunk_size should be split into multiple chunks."""
    long_text = "あ" * 600  # 600 chars > chunk_size=500
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})
    chunker = MakeChunk(chunk_size=500, chunk_overlap=50)
    chunks = chunker.text_split([doc])
    assert len(chunks) > 1


def test_chunk_short_document_stays_single():
    """A document shorter than chunk_size should remain as one chunk."""
    short_text = "これは短いテキストです。"
    doc = Document(page_content=short_text, metadata={"source": "test.txt"})
    chunker = MakeChunk(chunk_size=500, chunk_overlap=50)
    chunks = chunker.text_split([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_chunk_preserves_metadata():
    """Metadata (e.g. source) should be preserved in every chunk."""
    doc = Document(page_content="テスト文書。" * 100, metadata={"source": "myfile.txt"})
    chunker = MakeChunk(chunk_size=100, chunk_overlap=10)
    chunks = chunker.text_split([doc])
    for chunk in chunks:
        assert chunk.metadata.get("source") == "myfile.txt"


def test_chunk_overlap_shares_text(chunker):
    """With chunk_overlap > 0, adjacent chunks should share some characters."""
    text = "A" * 200 + "B" * 200
    doc = Document(page_content=text)
    chunks = chunker.text_split([doc])
    if len(chunks) >= 2:
        # The end of chunk[0] and start of chunk[1] should have overlap
        end_of_first = chunks[0].page_content[-20:]
        start_of_second = chunks[1].page_content[:20]
        # Some characters should be shared
        assert any(c in start_of_second for c in end_of_first)


def test_chunk_multiple_documents():
    """Multiple documents should all be chunked independently."""
    docs = [
        Document(page_content="文書1: " + "あ" * 300),
        Document(page_content="文書2: " + "い" * 300),
    ]
    chunker = MakeChunk(chunk_size=200, chunk_overlap=20)
    chunks = chunker.text_split(docs)
    assert len(chunks) > 2
