"""
Unit tests for src/embedding.py
"""
import numpy as np
import pytest
from langchain_core.documents import Document


def test_embed_query_returns_correct_dimension(embedding_model):
    """multilingual-e5-base outputs 768-dim vectors."""
    vector = embedding_model.embed_query("東京は日本の首都です。")
    assert len(vector) == 768


def test_embed_documents_returns_correct_count(embedding_model):
    docs = [
        Document(page_content="犬は哺乳類の動物です。"),
        Document(page_content="猫もペットとして人気があります。"),
        Document(page_content="自動車は移動手段の一つです。"),
    ]
    embeddings = embedding_model.embed_documents(docs)
    assert len(embeddings) == 3


def test_embed_query_is_deterministic(embedding_model):
    """Same input should produce identical vector."""
    text = "自然言語処理の研究"
    v1 = embedding_model.embed_query(text)
    v2 = embedding_model.embed_query(text)
    np.testing.assert_array_equal(v1, v2)


def test_similarity_japanese_synonyms(embedding_model):
    """'犬' and 'イヌ' should be closer to each other than either is to '自動車'."""
    v_dog_kanji = embedding_model.embed_query("犬")
    v_dog_kana = embedding_model.embed_query("イヌ")
    v_car = embedding_model.embed_query("自動車")

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_dog_dog = cosine(v_dog_kanji, v_dog_kana)
    sim_dog_car = cosine(v_dog_kanji, v_car)
    assert sim_dog_dog > sim_dog_car, (
        f"Expected 犬/イヌ similarity ({sim_dog_dog:.4f}) > 犬/自動車 similarity ({sim_dog_car:.4f})"
    )
