"""
Integration tests for src/api.py FastAPI endpoints.
These tests use FastAPI's TestClient, which wires through the full request/response
cycle without starting a real server.

NOTE: Tests marked with @pytest.mark.requires_ollama need Ollama running locally.
Run them with:  pytest tests/integration/ -m requires_ollama
Skip them with: pytest tests/integration/ -m "not requires_ollama"
"""
import io
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Patch environment before importing api to avoid loading production chroma_db
os.environ.setdefault("CHROMA_DIR", "/tmp/chroma_test_api")
os.environ.setdefault("DOCS_DIR", "/tmp/docs_test_api")

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_empty_question_returns_400():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


def test_query_whitespace_question_returns_400():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


def test_upload_invalid_extension_returns_400():
    file_content = b"This is a Word document content"
    response = client.post(
        "/upload",
        files={"file": ("test.docx", io.BytesIO(file_content), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_txt_file_returns_200(tmp_path):
    os.environ["DOCS_DIR"] = str(tmp_path / "docs")
    file_content = "テストファイルの内容です。日本語テキストが含まれています。".encode("utf-8")
    response = client.post(
        "/upload",
        files={"file": ("test_upload.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    assert "test_upload.txt" in response.json().get("message", "")


@pytest.mark.requires_ollama
def test_query_with_uploaded_doc_returns_answer_and_sources(tmp_path):
    """Requires Ollama running. Uploads a doc, then queries for content in it."""
    os.environ["DOCS_DIR"] = str(tmp_path / "docs")

    content = "富士山は日本で最も高い山で、標高は3776メートルです。".encode("utf-8")
    client.post(
        "/upload",
        files={"file": ("fuji.txt", io.BytesIO(content), "text/plain")},
    )

    response = client.post("/query", json={"question": "富士山の標高は何メートルですか？", "k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
