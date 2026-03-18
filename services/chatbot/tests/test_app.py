"""Tests for the FastAPI app — retriever and LLM are mocked."""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.search.return_value = [
        {
            "score": 0.85,
            "chunk_id": "abc_0000",
            "text": "Sally Hamouda is a professor of CS.",
            "url": "https://website.cs.vt.edu/faculty/sally.html",
            "title": "Sally Hamouda | CS | VT",
            "headings_path": ["Faculty"],
            "page_type": "faculty",
        }
    ]
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ask.return_value = "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    return llm


@pytest.fixture
def client(mock_retriever, mock_llm):
    with patch("chatbot.app.retriever", mock_retriever), \
         patch("chatbot.app.llm_client", mock_llm):
        from chatbot.app import app
        yield TestClient(app)


def test_ask_endpoint(client, mock_retriever, mock_llm):
    response = client.post("/ask", json={"question": "Who is Sally Hamouda?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["url"] == "https://website.cs.vt.edu/faculty/sally.html"


def test_ask_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
