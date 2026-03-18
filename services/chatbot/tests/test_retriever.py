"""Tests for the retriever module — model and Qdrant are mocked."""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from chatbot.retriever import Retriever, BGE_QUERY_PREFIX


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 1024
    model.encode.side_effect = lambda text, **kw: np.random.randn(1024).astype(np.float32)
    return model


@pytest.fixture
def mock_qdrant():
    return MagicMock()


@pytest.fixture
def retriever(mock_model, mock_qdrant):
    with patch("chatbot.retriever.SentenceTransformer", return_value=mock_model), \
         patch("chatbot.retriever.QdrantClient", return_value=mock_qdrant), \
         patch("chatbot.retriever.torch.cuda.is_available", return_value=False):
        return Retriever(
            embedding_model="mock-model",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection="test_collection",
            top_k=3,
        )


def test_bge_query_prefix():
    assert BGE_QUERY_PREFIX.startswith("Represent this sentence")


def test_search_returns_results(retriever, mock_qdrant):
    mock_point = MagicMock()
    mock_point.score = 0.85
    mock_point.payload = {
        "chunk_id": "abc_0000",
        "text": "Sally Hamouda is a professor.",
        "url": "https://website.cs.vt.edu/faculty/sally.html",
        "title": "Sally Hamouda | CS | VT",
        "headings_path": ["Faculty"],
        "page_type": "faculty",
    }
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_response

    results = retriever.search("Sally Hamouda")
    assert len(results) == 1
    assert results[0]["score"] == 0.85
    assert results[0]["text"] == "Sally Hamouda is a professor."
    assert results[0]["url"] == "https://website.cs.vt.edu/faculty/sally.html"


def test_search_prepends_bge_prefix(retriever, mock_model, mock_qdrant):
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points.return_value = mock_response

    retriever.search("test query")
    call_args = mock_model.encode.call_args
    assert call_args[0][0] == BGE_QUERY_PREFIX + "test query"


def test_search_empty_results(retriever, mock_qdrant):
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points.return_value = mock_response

    results = retriever.search("nonexistent topic")
    assert results == []
