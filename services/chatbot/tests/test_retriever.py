"""Tests for the retriever module — model and Qdrant are mocked."""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from chatbot.retriever import (
    Retriever, BGE_QUERY_PREFIX, contextualize_query,
    _build_follow_up_pattern, DEFAULT_FOLLOW_UP_KEYWORDS,
)


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


def test_search_with_context_uses_enriched_query(retriever, mock_model, mock_qdrant):
    """search_with_context passes the enriched query to the underlying search."""
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points.return_value = mock_response

    history = [{"role": "user", "content": "Who is Sally Hamouda?"}]
    retriever.search_with_context("What about their research?", history)

    encoded_text = mock_model.encode.call_args[0][0]
    assert "Sally Hamouda" in encoded_text
    assert "What about their research?" in encoded_text


def test_contextualize_query_no_history():
    """Without history, query is returned unchanged."""
    pattern = _build_follow_up_pattern(DEFAULT_FOLLOW_UP_KEYWORDS)
    assert contextualize_query("Who is Sally?", [], pattern) == "Who is Sally?"


def test_contextualize_query_with_history():
    """Follow-up question gets the last user message prepended."""
    pattern = _build_follow_up_pattern(DEFAULT_FOLLOW_UP_KEYWORDS)
    history = [
        {"role": "user", "content": "Who is Sally Hamouda?"},
        {"role": "assistant", "content": "Sally Hamouda is a professor of CS."},
    ]
    result = contextualize_query("What about their research?", history, pattern)
    assert "Sally Hamouda" in result
    assert "What about their research?" in result


def test_contextualize_query_multiple_turns():
    """With several turns, only the last user message is used for context."""
    pattern = _build_follow_up_pattern(DEFAULT_FOLLOW_UP_KEYWORDS)
    history = [
        {"role": "user", "content": "Tell me about the graduate program."},
        {"role": "assistant", "content": "The CS department offers MS and PhD."},
        {"role": "user", "content": "Who is the department head?"},
        {"role": "assistant", "content": "Dr. Cal Ribbens is the department head."},
    ]
    result = contextualize_query("What are their research interests?", history, pattern)
    assert "department head" in result
    assert "What are their research interests?" in result
    # Should NOT include old context about grad program
    assert "graduate program" not in result


def test_contextualize_query_standalone_question_not_enriched():
    """A standalone question with no follow-up signals should NOT be enriched."""
    pattern = _build_follow_up_pattern(DEFAULT_FOLLOW_UP_KEYWORDS)
    history = [
        {"role": "user", "content": "Who is Denis Gracanin?"},
        {"role": "assistant", "content": "Denis Gracanin is an associate professor."},
    ]
    result = contextualize_query("Tell me about graduate degrees in virginia tech", history, pattern)
    assert result == "Tell me about graduate degrees in virginia tech"
    assert "Denis" not in result


def test_contextualize_query_no_pattern_always_enriches():
    """When no pattern is provided, always enrich (backward compat)."""
    history = [
        {"role": "user", "content": "Who is Sally?"},
        {"role": "assistant", "content": "A professor."},
    ]
    result = contextualize_query("Tell me about graduate degrees", history, None)
    assert "Sally" in result
