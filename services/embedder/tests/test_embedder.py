"""Tests for the embedding module — model is always mocked."""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from embedder.embedder import Embedder, build_context


# --- build_context (pure function, no mock needed) ---

def test_build_context_full():
    chunk = {
        "title": "Faculty | CS | VT",
        "headings_path": ["People", "Faculty", "Tenured Faculty"],
        "text": "Professor Smith researches systems.",
    }
    ctx = build_context(chunk)
    assert "Title: Faculty | CS | VT" in ctx
    assert "Section: Tenured Faculty" in ctx
    assert "Path: People > Faculty > Tenured Faculty" in ctx
    assert "Professor Smith researches systems." in ctx


def test_build_context_no_headings():
    chunk = {
        "title": "Home",
        "headings_path": [],
        "text": "Welcome to CS.",
    }
    ctx = build_context(chunk)
    assert "Title: Home" in ctx
    assert "Section:" not in ctx
    assert "Path:" not in ctx
    assert "Welcome to CS." in ctx


def test_build_context_single_heading():
    chunk = {
        "title": "About",
        "headings_path": ["About Us"],
        "text": "We are the CS department.",
    }
    ctx = build_context(chunk)
    assert "Section: About Us" in ctx
    assert "Path: About Us" in ctx


# --- Embedder class (model mocked) ---

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 1024
    # Return numpy arrays mimicking real embeddings
    model.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 1024).astype(np.float32)
    return model


@pytest.fixture
def embedder(mock_model):
    with patch("embedder.embedder.SentenceTransformer", return_value=mock_model):
        return Embedder("mock-model")


def test_embedder_dimension(embedder):
    assert embedder.dimension == 1024


def test_embed_batch_returns_list_of_lists(embedder):
    result = embedder.embed_batch(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 1024
    assert isinstance(result[0], list)


def test_embed_batch_single(embedder):
    result = embedder.embed_batch(["hello"])
    assert len(result) == 1


def test_embed_batch_empty(embedder):
    result = embedder.embed_batch([])
    assert result == []
