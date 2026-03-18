"""Tests for EmbedderConfig."""
import pytest
from embedder.config import EmbedderConfig


def _base_env():
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "QDRANT_HOST": "localhost",
    }


def test_from_env_loads_required(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    assert cfg.minio_endpoint == "localhost:9000"
    assert cfg.qdrant_host == "localhost"


def test_from_env_defaults(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    assert cfg.minio_chunks_bucket == "chunks"
    assert cfg.minio_secure is False
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp_chunks"
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.embedding_batch_size == 32


def test_from_env_overrides(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("QDRANT_PORT", "7777")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    cfg = EmbedderConfig.from_env()
    assert cfg.qdrant_port == 7777
    assert cfg.embedding_batch_size == 64
    assert cfg.qdrant_collection == "test_chunks"


def test_from_env_missing_required_raises(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        EmbedderConfig.from_env()


def test_config_is_frozen(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    with pytest.raises((AttributeError, TypeError)):
        cfg.minio_endpoint = "other"  # type: ignore
