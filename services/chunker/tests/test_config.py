"""Tests for ChunkerConfig."""
import os
import pytest
from chunker.config import ChunkerConfig


def _base_env():
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
    }


def test_from_env_loads_required(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    assert cfg.minio_endpoint == "localhost:9000"
    assert cfg.minio_access_key == "minioadmin"
    assert cfg.minio_secret_key == "minioadmin"


def test_from_env_defaults(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    assert cfg.minio_cleaned_bucket == "crawled-pages-cleaned"
    assert cfg.minio_chunks_bucket == "chunks"
    assert cfg.minio_secure is False
    assert cfg.chunk_preferred_tokens == 400
    assert cfg.chunk_overlap_tokens == 64
    assert cfg.chunk_min_tokens == 120


def test_from_env_overrides(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHUNK_PREFERRED_TOKENS", "200")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "32")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "60")
    cfg = ChunkerConfig.from_env()
    assert cfg.chunk_preferred_tokens == 200
    assert cfg.chunk_overlap_tokens == 32
    assert cfg.chunk_min_tokens == 60


def test_from_env_missing_required_raises(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        ChunkerConfig.from_env()


def test_config_is_frozen(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    with pytest.raises((AttributeError, TypeError)):
        cfg.minio_endpoint = "other"  # type: ignore
