import pytest
from admin.config import AdminConfig


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("EMBEDDER_URL", "http://hokiehelp-embedder:8080")
    cfg = AdminConfig.from_env()
    assert cfg.minio_endpoint == "minio:9000"
    assert cfg.qdrant_port == 6333
    assert cfg.default_max_depth == 4
    assert cfg.default_schedule == "0 3 * * 0"
    assert cfg.data_dir == "/data"


def test_config_missing_required(monkeypatch):
    for k in ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "QDRANT_HOST", "EMBEDDER_URL"]:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        AdminConfig.from_env()
