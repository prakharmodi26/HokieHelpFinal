import os
import pytest
from crawler.config import CrawlerConfig


def test_config_from_env_defaults(monkeypatch):
    """Config loads with sensible defaults when only required vars are set."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")

    config = CrawlerConfig.from_env()

    assert config.minio_endpoint == "localhost:9000"
    assert config.minio_access_key == "minioadmin"
    assert config.minio_secret_key == "minioadmin"
    assert config.minio_bucket == "crawled-pages"
    assert config.minio_secure is False
    assert config.seed_url == "https://cs.vt.edu"
    assert config.max_depth == 2
    assert config.max_pages == 500
    assert config.allowed_domain == "cs.vt.edu"


def test_config_from_env_custom(monkeypatch):
    """Config respects custom environment variable overrides."""
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.prod:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "prodkey")
    monkeypatch.setenv("MINIO_SECRET_KEY", "prodsecret")
    monkeypatch.setenv("MINIO_BUCKET", "my-bucket")
    monkeypatch.setenv("MINIO_SECURE", "true")
    monkeypatch.setenv("CRAWL_SEED_URL", "https://cs.vt.edu/academics")
    monkeypatch.setenv("CRAWL_MAX_DEPTH", "5")
    monkeypatch.setenv("CRAWL_MAX_PAGES", "1000")

    config = CrawlerConfig.from_env()

    assert config.minio_endpoint == "minio.prod:9000"
    assert config.minio_bucket == "my-bucket"
    assert config.minio_secure is True
    assert config.seed_url == "https://cs.vt.edu/academics"
    assert config.max_depth == 5
    assert config.max_pages == 1000


def test_config_missing_required_var(monkeypatch):
    """Config raises ValueError when required vars are missing."""
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        CrawlerConfig.from_env()
