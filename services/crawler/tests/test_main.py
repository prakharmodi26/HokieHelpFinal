import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from crawler.main import cli


def test_cli_runs_crawl(monkeypatch):
    """CLI loads config, creates storage, runs crawl, and exits."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")

    mock_storage_cls = MagicMock()
    mock_run_crawl = AsyncMock(return_value={"pages_crawled": 5, "pages_failed": 1})

    with patch("crawler.main.MinioStorage", mock_storage_cls), \
         patch("crawler.main.run_crawl", mock_run_crawl):
        cli()

    mock_storage_cls.assert_called_once()
    mock_run_crawl.assert_called_once()
