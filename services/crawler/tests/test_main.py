"""Tests for the crawler CLI entrypoint."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from crawler.main import cli
from crawler.metadata import PageMetadata


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


# --- Metadata mirroring tests ---

def _make_metadata(url: str = "https://website.cs.vt.edu/about") -> PageMetadata:
    return PageMetadata(
        doc_id="abcd1234abcd1234",
        url=url,
        title="About",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        content_hash="deadbeef" * 8,
        markdown_size_bytes=2048,
        status_code=200,
        response_headers=None,
        internal_links=[],
        external_links=[],
        last_modified=None,
        etag=None,
    )


def test_cli_mirrors_metadata_to_cleaned_bucket(crawler_config, monkeypatch):
    """After cleaning, each page's metadata sidecar is mirrored to the cleaned bucket
    with markdown_size_bytes updated to match the cleaned content byte length."""
    cleaned_doc = "---\ndoc_id: 'abcd1234abcd1234'\n---\n\n# About\n\nCleaned content."
    meta = _make_metadata()

    mock_storage = MagicMock()
    mock_storage.list_objects.return_value = ["website.cs.vt.edu/about.md"]
    mock_storage.download_document.return_value = cleaned_doc
    mock_storage.download_metadata.return_value = meta

    monkeypatch.setattr("crawler.main.CrawlerConfig.from_env", lambda: crawler_config)
    monkeypatch.setattr("crawler.main.MinioStorage", lambda cfg: mock_storage)
    monkeypatch.setattr("crawler.main.asyncio.run",
                        lambda coro: {"pages_crawled": 1, "pages_failed": 0})
    monkeypatch.setattr("crawler.main.clean_markdown", lambda x: cleaned_doc)

    cli()

    cleaned_meta_calls = [
        c for c in mock_storage.upload_metadata.call_args_list
        if (c.kwargs.get("bucket") == "test-bucket-cleaned"
            or (len(c.args) > 2 and c.args[2] == "test-bucket-cleaned"))
    ]
    assert len(cleaned_meta_calls) == 1
    stored_meta = cleaned_meta_calls[0].args[1]
    assert stored_meta.markdown_size_bytes == len(cleaned_doc.encode("utf-8"))


def test_cli_skips_metadata_mirror_when_no_sidecar(crawler_config, monkeypatch):
    """Synthetic docs like _department-info.md have no sidecar; exception is swallowed."""
    mock_storage = MagicMock()
    mock_storage.list_objects.return_value = ["_department-info.md"]
    mock_storage.download_document.return_value = "# Dept info"
    mock_storage.download_metadata.side_effect = Exception("No sidecar")

    monkeypatch.setattr("crawler.main.CrawlerConfig.from_env", lambda: crawler_config)
    monkeypatch.setattr("crawler.main.MinioStorage", lambda cfg: mock_storage)
    monkeypatch.setattr("crawler.main.asyncio.run",
                        lambda coro: {"pages_crawled": 0, "pages_failed": 0})
    monkeypatch.setattr("crawler.main.clean_markdown", lambda x: x)

    cli()  # must not raise

    cleaned_meta_calls = [
        c for c in mock_storage.upload_metadata.call_args_list
        if (c.kwargs.get("bucket") == "test-bucket-cleaned"
            or (len(c.args) > 2 and c.args[2] == "test-bucket-cleaned"))
    ]
    assert len(cleaned_meta_calls) == 0


def test_cli_metadata_size_reflects_cleaned_content(crawler_config, monkeypatch):
    """markdown_size_bytes in the cleaned metadata matches the actual cleaned byte count."""
    raw_doc = "# Page\n\n" + "boilerplate " * 100
    cleaned_doc = "# Page\n\nCleaned."
    meta = _make_metadata()
    meta_with_raw_size = PageMetadata(
        **{**meta.__dict__, "markdown_size_bytes": len(raw_doc.encode("utf-8"))}
    )

    mock_storage = MagicMock()
    mock_storage.list_objects.return_value = ["website.cs.vt.edu/page.md"]
    mock_storage.download_document.return_value = raw_doc
    mock_storage.download_metadata.return_value = meta_with_raw_size

    monkeypatch.setattr("crawler.main.CrawlerConfig.from_env", lambda: crawler_config)
    monkeypatch.setattr("crawler.main.MinioStorage", lambda cfg: mock_storage)
    monkeypatch.setattr("crawler.main.asyncio.run",
                        lambda coro: {"pages_crawled": 1, "pages_failed": 0})
    monkeypatch.setattr("crawler.main.clean_markdown", lambda x: cleaned_doc)

    cli()

    cleaned_meta_calls = [
        c for c in mock_storage.upload_metadata.call_args_list
        if (c.kwargs.get("bucket") == "test-bucket-cleaned"
            or (len(c.args) > 2 and c.args[2] == "test-bucket-cleaned"))
    ]
    stored_meta = cleaned_meta_calls[0].args[1]
    assert stored_meta.markdown_size_bytes == len(cleaned_doc.encode("utf-8"))
    assert stored_meta.markdown_size_bytes != len(raw_doc.encode("utf-8"))
