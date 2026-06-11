"""Tests for the registry-seeded frontier orchestration in main."""

from unittest.mock import AsyncMock, MagicMock

from crawler import main as cmain


def test_cli_runs_discover_then_engine_then_docs(monkeypatch, tmp_path):
    cfg = MagicMock()
    cfg.frontier_db_path = str(tmp_path / "f.db")
    cfg.minio_bucket = "crawled-pages"
    cfg.minio_cleaned_bucket = "crawled-pages-cleaned"
    cfg.minio_external_bucket = "crawled-pages-external"
    cfg.external_hops = 1
    cfg.max_depth = 2
    cfg.batch_size = 50
    cfg.memory_threshold_percent = 70.0
    cfg.max_sessions = 5
    cfg.prune_content_threshold = 0.5
    cfg.max_hosts = 5
    cfg.max_urls_per_host = 50
    cfg.request_delay = 0.1

    monkeypatch.setattr(cmain.CrawlerConfig, "from_env", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cmain, "MinioStorage", MagicMock())
    monkeypatch.setattr(cmain, "populate_frontier", AsyncMock(return_value={"urls_enqueued": 3}))
    monkeypatch.setattr(cmain, "run_engine", AsyncMock(return_value={
        "pages_stored": 3, "pages_failed": 0, "documents": 0, "document_urls": set(),
    }))
    monkeypatch.setattr(cmain, "download_and_process_documents",
                        AsyncMock(return_value={"documents_processed": 0, "documents_failed": 0}))

    cmain.cli()  # should not raise
    assert cmain.populate_frontier.called
    assert cmain.run_engine.called
