"""End-to-end tests for embedding orchestration with all deps mocked."""
from unittest.mock import MagicMock, patch
import pytest

from embedder.main import run_embedding


VALID_CHUNKS = [
    {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Faculty info",
        "url": "https://example.com",
        "title": "Faculty",
        "page_type": "faculty",
        "headings_path": ["People", "Faculty"],
        "content_hash": "deadbeef",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 3,
    },
    {
        "chunk_id": "abc_0001",
        "document_id": "abc",
        "chunk_index": 1,
        "text": "More faculty info",
        "url": "https://example.com",
        "title": "Faculty",
        "page_type": "faculty",
        "headings_path": ["People", "Faculty"],
        "content_hash": "cafebabe",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 4,
    },
]


def _make_storage(chunk_files: dict) -> MagicMock:
    storage = MagicMock()
    storage.list_chunk_keys.return_value = list(chunk_files.keys())
    storage.download_chunks.side_effect = lambda key: chunk_files[key]
    return storage


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.dimension = 1024
    embedder.embed_batch.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    return embedder


def _make_indexer() -> MagicMock:
    indexer = MagicMock()
    indexer.delete_stale_chunks.return_value = 0
    return indexer


def test_run_embedding_processes_doc(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["docs_processed"] == 1
    assert stats["chunks_embedded"] == 2
    assert stats["chunks_skipped"] == 0
    assert stats["failed"] == 0
    assert embedder.embed_batch.called
    assert indexer.upsert_chunks.called


def test_run_embedding_skips_invalid(embedder_config):
    bad_chunks = [{"chunk_id": "", "text": "", "document_id": "", "url": ""}]
    storage = _make_storage({"chunks/bad.json": bad_chunks})
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["chunks_skipped"] == 1
    assert stats["chunks_embedded"] == 0
    assert not embedder.embed_batch.called


def test_run_embedding_handles_failure(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()
    indexer.upsert_chunks.side_effect = RuntimeError("Qdrant down")

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["failed"] == 1
    assert stats["docs_processed"] == 0


def test_run_embedding_deletes_stale(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()
    indexer.delete_stale_chunks.return_value = 3

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["stale_deleted"] == 3
    indexer.delete_stale_chunks.assert_called_once_with("abc", {"abc_0000", "abc_0001"})


def test_run_embedding_multiple_docs(embedder_config):
    storage = _make_storage({
        "chunks/abc.json": VALID_CHUNKS,
        "chunks/def.json": [
            {**VALID_CHUNKS[0], "chunk_id": "def_0000", "document_id": "def"},
        ],
    })
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["docs_processed"] == 2
    assert stats["chunks_embedded"] == 3
