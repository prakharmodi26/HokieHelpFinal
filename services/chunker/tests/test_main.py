"""End-to-end tests for the chunker main orchestration."""
import json
from unittest.mock import MagicMock, patch
import pytest

from chunker.main import run_chunking


SAMPLE_DOC = """\
---
doc_id: 'abc1234567890abc'
url: 'https://website.cs.vt.edu/people/faculty'
title: 'Faculty | CS | VT'
crawl_depth: 1
crawl_timestamp: '2026-03-16T03:00:00+00:00'
content_hash: 'deadbeef01234567'
---

# People

## Faculty

Professor Smith researches distributed systems and has published extensively.
Professor Jones works on machine learning and has won several awards.
"""

EMPTY_DOC = """\
---
doc_id: 'empty000000000000'
url: 'https://website.cs.vt.edu/empty'
title: 'Empty'
crawl_depth: 1
crawl_timestamp: '2026-03-16T03:00:00+00:00'
content_hash: 'aaaa0000bbbb1111'
---

"""


def _make_storage(docs: dict) -> MagicMock:
    """Build a fake ChunkerStorage that serves given key→content mapping."""
    storage = MagicMock()
    storage.list_markdown_keys.return_value = list(docs.keys())
    storage.download_document.side_effect = lambda key: docs[key]
    return storage


def test_run_chunking_processes_document(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/faculty.md": SAMPLE_DOC})
    stats = run_chunking(storage, chunker_config)
    assert stats["processed"] == 1
    assert stats["total_chunks"] >= 1
    assert stats["failed"] == 0
    assert storage.upload_chunks.called


def test_run_chunking_skips_empty_body(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/empty.md": EMPTY_DOC})
    stats = run_chunking(storage, chunker_config)
    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert not storage.upload_chunks.called


def test_run_chunking_handles_failure(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/bad.md": "good content"})
    storage.upload_chunks.side_effect = RuntimeError("MinIO unavailable")
    # download succeeds but upload fails — should count as failed
    stats = run_chunking(storage, chunker_config)
    assert stats["failed"] == 1
    assert stats["processed"] == 0


def test_run_chunking_chunk_ids_use_doc_id(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/faculty.md": SAMPLE_DOC})
    run_chunking(storage, chunker_config)
    call_args = storage.upload_chunks.call_args
    doc_id = call_args[0][0]
    chunks = call_args[0][1]
    assert doc_id == "abc1234567890abc"
    for c in chunks:
        assert c.chunk_id.startswith("abc1234567890abc_")


def test_run_chunking_multiple_docs(chunker_config):
    docs = {
        "website.cs.vt.edu/a.md": SAMPLE_DOC,
        "website.cs.vt.edu/b.md": SAMPLE_DOC,
    }
    storage = _make_storage(docs)
    stats = run_chunking(storage, chunker_config)
    assert stats["processed"] == 2
    assert storage.upload_chunks.call_count == 2
