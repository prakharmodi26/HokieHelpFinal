"""Tests for ChunkRecord model."""
import pytest
from chunker.models import ChunkRecord


def _sample_chunk(**overrides) -> ChunkRecord:
    defaults = dict(
        chunk_id="abc1234567890abc_0000",
        document_id="abc1234567890abc",
        chunk_index=0,
        text="## Faculty\n\nThe CS department has world-class faculty.",
        url="https://website.cs.vt.edu/people/faculty",
        title="Faculty | CS | VT",
        page_type="faculty",
        headings_path=["People", "Faculty"],
        content_hash="deadbeef01234567",
        crawl_timestamp="2026-03-16T03:00:00+00:00",
        token_count=13,
    )
    defaults.update(overrides)
    return ChunkRecord(**defaults)


def test_chunk_record_fields():
    c = _sample_chunk()
    assert c.chunk_id == "abc1234567890abc_0000"
    assert c.chunk_index == 0
    assert c.headings_path == ["People", "Faculty"]


def test_to_dict_has_all_keys():
    d = _sample_chunk().to_dict()
    for key in (
        "chunk_id", "document_id", "chunk_index", "text", "url", "title",
        "page_type", "headings_path", "content_hash", "crawl_timestamp", "token_count",
    ):
        assert key in d


def test_to_dict_values_match():
    c = _sample_chunk()
    d = c.to_dict()
    assert d["chunk_id"] == c.chunk_id
    assert d["headings_path"] == ["People", "Faculty"]
    assert d["chunk_index"] == 0


def test_from_dict_round_trips():
    c = _sample_chunk()
    restored = ChunkRecord.from_dict(c.to_dict())
    assert restored.chunk_id == c.chunk_id
    assert restored.text == c.text
    assert restored.headings_path == c.headings_path
    assert restored.crawl_timestamp == c.crawl_timestamp


def test_from_dict_handles_missing_optional_fields():
    d = _sample_chunk().to_dict()
    d["page_type"] = "general"
    restored = ChunkRecord.from_dict(d)
    assert restored.page_type == "general"
