"""Tests for chunk validation."""
import pytest
from embedder.validator import validate_chunks, ValidationResult


def _valid_chunk(**overrides):
    base = {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Some content here",
        "url": "https://example.com",
        "title": "Test Page",
        "page_type": "general",
        "headings_path": [],
        "content_hash": "deadbeef01234567",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 4,
    }
    base.update(overrides)
    return base


def test_valid_chunk_passes():
    result = validate_chunks([_valid_chunk()])
    assert len(result.valid) == 1
    assert len(result.invalid) == 0


def test_missing_chunk_id_rejected():
    result = validate_chunks([_valid_chunk(chunk_id="")])
    assert len(result.valid) == 0
    assert len(result.invalid) == 1
    assert "chunk_id" in result.invalid[0][1]


def test_missing_document_id_rejected():
    result = validate_chunks([_valid_chunk(document_id="")])
    assert len(result.valid) == 0
    assert "document_id" in result.invalid[0][1]


def test_empty_text_rejected():
    result = validate_chunks([_valid_chunk(text="")])
    assert len(result.valid) == 0
    assert "text" in result.invalid[0][1]


def test_whitespace_only_text_rejected():
    result = validate_chunks([_valid_chunk(text="   \n  ")])
    assert len(result.valid) == 0
    assert "text" in result.invalid[0][1]


def test_missing_url_rejected():
    result = validate_chunks([_valid_chunk(url="")])
    assert len(result.valid) == 0
    assert "url" in result.invalid[0][1]


def test_missing_title_gets_fallback():
    result = validate_chunks([_valid_chunk(title="")])
    assert len(result.valid) == 1
    assert result.valid[0]["title"] == "Untitled"


def test_none_title_gets_fallback():
    result = validate_chunks([_valid_chunk(title=None)])
    assert len(result.valid) == 1
    assert result.valid[0]["title"] == "Untitled"


def test_mixed_valid_and_invalid():
    chunks = [
        _valid_chunk(chunk_id="a_0000"),
        _valid_chunk(chunk_id=""),  # invalid
        _valid_chunk(chunk_id="c_0000", text=""),  # invalid
    ]
    result = validate_chunks(chunks)
    assert len(result.valid) == 1
    assert len(result.invalid) == 2


def test_empty_list_returns_empty():
    result = validate_chunks([])
    assert result.valid == []
    assert result.invalid == []
