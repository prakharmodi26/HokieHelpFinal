"""Tests for token-aware chunking logic."""
import pytest
from chunker.parser import Section
from chunker.splitter import (
    estimate_tokens,
    split_large_section,
    build_chunks,
)
from chunker.models import ChunkRecord


# --- estimate_tokens ---

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    assert estimate_tokens("Hello world") == len("Hello world") // 4


def test_estimate_tokens_400_char_string():
    text = "a" * 400
    assert estimate_tokens(text) == 100


# --- split_large_section ---

def _make_section(text: str) -> Section:
    return Section(headings_path=["Test"], text=text)


def test_split_large_section_small_input_stays_single():
    text = "Short text. " * 10  # ~12 tokens, well under 400
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_split_large_section_large_input_splits():
    # Create text > 400 tokens (> 1600 chars)
    para = "This is a paragraph with some content. " * 10  # ~390 chars per repeat
    text = (para + "\n\n") * 5  # ~2000+ chars → > 500 tokens
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) >= 2


def test_split_large_section_no_chunk_exceeds_limit():
    para = "Word " * 200  # 1000 chars = 250 tokens each
    text = (para + "\n\n") * 4   # 4 paragraphs, total ~4000 chars = ~1000 tokens
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 400 + 64  # allow a little slack for overlap


def test_split_large_section_overlap_present():
    # Two big paragraphs; the second chunk should start with some chars from the first
    para = "X" * 1800  # 450 tokens
    text = para + "\n\n" + "Y" * 1800
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) >= 2
    # Second chunk should contain some overlap from first (last 64*4=256 chars of first)
    overlap_chars = 64 * 4
    first_end = chunks[0][-overlap_chars:]
    assert first_end in chunks[1]


# --- build_chunks ---

def _make_sections(sizes_in_tokens: list[int]) -> list[Section]:
    """Create sections with approximate given token sizes."""
    sections = []
    for i, tokens in enumerate(sizes_in_tokens):
        text = f"## Section {i}\n\n" + "word " * (tokens * 4 // 5)
        sections.append(Section(headings_path=[f"Section {i}"], text=text))
    return sections


FAKE_FM = type("FM", (), {
    "doc_id": "abc1234567890abc",
    "url": "https://website.cs.vt.edu/test",
    "title": "Test Page",
    "content_hash": "deadbeef01234567",
    "crawl_timestamp": "2026-03-16T03:00:00",
})()

FAKE_CONFIG = type("C", (), {
    "chunk_preferred_tokens": 400,
    "chunk_overlap_tokens": 64,
    "chunk_min_tokens": 120,
})()


def test_build_chunks_returns_chunk_records():
    sections = _make_sections([200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], ChunkRecord)


def test_build_chunks_chunk_ids_sequential():
    sections = _make_sections([200, 200, 200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_id == f"abc1234567890abc_{i:04d}"


def test_build_chunks_small_sections_merged():
    # 3 small sections (50 tokens each) should be merged into fewer chunks
    sections = _make_sections([50, 50, 50])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) < 3  # should be merged


def test_build_chunks_large_section_split():
    # 1 section of ~600 tokens should produce 2+ chunks
    sections = _make_sections([600])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) >= 2


def test_build_chunks_metadata_fields():
    sections = _make_sections([200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    c = chunks[0]
    assert c.document_id == "abc1234567890abc"
    assert c.url == "https://website.cs.vt.edu/test"
    assert c.title == "Test Page"
    assert c.crawl_timestamp == "2026-03-16T03:00:00"
    assert len(c.content_hash) == 16


def test_build_chunks_empty_sections_returns_empty():
    chunks = build_chunks([], FAKE_FM, FAKE_CONFIG)
    assert chunks == []
