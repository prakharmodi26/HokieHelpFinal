"""Tests for the metadata module."""

import json
import hashlib
from datetime import datetime, timezone

import pytest

from crawler.metadata import PageMetadata, doc_id_for_url, metadata_key_for


# --- doc_id_for_url ---

def test_doc_id_for_url_is_16_hex_chars():
    assert len(doc_id_for_url("https://website.cs.vt.edu/about")) == 16
    assert all(c in "0123456789abcdef" for c in doc_id_for_url("https://website.cs.vt.edu/about"))

def test_doc_id_for_url_is_deterministic():
    url = "https://website.cs.vt.edu/about"
    assert doc_id_for_url(url) == doc_id_for_url(url)

def test_doc_id_for_url_normalises_trailing_slash():
    assert doc_id_for_url("https://website.cs.vt.edu/about") == \
           doc_id_for_url("https://website.cs.vt.edu/about/")

def test_doc_id_for_url_normalises_scheme_to_https():
    assert doc_id_for_url("http://website.cs.vt.edu/about") == \
           doc_id_for_url("https://website.cs.vt.edu/about")

def test_doc_id_for_url_normalises_fragment_away():
    assert doc_id_for_url("https://website.cs.vt.edu/page") == \
           doc_id_for_url("https://website.cs.vt.edu/page#section")

def test_doc_id_for_url_normalises_query_away():
    assert doc_id_for_url("https://website.cs.vt.edu/page") == \
           doc_id_for_url("https://website.cs.vt.edu/page?utm_source=x")

def test_doc_id_for_url_different_urls_differ():
    assert doc_id_for_url("https://website.cs.vt.edu/about") != \
           doc_id_for_url("https://website.cs.vt.edu/people")

def test_doc_id_matches_sha256_of_normalised_url():
    normalised = "https://website.cs.vt.edu/about"
    expected = hashlib.sha256(normalised.encode()).hexdigest()[:16]
    assert doc_id_for_url("https://website.cs.vt.edu/about/") == expected


# --- metadata_key_for ---

def test_metadata_key_for_replaces_md_extension():
    assert metadata_key_for("website.cs.vt.edu/about.md") == "website.cs.vt.edu/about.meta.json"

def test_metadata_key_for_double_extension():
    assert metadata_key_for("website.cs.vt.edu/About.html.md") == "website.cs.vt.edu/About.html.meta.json"

def test_metadata_key_for_nested_path():
    assert metadata_key_for("website.cs.vt.edu/people/faculty.md") == "website.cs.vt.edu/people/faculty.meta.json"


# --- PageMetadata ---

TS = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)

def _make_meta(**overrides) -> PageMetadata:
    defaults = dict(
        doc_id="abcd1234abcd1234",
        url="https://website.cs.vt.edu/about",
        title="About | CS | VT",
        crawl_depth=1,
        crawl_timestamp=TS,
        content_hash="deadbeef" * 8,
        markdown_size_bytes=1024,
        status_code=200,
        response_headers={"content-type": "text/html"},
        internal_links=["https://website.cs.vt.edu/people"],
        external_links=["https://eng.vt.edu"],
        last_modified=None,
        etag=None,
    )
    defaults.update(overrides)
    return PageMetadata(**defaults)

def test_page_metadata_fields():
    meta = _make_meta()
    assert meta.doc_id == "abcd1234abcd1234"
    assert meta.internal_links == ["https://website.cs.vt.edu/people"]
    assert meta.external_links == ["https://eng.vt.edu"]

def test_to_json_includes_all_required_fields():
    parsed = json.loads(_make_meta().to_json())
    for field in ("doc_id", "url", "title", "crawl_depth", "crawl_timestamp",
                  "content_hash", "markdown_size_bytes", "status_code",
                  "response_headers", "internal_links", "external_links",
                  "last_modified", "etag"):
        assert field in parsed

def test_to_json_serialises_timestamp_as_iso_string():
    parsed = json.loads(_make_meta().to_json())
    assert parsed["crawl_timestamp"] == "2026-03-16T12:00:00+00:00"

def test_from_json_round_trips():
    meta = _make_meta()
    restored = PageMetadata.from_json(meta.to_json())
    assert restored.doc_id == meta.doc_id
    assert restored.crawl_timestamp == meta.crawl_timestamp
    assert restored.content_hash == meta.content_hash
    assert restored.internal_links == meta.internal_links

def test_from_json_handles_null_optional_fields():
    meta = _make_meta(last_modified=None, etag=None, status_code=None, response_headers=None)
    restored = PageMetadata.from_json(meta.to_json())
    assert restored.last_modified is None
    assert restored.etag is None
