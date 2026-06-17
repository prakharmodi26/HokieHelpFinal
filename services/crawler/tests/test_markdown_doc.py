import pytest
from datetime import datetime, timezone
from crawler.markdown_doc import build_markdown_document, url_to_object_key


def test_build_markdown_document():
    """Builds a markdown string with YAML frontmatter from crawl data."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/academics/courses.html",
        title="CS Courses",
        markdown_content="# Courses\n\nHere are our courses.",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="deadbeef" * 8,
    )

    assert doc.startswith("---\n")
    assert "url: 'https://cs.vt.edu/academics/courses.html'" in doc
    assert "title: 'CS Courses'" in doc
    assert "crawl_depth: 1" in doc
    assert "crawl_timestamp: '2026-03-16T12:00:00+00:00'" in doc
    assert "doc_id: 'abcd1234abcd1234'" in doc
    assert "---\n\n# Courses\n\nHere are our courses." in doc


def test_build_markdown_document_no_title():
    """Uses URL as fallback when title is missing."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/about",
        title=None,
        markdown_content="About page content.",
        crawl_depth=0,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="deadbeef" * 8,
    )

    assert "title: 'https://cs.vt.edu/about'" in doc


def test_build_markdown_document_special_chars():
    """Handles titles with YAML-special characters like colons."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/courses/cs101",
        title="CS 101: Intro to CS",
        markdown_content="Course content.",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="deadbeef" * 8,
    )

    assert "title: 'CS 101: Intro to CS'" in doc


def test_build_markdown_document_includes_doc_id():
    doc = build_markdown_document(
        url="https://website.cs.vt.edu/about",
        title="About",
        markdown_content="# About",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="cafebabe" * 8,
    )
    assert "doc_id: 'abcd1234abcd1234'" in doc


def test_build_markdown_document_includes_content_hash():
    doc = build_markdown_document(
        url="https://website.cs.vt.edu/about",
        title="About",
        markdown_content="# About",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="cafebabe" * 8,
    )
    assert f"content_hash: '{'cafebabe' * 8}'" in doc


def test_build_markdown_document_doc_id_comes_before_url():
    doc = build_markdown_document(
        url="https://website.cs.vt.edu/about",
        title="About",
        markdown_content="# About",
        crawl_depth=0,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        doc_id="abcd1234abcd1234",
        content_hash="cafebabe" * 8,
    )
    end = doc.index("---", 3)
    fm = doc[:end]
    assert fm.index("doc_id:") < fm.index("url:")


def test_url_to_object_key_basic():
    """Converts a URL to a MinIO object key."""
    key = url_to_object_key("https://cs.vt.edu/academics/courses.html")
    assert key == "cs.vt.edu/academics/courses.html.md"


def test_url_to_object_key_trailing_slash():
    """Trailing slash is stripped so /about and /about/ map to the same key."""
    key = url_to_object_key("https://cs.vt.edu/academics/")
    assert key == "cs.vt.edu/academics.md"


def test_url_to_object_key_root():
    """Handles root URL."""
    key = url_to_object_key("https://cs.vt.edu")
    assert key == "cs.vt.edu/index.md"


def test_url_to_object_key_strips_query_and_fragment():
    """Strips query string and fragment from URL."""
    key = url_to_object_key("https://cs.vt.edu/page?foo=bar#section")
    assert key == "cs.vt.edu/page.md"


# --- new-field tests for full-domain ingestion ---
from datetime import datetime, timezone
from crawler.markdown_doc import build_markdown_document


def test_frontmatter_includes_host_and_page_type():
    doc = build_markdown_document(
        url="https://dining.vt.edu/menus",
        title="Menus",
        markdown_content="# Menus\nToday: pizza",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        doc_id="abc123",
        content_hash="deadbeef",
        source_host="dining.vt.edu",
        subdomain="dining",
        page_type="general",
        is_external=False,
        last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
    )
    assert "source_host: 'dining.vt.edu'" in doc
    assert "subdomain: 'dining'" in doc
    assert "page_type: 'general'" in doc
    assert "is_external: false" in doc
    assert doc.endswith("# Menus\nToday: pizza")


# --- frontmatter hardening against hostile metadata (ISC-13/14/15) ---
import yaml


def _parse_frontmatter_like_chunker(doc: str) -> dict:
    """Replicate the chunker's frontmatter extraction exactly: split at the
    first '---' after index 3, then yaml.safe_load the block."""
    assert doc.startswith("---")
    end = doc.find("---", 3)
    assert end != -1
    return yaml.safe_load(doc[3:end])


def test_frontmatter_survives_hostile_metadata():
    """A title/last_modified/url full of YAML-breaking chars must still produce
    frontmatter that the chunker's yaml.safe_load parses without raising."""
    doc = build_markdown_document(
        url="https://x.vt.edu/it's-a/page",
        title="Don't: break --- it\nNEWLINE: injected",
        markdown_content="# Body\n\nsome content",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        doc_id="abc123",
        content_hash="deadbeef",
        source_host="x.vt.edu",
        subdomain="x",
        last_modified="weird 'quote' --- value",
    )
    fm = _parse_frontmatter_like_chunker(doc)  # must not raise
    assert fm["url"] == "https://x.vt.edu/it's-a/page"
    assert "NEWLINE" not in str(fm.get("title")) or "\n" not in str(fm["title"])
    # body is preserved verbatim after the frontmatter close
    assert doc.rstrip().endswith("some content")


def test_frontmatter_value_with_triple_dash_does_not_truncate():
    """A literal '---' inside a value must not be mistaken for the closing
    delimiter (chunker uses find('---', 3))."""
    doc = build_markdown_document(
        url="https://x.vt.edu/p",
        title="A --- B --- C",
        markdown_content="# Body",
        crawl_depth=0,
        crawl_timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        doc_id="d1",
        content_hash="h1",
    )
    fm = _parse_frontmatter_like_chunker(doc)
    # all required keys survived (block wasn't truncated mid-frontmatter)
    for key in ("doc_id", "url", "title", "content_hash"):
        assert key in fm


def test_object_key_via_markdown_doc_is_minio_safe():
    """url_to_object_key now routes through safe_object_key (ISC-11)."""
    crash = (
        "https://www.pubs.ext.vt.edu/426-041.html"
        "https://www.pubs.ext.vt.edu/426-841.html"
    )
    key = url_to_object_key(crash)
    assert "://" not in key and ":" not in key and "//" not in key
