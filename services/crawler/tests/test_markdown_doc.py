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
    )

    assert doc.startswith("---\n")
    assert "url: 'https://cs.vt.edu/academics/courses.html'" in doc
    assert "title: 'CS Courses'" in doc
    assert "crawl_depth: 1" in doc
    assert "crawl_timestamp: '2026-03-16T12:00:00+00:00'" in doc
    assert "---\n\n# Courses\n\nHere are our courses." in doc


def test_build_markdown_document_no_title():
    """Uses URL as fallback when title is missing."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/about",
        title=None,
        markdown_content="About page content.",
        crawl_depth=0,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
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
    )

    assert "title: 'CS 101: Intro to CS'" in doc


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
