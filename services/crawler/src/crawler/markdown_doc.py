"""Build Markdown documents with YAML frontmatter from crawl results."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse


def build_markdown_document(
    *,
    url: str,
    title: str | None,
    markdown_content: str,
    crawl_depth: int,
    crawl_timestamp: datetime,
    doc_id: str,
    content_hash: str,
) -> str:
    """Return a Markdown string with YAML frontmatter metadata."""
    effective_title = title if title else url
    ts = crawl_timestamp.isoformat()

    frontmatter = (
        f"---\n"
        f"doc_id: '{doc_id}'\n"
        f"url: '{url}'\n"
        f"title: '{effective_title}'\n"
        f"crawl_depth: {crawl_depth}\n"
        f"crawl_timestamp: '{ts}'\n"
        f"content_hash: '{content_hash}'\n"
        f"---"
    )
    return f"{frontmatter}\n\n{markdown_content}"


def url_to_object_key(url: str) -> str:
    """Convert a URL to a structured MinIO object key.

    Example: https://cs.vt.edu/academics/courses.html -> cs.vt.edu/academics/courses.html.md
    """
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc
    path = parsed.path

    # Strip leading and trailing slashes to normalize /about and /about/
    path = path.strip("/")

    # Handle root URL (empty path)
    if not path:
        path = "index"

    # Add .md extension if not already a .md file
    if not path.endswith(".md"):
        path = path + ".md"

    return f"{host}/{path}"
