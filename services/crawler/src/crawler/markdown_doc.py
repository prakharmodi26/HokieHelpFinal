"""Build Markdown documents with YAML frontmatter from crawl results."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse


def _yaml_escape(value: str) -> str:
    """Escape single quotes for single-quoted YAML scalars."""
    return value.replace("'", "''")


def build_markdown_document(
    *,
    url: str,
    title: str | None,
    markdown_content: str,
    crawl_depth: int,
    crawl_timestamp: datetime,
    doc_id: str,
    content_hash: str,
    source_host: str | None = None,
    subdomain: str | None = None,
    page_type: str = "general",
    is_external: bool = False,
    last_modified: str | None = None,
) -> str:
    """Return a Markdown string with YAML frontmatter metadata."""
    effective_title = title if title else url
    ts = crawl_timestamp.isoformat()

    lines = [
        "---",
        f"doc_id: '{doc_id}'",
        f"url: '{url}'",
        f"title: '{_yaml_escape(effective_title)}'",
        f"crawl_depth: {crawl_depth}",
        f"crawl_timestamp: '{ts}'",
        f"content_hash: '{content_hash}'",
        f"source_host: '{source_host or ''}'",
        f"subdomain: '{subdomain or ''}'",
        f"page_type: '{page_type}'",
        f"is_external: {'true' if is_external else 'false'}",
        f"last_modified: '{last_modified or ''}'",
        "---",
    ]
    return "\n".join(lines) + f"\n\n{markdown_content}"


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
