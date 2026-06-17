"""Build Markdown documents with YAML frontmatter from crawl results."""

from __future__ import annotations

from datetime import datetime

from crawler.urls import safe_object_key


def _yaml_scalar(value: str) -> str:
    """Make a string safe inside a single-quoted YAML frontmatter scalar.

    Three hazards are neutralized so site-controlled metadata (title, url,
    last-modified header, …) can never break the downstream chunker's
    yaml.safe_load OR its frontmatter delimiter scan:
      - single quotes are doubled ('' is the YAML single-quote escape);
      - newlines / tabs become spaces (a raw newline ends the scalar early);
      - '---' is despaced to '- - -' (the chunker finds the frontmatter close
        with markdown.find('---', 3), so a literal '---' inside a value would
        be mistaken for the closing delimiter and truncate the block).
    """
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = value.replace("---", "- - -")
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
        f"doc_id: '{_yaml_scalar(doc_id)}'",
        f"url: '{_yaml_scalar(url)}'",
        f"title: '{_yaml_scalar(effective_title)}'",
        f"crawl_depth: {crawl_depth}",
        f"crawl_timestamp: '{ts}'",
        f"content_hash: '{_yaml_scalar(content_hash)}'",
        f"source_host: '{_yaml_scalar(source_host or '')}'",
        f"subdomain: '{_yaml_scalar(subdomain or '')}'",
        f"page_type: '{_yaml_scalar(page_type)}'",
        f"is_external: {'true' if is_external else 'false'}",
        f"last_modified: '{_yaml_scalar(last_modified or '')}'",
        "---",
    ]
    return "\n".join(lines) + f"\n\n{markdown_content}"


def url_to_object_key(url: str) -> str:
    """Convert a URL to a structured, MinIO-legal object key.

    Example: https://cs.vt.edu/academics/courses.html -> cs.vt.edu/academics/courses.html.md

    Delegates to crawler.urls.safe_object_key so a weird-but-accepted URL can
    never produce a key MinIO rejects (the concatenated-href crash). Behaviour
    is unchanged for already-clean URLs.
    """
    return safe_object_key(url, suffix=".md")
