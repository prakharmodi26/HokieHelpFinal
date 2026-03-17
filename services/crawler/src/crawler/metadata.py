"""Page metadata dataclass and helpers for sidecar .meta.json files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, urlunparse


def _normalise_url(url: str) -> str:
    """Return canonical URL: https scheme, no trailing slash, no query, no fragment."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(("https", parsed.netloc, path, "", "", ""))


def doc_id_for_url(url: str) -> str:
    """Return a stable 16-char hex ID for a URL (first 16 hex chars of SHA-256)."""
    return hashlib.sha256(_normalise_url(url).encode()).hexdigest()[:16]


def metadata_key_for(object_key: str) -> str:
    """Derive sidecar key: website.cs.vt.edu/About.html.md -> website.cs.vt.edu/About.html.meta.json"""
    if object_key.endswith(".md"):
        return object_key[:-3] + ".meta.json"
    return object_key + ".meta.json"


@dataclass
class PageMetadata:
    """All metadata captured for a single crawled page."""

    doc_id: str
    url: str
    title: Optional[str]
    crawl_depth: int
    crawl_timestamp: datetime
    content_hash: str
    markdown_size_bytes: int
    status_code: Optional[int]
    response_headers: Optional[dict]
    internal_links: list[str]
    external_links: list[str]
    last_modified: Optional[str]
    etag: Optional[str]

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        payload = {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "crawl_depth": self.crawl_depth,
            "crawl_timestamp": self.crawl_timestamp.isoformat(),
            "content_hash": self.content_hash,
            "markdown_size_bytes": self.markdown_size_bytes,
            "status_code": self.status_code,
            "response_headers": self.response_headers,
            "internal_links": self.internal_links,
            "external_links": self.external_links,
            "last_modified": self.last_modified,
            "etag": self.etag,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> PageMetadata:
        """Deserialise from a JSON string."""
        data = json.loads(raw)
        return cls(
            doc_id=data["doc_id"],
            url=data["url"],
            title=data.get("title"),
            crawl_depth=data["crawl_depth"],
            crawl_timestamp=datetime.fromisoformat(data["crawl_timestamp"]),
            content_hash=data["content_hash"],
            markdown_size_bytes=data["markdown_size_bytes"],
            status_code=data.get("status_code"),
            response_headers=data.get("response_headers"),
            internal_links=data.get("internal_links", []),
            external_links=data.get("external_links", []),
            last_modified=data.get("last_modified"),
            etag=data.get("etag"),
        )
