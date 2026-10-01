"""Page metadata dataclass and helpers for sidecar .meta.json files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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


# Headers that must never be persisted to MinIO sidecars — they carry
# credentials or session tokens that would leak into the data lake.
_SENSITIVE_HEADERS = frozenset({
    "set-cookie", "authorization", "cookie", "x-api-key",
    "x-auth-token", "proxy-authorization", "www-authenticate",
})


def _filter_sensitive_headers(headers: dict | None) -> dict | None:
    """Return a copy of *headers* with sensitive entries removed."""
    if headers is None:
        return None
    return {k: v for k, v in headers.items() if k.lower() not in _SENSITIVE_HEADERS}


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

    # --- enrichment for full-domain ingestion (all optional/defaulted) ---
    source_host: Optional[str] = None
    subdomain: Optional[str] = None
    scope: str = "vt_page"
    is_external: bool = False
    hop_history: list[str] = field(default_factory=list)
    discovery_source: Optional[str] = None
    content_type: Optional[str] = None
    raw_markdown_size_bytes: Optional[int] = None
    fit_markdown_size_bytes: Optional[int] = None
    lang: Optional[str] = None
    word_count: Optional[int] = None
    render_tier: str = "fast"
    robots_allowed: bool = True
    fetched_at: Optional[datetime] = None

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
            "response_headers": _filter_sensitive_headers(self.response_headers),
            "internal_links": self.internal_links,
            "external_links": self.external_links,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "source_host": self.source_host,
            "subdomain": self.subdomain,
            "scope": self.scope,
            "is_external": self.is_external,
            "hop_history": self.hop_history,
            "discovery_source": self.discovery_source,
            "content_type": self.content_type,
            "raw_markdown_size_bytes": self.raw_markdown_size_bytes,
            "fit_markdown_size_bytes": self.fit_markdown_size_bytes,
            "lang": self.lang,
            "word_count": self.word_count,
            "render_tier": self.render_tier,
            "robots_allowed": self.robots_allowed,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> PageMetadata:
        """Deserialise from a JSON string."""
        data = json.loads(raw)
        _fetched_at_raw = data.get("fetched_at")
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
            source_host=data.get("source_host"),
            subdomain=data.get("subdomain"),
            scope=data.get("scope", "vt_page"),
            is_external=data.get("is_external", False),
            hop_history=data.get("hop_history", []),
            discovery_source=data.get("discovery_source"),
            content_type=data.get("content_type"),
            raw_markdown_size_bytes=data.get("raw_markdown_size_bytes"),
            fit_markdown_size_bytes=data.get("fit_markdown_size_bytes"),
            lang=data.get("lang"),
            word_count=data.get("word_count"),
            render_tier=data.get("render_tier", "fast"),
            robots_allowed=data.get("robots_allowed", True),
            fetched_at=datetime.fromisoformat(_fetched_at_raw) if _fetched_at_raw else None,
        )
