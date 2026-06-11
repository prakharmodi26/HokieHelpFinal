# services/crawler/src/crawler/engine.py
"""Frontier-driven crawl engine: lease URLs, render with arun_many, store, expand.

Each streamed result is checkpointed immediately (mark_done / mark_failed) so a
restart resumes from the frontier. Children are classified and enqueued with the
full hop history. Empty fast-tier pages are signalled for a JS-tier retry.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, MemoryAdaptiveDispatcher, RateLimiter

from crawler.documents import collect_document_links
from crawler.extract import looks_empty, select_content, word_count
from crawler.frontier import Frontier, FrontierItem
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.parser_hint import infer_page_type
from crawler.render import bucket_for_scope, fast_run_config, js_run_config
from crawler.scope import Scope, classify_url, subdomain_of

logger = logging.getLogger(__name__)

# Sentinel returned by process_result when a fast-tier render came back empty
# and the page should be retried on the JS tier.
NEEDS_JS = "needs_js"


def _ensure_row(frontier: Frontier, item: FrontierItem) -> None:
    """Make sure the frontier row for `item` exists before a terminal mark.

    In the normal flow items come from lease_pending so the row exists and the
    enqueue is a no-op (digest conflict -> False). This self-heals items handed
    to process_result directly so mark_done/mark_failed never silently update
    zero rows.
    """
    frontier.enqueue(
        item.url, scope=item.scope, depth=item.depth,
        hop_history=item.hop_history, discovery_source=item.discovery_source,
    )


def process_result(
    result,
    item: FrontierItem,
    frontier: Frontier,
    storage,
    *,
    main_bucket: str,
    external_bucket: str,
    external_hops: int,
    max_depth: int,
    tier: str = "fast",
    host_filter=None,
):
    """Store one crawl result and expand its links. Returns either a set of
    discovered document URLs, or the sentinel 'needs_js' for fast-tier empties."""
    if not result.success:
        _ensure_row(frontier, item)
        frontier.mark_failed(
            item.url,
            error=getattr(result, "error_message", None) or "unknown",
            http_status=getattr(result, "status_code", None),
        )
        return set()

    text, used = select_content(result.markdown)
    if looks_empty(text):
        if tier == "fast":
            return NEEDS_JS  # retry in JS tier; leave row leased for now
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason="empty after JS render")
        return set()

    status_code = getattr(result, "status_code", None) or 200
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    if frontier.content_hash_seen(content_hash):
        # Duplicate content already stored under another URL: checkpoint without
        # re-uploading, but still harvest document links from this page.
        _ensure_row(frontier, item)
        frontier.mark_done(item.url, content_hash=content_hash, http_status=status_code)
        return collect_document_links(result.links)

    scope = Scope(item.scope)
    now = datetime.now(timezone.utc)
    headers = result.response_headers or {}
    title = (result.metadata or {}).get("title")
    page_type = infer_page_type(item.url)
    host = urlparse(item.url).hostname
    raw = getattr(result.markdown, "raw_markdown", "") or ""

    document = build_markdown_document(
        url=item.url, title=title, markdown_content=text,
        crawl_depth=item.depth, crawl_timestamp=now,
        doc_id=doc_id_for_url(item.url), content_hash=content_hash,
        source_host=host, subdomain=subdomain_of(item.url),
        page_type=page_type, is_external=(scope is Scope.EXTERNAL_PAGE),
        last_modified=headers.get("last-modified"),
    )
    bucket = bucket_for_scope(scope, main_bucket, external_bucket)
    object_key = url_to_object_key(item.url)
    storage.upload_document(object_key, document, bucket=bucket)

    links = result.links or {}
    meta = PageMetadata(
        doc_id=doc_id_for_url(item.url), url=item.url, title=title,
        crawl_depth=item.depth, crawl_timestamp=now, content_hash=content_hash,
        markdown_size_bytes=len(text.encode()),
        status_code=getattr(result, "status_code", None),
        response_headers=headers or None,
        internal_links=[l["href"] for l in links.get("internal", []) if l.get("href")],
        external_links=[l["href"] for l in links.get("external", []) if l.get("href")],
        last_modified=headers.get("last-modified"),
        etag=headers.get("etag"),
        source_host=host, subdomain=subdomain_of(item.url), scope=scope.value,
        is_external=(scope is Scope.EXTERNAL_PAGE), hop_history=item.hop_history,
        discovery_source=item.discovery_source,
        content_type=headers.get("content-type"),
        raw_markdown_size_bytes=len(raw.encode()),
        # When fit_markdown was empty we stored raw, so the fit size is 0.
        fit_markdown_size_bytes=len(text.encode()) if used == "fit" else 0,
        word_count=word_count(text), render_tier=tier, robots_allowed=True,
        fetched_at=now,
    )
    storage.upload_metadata(object_key, meta, bucket=bucket)
    _ensure_row(frontier, item)
    frontier.mark_done(item.url, content_hash=content_hash, http_status=status_code)

    # Expand: enqueue children (only from VT pages, to bound external fan-out).
    if scope is Scope.VT_PAGE and item.depth < max_depth:
        for cat in ("internal", "external"):
            for link in links.get(cat, []):
                href = link.get("href")
                if not href:
                    continue
                child_scope, child_hops = classify_url(
                    href, parent_hops=item.hop_history, external_hops=external_hops,
                    host_filter=host_filter,
                )
                if child_scope is Scope.OUT_OF_SCOPE:
                    continue
                # DOCUMENT children are deliberately NOT enqueued: PDFs/DOCX in a
                # browser-render frontier would waste a render (and usually fail).
                # collect_document_links below routes them to the documents pass.
                if child_scope is Scope.DOCUMENT:
                    continue
                frontier.enqueue(
                    href, scope=child_scope.value, depth=item.depth + 1,
                    hop_history=child_hops, discovery_source="bfs",
                )
    return collect_document_links(result.links)


def browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True, text_mode=True, light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )


def make_dispatcher(memory_pct: float, max_sessions: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=memory_pct,
        check_interval=1.0,
        max_session_permit=max_sessions,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 3.0), max_delay=60.0, max_retries=3,
            rate_limit_codes=[429, 503],
        ),
    )
