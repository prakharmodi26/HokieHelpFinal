# services/crawler/src/crawler/engine.py
"""Frontier-driven crawl engine: lease URLs, render with arun_many, store, expand.

Each streamed result is checkpointed immediately (mark_done / mark_failed) so a
restart resumes from the frontier. Children are classified and enqueued with the
full hop history. Empty fast-tier pages are signalled for a JS-tier retry.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, MemoryAdaptiveDispatcher, RateLimiter

from crawler.catalog_card import card_markdown, extract_card
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


class _WedgedBatch(Exception):
    """Raised when an arun_many result stream goes idle past the watchdog
    timeout — a single page whose render never returns (binary downloads with
    no extension, SPA login pages with live websockets) stalled the whole
    batch. page_timeout only bounds navigation, not post-load scans/waits, so
    such pages can hang the stream forever; this is the recovery signal."""


async def _iter_with_idle_timeout(results, idle_timeout: float):
    """Yield results from an arun_many stream, raising _WedgedBatch if no new
    result arrives within idle_timeout seconds. The underlying generator is
    closed (cancelling the stuck in-flight render) before the signal is raised.
    """
    agen = results.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(agen.__anext__(), timeout=idle_timeout)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                await agen.aclose()
            raise _WedgedBatch
        yield item


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


def _store_catalog_card(result, item, frontier, storage, *, main_bucket, tier):
    """Catalog-mode host: store a metadata-only card, never the page body.

    Head metadata (citation_*/DC/JSON-LD/OG/title) becomes a compact card doc
    with page_type=catalog_card. No children are enqueued and no document
    links are harvested — catalog hosts are "know it exists" coverage; the
    per-record depth for repositories comes from the OAI/Dataverse harvest
    pass, not from page crawling.

    Cards go to the MAIN bucket and reach the cleaned bucket through the
    existing mirror_vt_pages pass, identical to full pages.
    """
    card = extract_card(getattr(result, "html", "") or "", item.url)
    body = card_markdown(card)
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    status_code = getattr(result, "status_code", None) or 200
    now = datetime.now(timezone.utc)
    headers = result.response_headers or {}
    host = urlparse(item.url).hostname

    document = build_markdown_document(
        url=item.url, title=card.get("title"), markdown_content=body,
        crawl_depth=item.depth, crawl_timestamp=now,
        doc_id=doc_id_for_url(item.url), content_hash=content_hash,
        source_host=host, subdomain=subdomain_of(item.url),
        page_type="catalog_card", is_external=False,
        last_modified=headers.get("last-modified"),
    )
    object_key = url_to_object_key(item.url)
    storage.upload_document(object_key, document, bucket=main_bucket)
    meta = PageMetadata(
        doc_id=doc_id_for_url(item.url), url=item.url, title=card.get("title"),
        crawl_depth=item.depth, crawl_timestamp=now, content_hash=content_hash,
        markdown_size_bytes=len(body.encode()),
        status_code=getattr(result, "status_code", None),
        response_headers=headers or None,
        internal_links=[], external_links=[],
        last_modified=headers.get("last-modified"), etag=headers.get("etag"),
        source_host=host, subdomain=subdomain_of(item.url),
        scope=item.scope, is_external=False, hop_history=item.hop_history,
        discovery_source=item.discovery_source,
        content_type=headers.get("content-type"),
        word_count=word_count(body), render_tier=tier, robots_allowed=True,
        fetched_at=now,
    )
    storage.upload_metadata(object_key, meta, bucket=main_bucket)
    _ensure_row(frontier, item)
    frontier.mark_done(item.url, content_hash=content_hash, http_status=status_code)
    return set()


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
    host_counts: dict[str, int] | None = None,
    max_urls_per_host: int = -1,
    mode: str = "full",
):
    """Store one crawl result and expand its links. Returns either a set of
    discovered document URLs, or the sentinel 'needs_js' for fast-tier empties.

    `mode` is the registry host mode: "full" (default — extract/clean + BFS
    expansion) or "catalog" (metadata-only card, no expansion). "special"
    items never reach process_result — run_engine skips them before render.
    """
    if not result.success:
        _ensure_row(frontier, item)
        frontier.mark_failed(
            item.url,
            error=getattr(result, "error_message", None) or "unknown",
            http_status=getattr(result, "status_code", None),
        )
        return set()

    if mode == "catalog":
        return _store_catalog_card(
            result, item, frontier, storage, main_bucket=main_bucket, tier=tier,
        )

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
                child_host = (urlparse(href).hostname or "").lower()
                if (
                    max_urls_per_host != -1
                    and host_counts is not None
                    and host_counts.get(child_host, 0) >= max_urls_per_host
                ):
                    continue
                if frontier.enqueue(
                    href, scope=child_scope.value, depth=item.depth + 1,
                    hop_history=child_hops, discovery_source="bfs",
                ) and host_counts is not None:
                    host_counts[child_host] = host_counts.get(child_host, 0) + 1
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


def _match_item(by_url: dict[str, FrontierItem], result) -> FrontierItem | None:
    """Map a streamed result back to its leased frontier item.

    Verified against the installed crawl4ai 0.8.6: CrawlResult.url is the
    ORIGINAL requested URL (aprocess_html is called with url=url) and the
    post-redirect final URL lives in CrawlResult.redirected_url. So result.url
    is the primary key; redirected_url is a defensive fallback in case a code
    path ever surfaces the final URL instead.
    """
    item = by_url.get(result.url)
    if item is None:
        item = by_url.get(getattr(result, "redirected_url", None) or "")
    return item


async def run_engine(
    frontier: Frontier,
    storage,
    *,
    main_bucket: str,
    external_bucket: str,
    external_hops: int,
    max_depth: int,
    batch_size: int,
    memory_pct: float,
    max_sessions: int,
    threshold: float,
    host_filter=None,
    max_urls_per_host: int = -1,
    modes: dict[str, str] | None = None,
    batch_idle_timeout: float = 180.0,
) -> dict:
    """Lease pending URLs in batches, render (fast tier), checkpoint, retry empties
    on the JS tier, until the frontier has no pending rows.

    `modes` maps registry hosts to "full" | "catalog" | "special"
    (registry.load_modes). Hosts not in the map — including every
    BFS-discovered host — default to "full". None (the default) preserves the
    historical all-full behaviour.

    `batch_idle_timeout` guards against a single page whose render never returns
    wedging the whole crawl: if the result stream produces nothing new for that
    many seconds, the stuck URLs are failed, the browser is recycled, and the
    crawl continues.
    """
    stats = {"pages_stored": 0, "pages_failed": 0, "needs_js": 0, "documents": 0}

    def mode_of(url: str) -> str:
        if not modes:
            return "full"
        return modes.get((urlparse(url).hostname or "").lower(), "full")
    document_urls: set[str] = set()
    fast_cfg = fast_run_config(threshold)
    js_cfg = js_run_config(threshold)

    # Crash recovery: any row still 'leased' at engine start was stranded by a
    # prior run that died mid-batch (this run hasn't leased anything yet, and
    # there are no concurrent runners in v1) — return them to the pool.
    requeued = frontier.requeue_leased()
    if requeued:
        logger.info("Requeued %d leased rows stranded by a prior run", requeued)

    # Per-host URL cap (-1 = unlimited). Counts include rows from a resumed
    # frontier so a restart cannot blow past the cap.
    host_counts = frontier.count_by_host() if max_urls_per_host != -1 else None

    def _handle(result, by_url, tier):
        """Process one streamed result against its leased item. Returns
        (item, outcome), or (None, None) if the result can't be matched."""
        item = _match_item(by_url, result)
        if item is None:
            # Unmatchable result: its row stays leased and is recovered by
            # requeue_leased() on the next run.
            logger.warning("No leased item matches result url %s", result.url)
            return None, None
        outcome = process_result(
            result, item, frontier, storage,
            main_bucket=main_bucket, external_bucket=external_bucket,
            external_hops=external_hops, max_depth=max_depth, tier=tier,
            host_filter=host_filter, host_counts=host_counts,
            max_urls_per_host=max_urls_per_host, mode=mode_of(item.url),
        )
        return item, outcome

    def _fail_unreturned(by_url, handled, reason):
        """Mark leased URLs that never produced a result as failed so they are
        not re-leased and re-wedged on the next pass of this run."""
        for url, item in by_url.items():
            if url not in handled:
                _ensure_row(frontier, item)
                frontier.mark_failed(url, error=reason)
                logger.warning("Wedged URL failed (%s): %s", reason, url)

    drained = False
    while not drained:
        # The crawler (and its dispatcher) is created fresh each time the engine
        # recovers from a wedged batch: exiting the context kills the stuck
        # Chromium processes that the dead batch left behind.
        async with AsyncWebCrawler(config=browser_config()) as crawler:
            dispatcher = make_dispatcher(memory_pct, max_sessions)
            recycle = False
            while not recycle:
                batch = frontier.lease_pending(limit=batch_size)
                if not batch:
                    drained = True
                    break

                # "special" hosts are never rendered or full-text ingested here.
                # The only crawl-seeded special handler is courseleaf_catalog
                # (catalog.vt.edu); Task 8.4 adds its dedicated structured course
                # ingest pass — until then its rows are checkpointed as skipped.
                # dining_tool hosts are excluded by load_seed_hosts and never
                # reach the frontier at all.
                render_batch = []
                for it in batch:
                    if mode_of(it.url) == "special":
                        frontier.mark_skipped(
                            it.url, reason="special handler pass pending (Task 8.4: courseleaf)"
                        )
                        continue
                    render_batch.append(it)
                if not render_batch:
                    continue
                by_url = {it.url: it for it in render_batch}

                # --- fast tier ---
                js_retry: list[FrontierItem] = []
                handled: set[str] = set()
                results = await crawler.arun_many(
                    urls=list(by_url.keys()), config=fast_cfg, dispatcher=dispatcher,
                )
                try:
                    async for result in _iter_with_idle_timeout(results, batch_idle_timeout):
                        item, outcome = _handle(result, by_url, "fast")
                        if item is None:
                            continue
                        handled.add(item.url)
                        if outcome == NEEDS_JS:
                            js_retry.append(item)
                            stats["needs_js"] += 1
                        else:
                            document_urls |= outcome
                except _WedgedBatch:
                    _fail_unreturned(
                        by_url, handled,
                        f"render watchdog: no result for {batch_idle_timeout:.0f}s (fast tier)",
                    )
                    recycle = True
                    break  # recycle the browser, then resume leasing

                # --- JS tier for empties ---
                if js_retry:
                    js_by_url = {it.url: it for it in js_retry}
                    js_handled: set[str] = set()
                    js_results = await crawler.arun_many(
                        urls=list(js_by_url.keys()), config=js_cfg, dispatcher=dispatcher,
                    )
                    try:
                        async for result in _iter_with_idle_timeout(js_results, batch_idle_timeout):
                            item, outcome = _handle(result, js_by_url, "js")
                            if item is None:
                                continue
                            js_handled.add(item.url)
                            # tier="js" never returns the sentinel (empties are
                            # marked skipped), but guard anyway so a string can
                            # never pollute the document URL set.
                            if outcome != NEEDS_JS:
                                document_urls |= outcome
                    except _WedgedBatch:
                        _fail_unreturned(
                            js_by_url, js_handled,
                            f"render watchdog: no result for {batch_idle_timeout:.0f}s (js tier)",
                        )
                        recycle = True
                        break

    # Frontier counts are the authoritative tally (each result was checkpointed
    # as it streamed in); no per-result counter needed.
    counts = frontier.count_by_status()
    stats["pages_stored"] = counts.get("done", 0)
    stats["pages_failed"] = counts.get("failed", 0)
    stats["documents"] = len(document_urls)
    stats["document_urls"] = document_urls
    return stats
