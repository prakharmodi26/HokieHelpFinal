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
import json
import logging
import threading
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
from crawler.storage import StorageFullError
from crawler.scope import Scope, classify_url, subdomain_of
from crawler.validate import (
    has_boilerplate_leak,
    validate_catalog_card,
    validate_html_content,
    validate_http_response,
)

logger = logging.getLogger(__name__)

# Sentinel returned by process_result when a fast-tier render came back empty
# and the page should be retried on the JS tier.
NEEDS_JS = "needs_js"

# Sentinel returned by process_result when content is blocked (login wall,
# bot challenge). The frontier row is marked 'skipped' with reason
# "blocked:{validation_reason}" — no storage, no re-crawl within this run.
# The next scheduled crawl run (fresh frontier) will retry it.
BLOCKED = "blocked"


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

    Cards go to the MAIN bucket (bucket=main_bucket), identical to full pages.
    """
    card = extract_card(getattr(result, "html", "") or "", item.url)

    # Structural validation: reject cards with no metadata (empty title AND
    # abstract, or title is just the URL).
    card_result = validate_catalog_card(card)
    if not card_result.valid:
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason=card_result.reason)
        logger.info("Rejected (%s): %s", card_result.reason, item.url)
        return set()

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


class RunAborted(RuntimeError):
    """Raised to abort the whole crawl run (non-zero exit) when process_result
    fails on many consecutive items — a sign the storage backend went
    unavailable mid-run. Without this the poison-pill marks every page failed
    while the run still reports 'completed' with nothing stored (issue #5)."""


# Consecutive process_result failures that trip the circuit breaker. Scattered
# poison pages reset it on the next success; only a systematic failure (backend
# down) reaches this many in a row.
_MAX_CONSECUTIVE_WRITE_FAILURES = 50


def process_result(
    result,
    item: FrontierItem,
    frontier: Frontier,
    storage,
    *,
    main_bucket: str,
    max_depth: int,
    tier: str = "fast",
    host_filter=None,
    affiliated_matcher=None,
    host_counts: dict[str, int] | None = None,
    max_urls_per_host: int = -1,
    host_caps: dict[str, int] | None = None,
    mode: str = "full",
    max_html_size_bytes: int = 10 * 1024 * 1024,
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

    # HTTP response validation: reject 4xx/5xx status codes (except 429,
    # which is_blocked handles) and oversized HTML.
    http_result = validate_http_response(result, max_html_size_bytes=max_html_size_bytes)
    if not http_result.valid:
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason=http_result.reason)
        logger.info("Rejected (%s): %s", http_result.reason, item.url)
        return set()

    if mode == "catalog":
        return _store_catalog_card(
            result, item, frontier, storage, main_bucket=main_bucket, tier=tier,
        )

    # HTML content validation: bot challenges, login walls, soft 404s.
    # Runs BEFORE select_content so we reject bad pages without paying the
    # PruningContentFilter extraction cost.
    content_result = validate_html_content(result)
    if not content_result.valid:
        if content_result.blocked:
            # Blocked content (login wall, bot challenge): mark skipped so the
            # frontier row is not left leased. No re-crawl within this run —
            # the next scheduled crawl (fresh frontier) will retry it.
            _ensure_row(frontier, item)
            frontier.mark_skipped(
                item.url, reason=f"blocked:{content_result.reason}",
            )
            logger.info(
                "Blocked (%s), marked skipped: %s",
                content_result.reason, item.url,
            )
            return BLOCKED
        # Permanent rejection: mark skipped, won't re-crawl.
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason=content_result.reason)
        logger.info("Rejected (%s): %s", content_result.reason, item.url)
        return set()

    text, used = select_content(result.markdown)

    # Boilerplate leak check: if "Skip to main content" or similar markers
    # survived PruningContentFilter, the extraction failed and the content is
    # pure nav/chrome.
    if has_boilerplate_leak(text):
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason="boilerplate_leak")
        logger.info("Rejected (boilerplate_leak): %s", item.url)
        return set()

    if looks_empty(text):
        if tier == "fast":
            return NEEDS_JS  # retry in JS tier; leave row leased for now
        _ensure_row(frontier, item)
        frontier.mark_skipped(item.url, reason="empty after JS render")
        logger.debug("Skipped (empty after JS): %s (tier=%s)", item.url, tier)
        return set()

    status_code = getattr(result, "status_code", None) or 200
    content_hash = hashlib.sha256(text.encode()).hexdigest()

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
        page_type=page_type, is_external=False,
        last_modified=headers.get("last-modified"),
    )
    bucket = bucket_for_scope(scope, main_bucket)
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
        is_external=False, hop_history=item.hop_history,
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
    logger.debug(
        "Stored page: %s (scope=%s, status=%d, %d chars, tier=%s)",
        item.url, scope.value, status_code, len(text), tier,
    )

    # Expand: enqueue VT-affiliated children only (external HTML is OUT_OF_SCOPE
    # and is filtered out by classify_url — no EXTERNAL_PAGE path exists).
    if scope is Scope.VT_PAGE and item.depth < max_depth:
        for cat in ("internal", "external"):
            for link in links.get(cat, []):
                href = link.get("href")
                if not href:
                    continue
                child_scope, child_hops = classify_url(
                    href, parent_hops=item.hop_history,
                    host_filter=host_filter, affiliated_matcher=affiliated_matcher,
                )
                if child_scope is Scope.OUT_OF_SCOPE:
                    continue
                # DOCUMENT children are deliberately NOT enqueued: PDFs/DOCX in a
                # browser-render frontier would waste a render (and usually fail).
                # collect_document_links below routes them to the documents pass.
                if child_scope is Scope.DOCUMENT:
                    continue
                child_host = (urlparse(href).hostname or "").lower()
                if host_counts is not None:
                    cap = host_caps.get(child_host, max_urls_per_host) if host_caps else max_urls_per_host
                    if cap != -1 and host_counts.get(child_host, 0) >= cap:
                        continue
                enqueued = frontier.enqueue(
                    href, scope=child_scope.value, depth=item.depth + 1,
                    hop_history=child_hops, discovery_source="bfs",
                )
                if not enqueued:
                    # Within-run duplicate: this URL was already discovered
                    # (or was a seed) earlier in this crawl. Expected and
                    # normal — a department homepage linked from 50 pages
                    # correctly enqueues once. Logged at debug to avoid
                    # flooding logs on a 50k-page crawl.
                    logger.debug("Within-run duplicate URL skipped: %s", href)
                    continue
                if host_counts is not None:
                    host_counts[child_host] = host_counts.get(child_host, 0) + 1
    return collect_document_links(
        result.links,
        affiliated_matcher=affiliated_matcher,
        host_filter=host_filter,
    )


def browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True, text_mode=True, light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )


def make_dispatcher(
    memory_pct: float,
    max_sessions: int,
    *,
    base_delay: tuple[float, float] = (1.0, 3.0),
    max_delay: float = 60.0,
    max_retries: int = 3,
    rate_limit_codes: tuple[int, ...] = (429, 503),
) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=memory_pct,
        check_interval=1.0,
        max_session_permit=max_sessions,
        rate_limiter=RateLimiter(
            base_delay=tuple(base_delay), max_delay=max_delay, max_retries=max_retries,
            rate_limit_codes=list(rate_limit_codes),
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
    max_depth: int,
    batch_size: int,
    memory_pct: float,
    max_sessions: int,
    threshold: float,
    host_filter=None,
    affiliated_matcher=None,
    max_urls_per_host: int = -1,
    host_caps: dict[str, int] | None = None,
    modes: dict[str, str] | None = None,
    batch_idle_timeout: float = 180.0,
    max_wedge_attempts: int = 3,
    run_id: str = "",
    progress_log_interval: int = 10,
    rate_base_delay: tuple[float, float] = (1.0, 3.0),
    rate_max_delay: float = 60.0,
    rate_max_retries: int = 3,
    shutdown_event: threading.Event | None = None,
    max_html_size_bytes: int = 10 * 1024 * 1024,
) -> dict:
    """Lease pending URLs in batches, render (fast tier), checkpoint, retry empties
    on the JS tier, until the frontier has no pending rows.

    `modes` maps registry hosts to "full" | "catalog" | "special"
    (registry.load_modes). Hosts not in the map — including every
    BFS-discovered host — default to "full". None (the default) preserves the
    historical all-full behaviour.

    `batch_idle_timeout` guards against a single page whose render never returns
    wedging the whole crawl: if the result stream produces nothing new for that
    many seconds, the browser is recycled and the crawl continues. The
    un-returned URLs are NOT blanket-failed — they are requeued as pending with
    a per-URL wedge counter and only marked failed after they wedge
    `max_wedge_attempts` times, so one poison page can't drop its innocent
    batch siblings that simply had not streamed yet (BUG-A1).
    """
    stats = {"pages_stored": 0, "pages_failed": 0, "needs_js": 0, "blocked": 0, "documents": 0}
    wedged_batches = 0
    batch_count = 0

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

    # Per-host URL cap tracking. Activated when there is a global cap or any
    # per-host override; counts include resumed frontier rows so restarts cannot
    # blow past the cap.
    host_counts = frontier.count_by_host() if (max_urls_per_host != -1 or host_caps) else None

    # Consecutive process_result failures — reset on each success. Scattered
    # poison pages never accumulate; only a systematic failure (backend gone
    # mid-run) reaches the breaker threshold and aborts the run (issue #5).
    consecutive_failures = 0

    def _handle(result, by_url, tier):
        """Process one streamed result against its leased item. Returns
        (item, outcome), or (None, None) if the result can't be matched.

        process_result is wrapped: if storing/expanding one result raises for
        ANY reason (a poison URL that still produced a MinIO-illegal key, a
        malformed header, an unforeseen edge), that single item is marked
        failed and the crawl continues. A bad item can never propagate out of
        the result loop and kill the run — this is the poison-pill guarantee.
        A systematic run of failures instead trips the circuit breaker below.
        """
        nonlocal consecutive_failures
        item = _match_item(by_url, result)
        if item is None:
            # Unmatchable result: its row stays leased and is recovered by
            # requeue_leased() on the next run.
            logger.warning("No leased item matches result url %s", result.url)
            return None, None
        try:
            outcome = process_result(
                result, item, frontier, storage,
                main_bucket=main_bucket, max_depth=max_depth, tier=tier,
                host_filter=host_filter, affiliated_matcher=affiliated_matcher,
                host_counts=host_counts,
                max_urls_per_host=max_urls_per_host, host_caps=host_caps,
                mode=mode_of(item.url),
                max_html_size_bytes=max_html_size_bytes,
            )
        except StorageFullError:
            # A full storage backend is fatal for the WHOLE run, not a poison
            # item. Let it propagate so run_engine aborts and the Job exits
            # non-zero, instead of marking every page failed and reporting
            # "completed" with nothing stored (issue #5).
            raise
        except Exception as exc:
            logger.error(
                "process_result raised for %s (%s tier) — failing this item and "
                "continuing: %s", item.url, tier, exc,
            )
            _ensure_row(frontier, item)
            frontier.mark_failed(item.url, error=f"process_result: {exc}")
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_WRITE_FAILURES:
                raise RunAborted(
                    f"{consecutive_failures} consecutive process_result failures "
                    "— storage/backend likely unavailable; aborting run"
                )
            return item, set()
        consecutive_failures = 0
        return item, outcome

    def _requeue_unreturned(by_url, handled, reason):
        """Account for leased URLs that a render wedge left un-returned.

        Each such URL is requeued as pending with an incremented wedge counter
        (frontier.requeue_wedged) and only marked failed after it has wedged
        max_wedge_attempts times. A render wedge idles the WHOLE batch's stream,
        but only the genuine straggler should be penalised — its innocent
        siblings that had not streamed yet are retried on a later pass instead
        of being blanket-failed and lost for the run (BUG-A1). URLs already in
        `handled` (streamed a result, incl. NEEDS_JS empties left leased for
        their JS retry) are untouched."""
        for url, item in by_url.items():
            if url not in handled:
                _ensure_row(frontier, item)
                status = frontier.requeue_wedged(
                    url, max_wedges=max_wedge_attempts, error=reason,
                )
                if status == "failed":
                    logger.warning(
                        "Wedged URL failed after %d attempts (%s): %s",
                        max_wedge_attempts, reason, url,
                    )
                else:
                    logger.info("Wedged URL requeued as pending (%s): %s", reason, url)

    drained = False
    while not drained:
        # Graceful shutdown: if SIGTERM/SIGINT set the event, stop leasing new
        # batches. The in-flight batch (if any) was already allowed to finish
        # before we looped back here, so this is a clean drain point — no
        # stranded leased rows beyond what requeue_leased() reclaims next run.
        if shutdown_event is not None and shutdown_event.is_set():
            logger.warning(
                "Shutdown requested — draining engine after %d batches "
                "(%d pages stored, %d failed)", batch_count,
                stats["pages_stored"], stats["pages_failed"],
            )
            drained = True
            break
        # Recover rows stranded 'leased' by a wedge-recycle in a PRIOR pass of
        # this loop: an empty page marked NEEDS_JS is added to `handled`, so when
        # a fast-tier wedge breaks out of the batch before its JS retry runs,
        # _requeue_unreturned skips it and it is left leased forever (its JS
        # render never happens). No batch is in flight here, so every leased row is
        # genuinely orphaned — return them to the pool. No-op on the first pass
        # (the pre-loop requeue above already cleared prior-run strandings).
        restranded = frontier.requeue_leased()
        if restranded:
            logger.info("Requeued %d leased rows stranded by a wedge-recycle", restranded)
        # The crawler (and its dispatcher) is created fresh each time the engine
        # recovers from a wedged batch: exiting the context kills the stuck
        # Chromium processes that the dead batch left behind.
        async with AsyncWebCrawler(config=browser_config()) as crawler:
            dispatcher = make_dispatcher(
                memory_pct, max_sessions,
                base_delay=rate_base_delay, max_delay=rate_max_delay,
                max_retries=rate_max_retries,
            )
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
                        elif outcome == BLOCKED:
                            stats["blocked"] += 1
                        else:
                            document_urls |= outcome
                except _WedgedBatch:
                    _requeue_unreturned(
                        by_url, handled,
                        f"render watchdog: no result for {batch_idle_timeout:.0f}s (fast tier)",
                    )
                    wedged_batches += 1
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
                            # tier="js" never returns the sentinels (empties
                            # are marked skipped, blocked content is re-crawled
                            # next run), but guard anyway so a string can
                            # never pollute the document URL set.
                            if outcome == BLOCKED:
                                stats["blocked"] += 1
                            elif outcome != NEEDS_JS:
                                document_urls |= outcome
                    except _WedgedBatch:
                        _requeue_unreturned(
                            js_by_url, js_handled,
                            f"render watchdog: no result for {batch_idle_timeout:.0f}s (js tier)",
                        )
                        wedged_batches += 1
                        recycle = True
                        break

                # Emit structured batch metrics for log aggregation.
                batch_count += 1
                logger.info(
                    json.dumps({
                        "event": "batch_complete",
                        "run_id": run_id,
                        "batch_count": batch_count,
                        "batch_size": len(render_batch),
                        "js_retries": len(js_retry),
                        "cumulative": {k: v for k, v in stats.items() if k != "document_urls"},
                        "wedged_batches": wedged_batches,
                    })
                )

                # Mid-run progress: emit frontier counts every N batches so
                # admin can watch progress without waiting for completion.
                if progress_log_interval > 0 and batch_count % progress_log_interval == 0:
                    progress_counts = frontier.count_by_status()
                    logger.info(
                        json.dumps({
                            "event": "progress",
                            "run_id": run_id,
                            "batch_count": batch_count,
                            "frontier": progress_counts,
                        })
                    )

    # Frontier counts are the authoritative tally (each result was checkpointed
    # as it streamed in); no per-result counter needed.
    counts = frontier.count_by_status()
    stats["pages_stored"] = counts.get("done", 0)
    stats["pages_failed"] = counts.get("failed", 0)
    stats["pages_skipped"] = counts.get("skipped", 0)
    stats["documents"] = len(document_urls)
    stats["document_urls"] = document_urls
    stats["wedged_batches"] = wedged_batches
    stats["requeued_leased"] = requeued
    return stats
