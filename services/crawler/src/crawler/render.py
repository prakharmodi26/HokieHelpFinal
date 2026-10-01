"""Two-tier render configs + scope→bucket routing for the crawl engine.

Fast tier: quick render with generic fit_markdown extraction, robots respected.
JS tier:   for pages that came back empty (SPAs, lazy-loaded dining menus) —
           scans the full page, waits for late JS, longer timeout.

Routing:   VT pages and documents → main bucket (chunker input).
           External HTML is OUT_OF_SCOPE and never reaches this function;
           the external bucket no longer exists.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterable, Sequence

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from crawler.extract import make_markdown_generator
from crawler.scope import Scope

logger = logging.getLogger(__name__)

# Chunked-render defaults. Small chunks bound the blast radius of a wedged page
# (only its own chunk stalls), give per-chunk progress visibility, and let the
# caller persist results incrementally instead of one all-or-nothing batch.
DEFAULT_RENDER_CHUNK_SIZE = 12
DEFAULT_RENDER_IDLE_TIMEOUT = 120.0


def fast_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "header", "footer", "aside", "script", "style", "noscript", "iframe"],
        remove_overlay_elements=True,
        max_retries=2,
        page_timeout=30000,
        delay_before_return_html=0.3,
        stream=True,
        verbose=False,
    )


def js_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "header", "footer", "aside", "script", "style", "noscript", "iframe"],
        remove_overlay_elements=True,
        max_retries=2,
        scan_full_page=True,
        scroll_delay=0.5,
        wait_for_images=True,
        delay_before_return_html=1.0,
        page_timeout=90000,
        stream=True,
        verbose=False,
    )


class WedgedChunk(Exception):
    """Raised when a render chunk's result stream goes idle past the watchdog
    timeout. A single page whose render never returns (binary downloads with no
    extension, SPA pages with live websockets) can hang an arun_many stream
    forever — page_timeout only bounds navigation, not post-load scans/waits.
    This is the recovery signal; the stuck stream is closed before it is raised.

    Self-contained here rather than imported from engine.py so the live main
    crawl is never coupled to changes in the catalog render path.
    """


async def _iter_with_idle_timeout(results, idle_timeout: float):
    """Yield results from an arun_many stream, raising WedgedChunk if no new
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
            raise WedgedChunk
        yield item


def _chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


async def render_in_chunks(
    urls: Sequence[str],
    config: CrawlerRunConfig,
    *,
    chunk_size: int = DEFAULT_RENDER_CHUNK_SIZE,
    idle_timeout: float = DEFAULT_RENDER_IDLE_TIMEOUT,
    max_retries: int = 0,
    crawler_factory=None,
) -> AsyncIterator[tuple[str, str]]:
    """Render URLs in fixed-size chunks, yielding ``(url, html)`` as each
    successful result streams in.

    Each chunk renders under a fresh ``AsyncWebCrawler`` so that exiting the
    context kills any Chromium left stuck by a wedged page. Every chunk is
    guarded by an idle watchdog: if no new result arrives within
    ``idle_timeout`` seconds, the chunk's un-returned URLs are deferred (not
    abandoned outright) and rendering continues with the next chunk. A single
    hung, stale, or mismatched URL can therefore stall at most its own chunk —
    never the whole harvest.

    Failed (non-success) results and watchdog-deferred URLs are collected into
    a retry pool. After a full pass, up to ``max_retries`` additional passes
    re-render just the pool, so a whole subject page (an entire course set) is
    not silently lost to one transient render hiccup. A URL that already
    succeeded is never re-rendered. URLs still failing after the final pass are
    logged as a genuine give-up accounting (nothing vanishes silently).
    ``max_retries=0`` (the default) preserves single-pass behaviour.

    ``crawler_factory`` is a test seam returning an async-context-manager
    crawler exposing ``arun_many(urls, config)``; production uses crawl4ai.
    """
    pending = list(urls)
    if not pending:
        return
    make_crawler = crawler_factory or (lambda: AsyncWebCrawler())
    attempt = 0
    while pending:
        chunks = list(_chunked(pending, max(1, chunk_size)))
        total = len(chunks)
        retry_pool: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            seen: set[str] = set()
            async with make_crawler() as crawler:
                results = await crawler.arun_many(urls=list(chunk), config=config)
                try:
                    async for result in _iter_with_idle_timeout(results, idle_timeout):
                        url = getattr(result, "url", "")
                        # First result per URL wins: ignore a duplicate result
                        # for the same URL within a chunk so it can't be both
                        # yielded and requeued (no double-yield across retries).
                        if url in seen:
                            continue
                        seen.add(url)
                        if getattr(result, "success", False):
                            yield url, getattr(result, "html", "") or ""
                        else:
                            retry_pool.append(url)
                            logger.warning(
                                "Render failed for %s: %s", url,
                                getattr(result, "error_message", "unknown"),
                            )
                except WedgedChunk:
                    stuck = [u for u in chunk if u not in seen]
                    retry_pool.extend(stuck)
                    logger.warning(
                        "Render watchdog: chunk %d/%d idle >%.0fs; deferring %d "
                        "un-returned URL(s): %s",
                        idx, total, idle_timeout, len(stuck), stuck,
                    )
            logger.info("Rendered chunk %d/%d (%d URLs)", idx, total, len(chunk))

        if not retry_pool:
            return
        if attempt >= max_retries:
            logger.warning(
                "Render giving up on %d URL(s) after %d attempt(s): %s",
                len(retry_pool), attempt + 1, retry_pool,
            )
            return
        attempt += 1
        logger.info(
            "Render retry %d/%d: re-rendering %d failed/deferred URL(s)",
            attempt, max_retries, len(retry_pool),
        )
        pending = retry_pool


def bucket_for_scope(scope: Scope, main_bucket: str) -> str:
    """Return the storage bucket for the given scope.

    All active scopes (VT_PAGE, DOCUMENT, and the legacy EXTERNAL_PAGE
    sentinel) route to the main bucket.  The external bucket is removed;
    external HTML is OUT_OF_SCOPE before reaching this function.
    """
    return main_bucket
