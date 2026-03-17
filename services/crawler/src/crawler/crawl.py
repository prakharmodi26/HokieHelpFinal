"""Deep crawl orchestration using Crawl4AI."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from crawler.config import CrawlerConfig
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.storage import MinioStorage

logger = logging.getLogger(__name__)


def _rewrite_to_website(url: str) -> str:
    """Rewrite a cs.vt.edu URL to website.cs.vt.edu."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(netloc="website.cs.vt.edu", scheme="https"))


def _make_markdown_config(prune_threshold: float, request_delay: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        verbose=False,
        delay_before_return_html=request_delay,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=prune_threshold,
                threshold_type="fixed",
            ),
        ),
    )


def _store_result(
    result,
    storage: MinioStorage,
    stats: dict,
    seen_content_hashes: set[str],
    stored_urls: set[str],
    depth: int = 0,
) -> None:
    """Store a successful crawl result; update stats in-place."""
    title = result.metadata.get("title") if result.metadata else None
    now = datetime.now(timezone.utc)

    markdown_content = result.markdown.fit_markdown or result.markdown.raw_markdown
    if not markdown_content:
        logger.warning("Empty markdown for %s, skipping", result.url)
        stats["pages_failed"] += 1
        return

    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()
    if content_hash in seen_content_hashes:
        logger.info("Duplicate/unchanged content for %s, skipping", result.url)
        stats["pages_skipped_duplicate"] += 1
        return
    seen_content_hashes.add(content_hash)

    # Build metadata from all available CrawlResult fields
    response_headers = getattr(result, "response_headers", None) or {}
    doc_id = doc_id_for_url(result.url)
    metadata = PageMetadata(
        doc_id=doc_id,
        url=result.url,
        title=title,
        crawl_depth=depth,
        crawl_timestamp=now,
        content_hash=content_hash,
        markdown_size_bytes=len(markdown_content.encode()),
        status_code=getattr(result, "status_code", None),
        response_headers=response_headers if response_headers else None,
        internal_links=[
            lnk["href"] for lnk in (result.links or {}).get("internal", [])
            if lnk.get("href")
        ],
        external_links=[
            lnk["href"] for lnk in (result.links or {}).get("external", [])
            if lnk.get("href")
        ],
        last_modified=response_headers.get("last-modified"),
        etag=response_headers.get("etag"),
    )

    document = build_markdown_document(
        url=result.url,
        title=title,
        markdown_content=markdown_content,
        crawl_depth=depth,
        crawl_timestamp=now,
        doc_id=doc_id,
        content_hash=content_hash,
    )
    object_key = url_to_object_key(result.url)
    storage.upload_document(object_key, document)
    storage.upload_metadata(object_key, metadata)
    stored_urls.add(result.url)
    stats["pages_crawled"] += 1
    logger.info("Stored page %d: %s (depth %d)", stats["pages_crawled"], result.url, depth)


async def run_crawl(config: CrawlerConfig, storage: MinioStorage) -> dict:
    """Execute a deep crawl and upload each page to storage.

    Returns a stats dict with pages_crawled, pages_failed, pages_skipped_duplicate counts.

    Two-phase strategy:
    1. BFS crawl of website.cs.vt.edu. Any cs.vt.edu redirect results are rewritten
       to website.cs.vt.edu and queued for phase 2. External links to *.cs.vt.edu
       subdomains are also queued.
    2. Individual fetches for queued URLs. If they redirect to cs.vt.edu/path, the
       path is rewritten to website.cs.vt.edu/path and fetched again.
    """
    # Phase 1: BFS crawl of website.cs.vt.edu
    # Include cs.vt.edu in filter so redirect chains aren't cut mid-hop.
    crawl_allowed = list(config.allowed_domains)
    if "website.cs.vt.edu" in crawl_allowed and "cs.vt.edu" not in crawl_allowed:
        crawl_allowed.append("cs.vt.edu")

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=crawl_allowed,
            blocked_domains=list(config.blocked_domains),
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=config.max_depth,
        include_external=False,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
    )

    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=True,
        delay_before_return_html=config.request_delay,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=config.prune_threshold,
                threshold_type="fixed",
            ),
        ),
    )

    stats = {"pages_crawled": 0, "pages_failed": 0, "pages_skipped_duplicate": 0}
    # Seed from previously stored hashes to enable incremental recrawl
    seen_content_hashes: set[str] = set(storage.load_all_content_hashes().keys())
    stored_urls: set[str] = set()
    # URLs to individually fetch in phase 2 (subdomains + cs.vt.edu rewrites)
    pending_fetches: set[str] = set()

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=config.seed_url, config=bfs_config):
            if not result.success:
                logger.warning(
                    "Failed to crawl %s: %s",
                    result.url,
                    getattr(result, "error_message", "unknown error"),
                )
                stats["pages_failed"] += 1
                continue

            final_host = urlparse(result.url).hostname

            # Page landed on cs.vt.edu — rewrite and queue for phase 2
            if final_host == "cs.vt.edu":
                rewritten = _rewrite_to_website(result.url)
                if rewritten not in stored_urls:
                    logger.info(
                        "Redirect to cs.vt.edu detected: %s → queuing %s",
                        result.url,
                        rewritten,
                    )
                    pending_fetches.add(rewritten)
                stats["pages_failed"] += 1
                continue

            # Other non-allowed domain — skip
            if final_host not in config.allowed_domains:
                logger.info(
                    "Skipping %s — final domain %s not in allowed_domains",
                    result.url,
                    final_host,
                )
                stats["pages_failed"] += 1
                continue

            depth = result.metadata.get("depth", 0)
            _store_result(result, storage, stats, seen_content_hashes, stored_urls, depth)

            # Scan external links for *.cs.vt.edu subdomains not in blocked_domains
            for link in (result.links or {}).get("external", []):
                href = link.get("href", "")
                if not href:
                    continue
                parsed = urlparse(href)
                host = parsed.hostname or ""
                if (
                    host.endswith(".cs.vt.edu")
                    and host != "website.cs.vt.edu"
                    and host not in config.blocked_domains
                    and href not in pending_fetches
                    and href not in stored_urls
                ):
                    logger.info("Queuing external cs.vt.edu link for phase 2: %s", href)
                    pending_fetches.add(href)

        # Phase 2: individually fetch queued URLs, following cs.vt.edu redirects
        if pending_fetches:
            logger.info("Phase 2: fetching %d queued URLs", len(pending_fetches))
            single_config = _make_markdown_config(config.prune_threshold, config.request_delay)

            for url in pending_fetches:
                if url in stored_urls:
                    continue

                result = await crawler.arun(url=url, config=single_config)
                if not result.success:
                    logger.warning("Phase 2 failed: %s — %s", url, getattr(result, "error_message", ""))
                    stats["pages_failed"] += 1
                    continue

                final_host = urlparse(result.url).hostname

                # Redirect landed on cs.vt.edu — rewrite and fetch the website equivalent
                if final_host == "cs.vt.edu":
                    rewritten = _rewrite_to_website(result.url)
                    if rewritten in stored_urls:
                        continue
                    logger.info("Phase 2 rewrite: %s → %s", result.url, rewritten)
                    result = await crawler.arun(url=rewritten, config=single_config)
                    if not result.success:
                        logger.warning("Phase 2 rewrite fetch failed: %s", rewritten)
                        stats["pages_failed"] += 1
                        continue
                    final_host = urlparse(result.url).hostname

                if final_host not in config.allowed_domains:
                    logger.info("Phase 2 skipping %s — domain %s not allowed", result.url, final_host)
                    stats["pages_failed"] += 1
                    continue

                _store_result(result, storage, stats, seen_content_hashes, stored_urls, depth=0)

    logger.info(
        "Deduplication: %d duplicate pages skipped",
        stats["pages_skipped_duplicate"],
    )
    return stats
