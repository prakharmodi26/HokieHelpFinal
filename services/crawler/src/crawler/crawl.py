"""Deep crawl orchestration using Crawl4AI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, ContentTypeFilter

from crawler.config import CrawlerConfig
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.storage import MinioStorage

logger = logging.getLogger(__name__)


async def run_crawl(config: CrawlerConfig, storage: MinioStorage) -> dict:
    """Execute a deep crawl and upload each page to storage.

    Returns a stats dict with pages_crawled and pages_failed counts.
    """
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=[config.allowed_domain],
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=config.max_depth,
        include_external=False,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=True,
    )

    stats = {"pages_crawled": 0, "pages_failed": 0}

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            url=config.seed_url, config=run_config
        ):
            if not result.success:
                logger.warning(
                    "Failed to crawl %s: %s",
                    result.url,
                    getattr(result, "error_message", "unknown error"),
                )
                stats["pages_failed"] += 1
                continue

            depth = result.metadata.get("depth", 0)
            title = result.metadata.get("title")
            now = datetime.now(timezone.utc)

            markdown_content = result.markdown.raw_markdown
            if not markdown_content:
                logger.warning("Empty markdown for %s, skipping", result.url)
                stats["pages_failed"] += 1
                continue

            document = build_markdown_document(
                url=result.url,
                title=title,
                markdown_content=markdown_content,
                crawl_depth=depth,
                crawl_timestamp=now,
            )

            object_key = url_to_object_key(result.url)
            storage.upload_document(object_key, document)
            stats["pages_crawled"] += 1
            logger.info(
                "Stored page %d: %s (depth %d)",
                stats["pages_crawled"],
                result.url,
                depth,
            )

    return stats
