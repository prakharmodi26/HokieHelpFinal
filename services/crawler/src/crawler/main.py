"""Crawler entrypoint — wire config, storage, and crawl together."""

from __future__ import annotations

import asyncio
import logging
import sys

from crawler.config import CrawlerConfig
from crawler.crawl import run_crawl
from crawler.storage import MinioStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cli() -> None:
    """CLI entrypoint for the crawler."""
    logger.info("Starting HokieHelp crawler")

    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    logger.info(
        "Crawling %s (max_depth=%d, max_pages=%d)",
        config.seed_url,
        config.max_depth,
        config.max_pages,
    )

    storage = MinioStorage(config)
    stats = asyncio.run(run_crawl(config, storage))

    logger.info(
        "Crawl complete: %d pages stored, %d failed",
        stats["pages_crawled"],
        stats["pages_failed"],
    )


if __name__ == "__main__":
    cli()
