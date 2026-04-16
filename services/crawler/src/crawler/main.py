"""Crawler entrypoint — wire config, storage, and crawl together."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
from datetime import datetime, timezone

from crawler.cleaner import build_department_info_doc, clean_markdown, is_error_page
from crawler.config import CrawlerConfig
from crawler.crawl import run_crawl
from crawler.storage import MinioStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_BUCKET = "logs"


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
    stats, visit_log = asyncio.run(run_crawl(config, storage))

    # Upload visit log to logs bucket
    storage.ensure_bucket(LOGS_BUCKET)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_key = f"websites-visited-{timestamp}.md"
    storage.upload_document(log_key, visit_log.render(), bucket=LOGS_BUCKET)
    logger.info("Visit log uploaded to %s/%s", LOGS_BUCKET, log_key)

    logger.info(
        "Crawl complete: %d pages stored, %d failed, %d documents processed, %d documents failed",
        stats["pages_crawled"],
        stats["pages_failed"],
        stats.get("documents_processed", 0),
        stats.get("documents_failed", 0),
    )

    # Post-crawl: clean all raw pages and upload to cleaned bucket
    logger.info("Starting post-crawl cleaning → %s", config.minio_cleaned_bucket)
    storage.ensure_bucket(config.minio_cleaned_bucket)

    raw_keys = storage.list_objects()
    cleaned_count = 0
    error_count = 0
    for key in raw_keys:
        raw_content = storage.download_document(key)
        cleaned_content = clean_markdown(raw_content)
        if is_error_page(cleaned_content):
            logger.debug("Skipping CMS error page: %s", key)
            error_count += 1
            continue
        storage.upload_document(key, cleaned_content, bucket=config.minio_cleaned_bucket)

        # Mirror metadata sidecar with updated markdown_size_bytes
        try:
            raw_meta = storage.download_metadata(key)
            cleaned_meta = dataclasses.replace(
                raw_meta,
                markdown_size_bytes=len(cleaned_content.encode("utf-8")),
            )
            storage.upload_metadata(key, cleaned_meta, bucket=config.minio_cleaned_bucket)
        except Exception as exc:
            logger.debug("No metadata sidecar for %s, skipping mirror: %s", key, exc)

        cleaned_count += 1

    # Store department info doc so RAG has campus addresses, phones, social links
    dept_info = build_department_info_doc()
    storage.upload_document(
        "_department-info.md", dept_info, bucket=config.minio_cleaned_bucket
    )
    logger.info(
        "Cleaning complete: %d pages cleaned, %d error pages skipped + department info stored in %s",
        cleaned_count,
        error_count,
        config.minio_cleaned_bucket,
    )


if __name__ == "__main__":
    cli()
