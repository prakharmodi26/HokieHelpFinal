"""Chunker entrypoint — read cleaned docs, chunk, write chunk records."""
from __future__ import annotations

import logging
import sys

from chunker.config import ChunkerConfig
from chunker.parser import parse_frontmatter, split_sections
from chunker.splitter import build_chunks
from chunker.storage import ChunkerStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_chunking(storage: ChunkerStorage, config: ChunkerConfig) -> dict:
    """Process all cleaned documents and write chunk records.

    Returns a stats dict with keys: processed, skipped, total_chunks, failed.
    """
    keys = storage.list_markdown_keys()
    logger.info("Found %d markdown documents to chunk", len(keys))

    stats = {"processed": 0, "skipped": 0, "total_chunks": 0, "failed": 0}

    for key in keys:
        try:
            content = storage.download_document(key)
            frontmatter, body = parse_frontmatter(content)

            if not body.strip():
                logger.debug("Skipping %s — empty body", key)
                stats["skipped"] += 1
                continue

            sections = split_sections(body)
            if not sections:
                logger.debug("Skipping %s — no sections after split", key)
                stats["skipped"] += 1
                continue

            chunks = build_chunks(sections, frontmatter, config)
            storage.upload_chunks(frontmatter.doc_id, chunks)

            stats["processed"] += 1
            stats["total_chunks"] += len(chunks)
            logger.info("Chunked %s → doc_id=%s, %d chunks", key, frontmatter.doc_id, len(chunks))

        except Exception as exc:
            logger.error("Failed to chunk %s: %s", key, exc)
            stats["failed"] += 1

    logger.info(
        "Chunking complete: %d processed, %d skipped, %d total chunks, %d failed",
        stats["processed"], stats["skipped"], stats["total_chunks"], stats["failed"],
    )
    return stats


def cli() -> None:
    """CLI entrypoint for the chunker service."""
    logger.info("Starting HokieHelp chunker")

    try:
        config = ChunkerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = ChunkerStorage(config)
    stats = run_chunking(storage, config)

    if stats["failed"] > 0:
        logger.warning("%d documents failed to chunk", stats["failed"])
        sys.exit(1)


if __name__ == "__main__":
    cli()
