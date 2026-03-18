"""Embedder entrypoint — read chunks, embed, index into Qdrant."""
from __future__ import annotations

import logging
import sys
from typing import Any

from embedder.config import EmbedderConfig
from embedder.embedder import Embedder, build_context
from embedder.indexer import QdrantIndexer
from embedder.storage import EmbedderStorage
from embedder.validator import validate_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_embedding(
    storage: Any,
    embedder: Any,
    indexer: Any,
    config: EmbedderConfig,
) -> dict:
    """Process all chunk files: validate -> embed -> upsert -> clean stale.

    Returns stats dict.
    """
    keys = storage.list_chunk_keys()
    logger.info("Found %d chunk files to process", len(keys))

    stats = {
        "docs_processed": 0,
        "chunks_embedded": 0,
        "chunks_skipped": 0,
        "stale_deleted": 0,
        "failed": 0,
    }

    for key in keys:
        try:
            raw_chunks = storage.download_chunks(key)
            result = validate_chunks(raw_chunks)

            for _, reason in result.invalid:
                logger.warning("Skipped invalid chunk in %s: %s", key, reason)
            stats["chunks_skipped"] += len(result.invalid)

            if not result.valid:
                logger.debug("No valid chunks in %s, skipping", key)
                continue

            # Build contexts and embed
            contexts = [build_context(c) for c in result.valid]

            # Batch embedding according to config
            all_embeddings = []
            batch_size = config.embedding_batch_size
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i : i + batch_size]
                all_embeddings.extend(embedder.embed_batch(batch))

            # Upsert into Qdrant
            indexer.upsert_chunks(result.valid, all_embeddings)

            # Delete stale chunks for this document
            doc_id = result.valid[0]["document_id"]
            current_ids = {c["chunk_id"] for c in result.valid}
            deleted = indexer.delete_stale_chunks(doc_id, current_ids)

            stats["docs_processed"] += 1
            stats["chunks_embedded"] += len(result.valid)
            stats["stale_deleted"] += deleted
            logger.info(
                "Indexed %s -> doc=%s, %d embedded, %d stale deleted",
                key, doc_id, len(result.valid), deleted,
            )

        except Exception as exc:
            logger.error("Failed to process %s: %s", key, exc)
            stats["failed"] += 1

    logger.info(
        "Embedding complete: %d docs, %d chunks embedded, %d skipped, %d stale deleted, %d failed",
        stats["docs_processed"],
        stats["chunks_embedded"],
        stats["chunks_skipped"],
        stats["stale_deleted"],
        stats["failed"],
    )
    return stats


def cli() -> None:
    """CLI entrypoint for the embedder service."""
    logger.info("Starting HokieHelp embedder")

    try:
        config = EmbedderConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = EmbedderStorage(config)
    embedder = Embedder(config.embedding_model)
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection=config.qdrant_collection,
        vector_size=embedder.dimension,
    )

    stats = run_embedding(storage, embedder, indexer, config)

    if stats["failed"] > 0:
        logger.warning("%d documents failed", stats["failed"])
        sys.exit(1)


if __name__ == "__main__":
    cli()
