"""Create Qdrant keyword payload indexes used by retrieval filters.

Idempotent migration for existing collections:

    QDRANT_HOST=localhost QDRANT_COLLECTION=hokiehelp_chunks python scripts/migrate_payload_indexes.py
"""
from __future__ import annotations

import argparse
import logging
import os

from qdrant_client import QdrantClient, models


KEYWORD_FIELDS = (
    "source_host",
    "subdomain",
    "page_type",
    "subject",
    "course_level",
)

logger = logging.getLogger(__name__)


def _is_already_exists_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "already has" in message


def create_keyword_indexes(client: QdrantClient, collection: str) -> None:
    for field_name in KEYWORD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created keyword payload index: %s", field_name)
        except Exception as exc:
            if _is_already_exists_error(exc):
                logger.info("Keyword payload index already exists: %s", field_name)
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QDRANT_PORT", "6333")))
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = QdrantClient(host=args.host, port=args.port)
    create_keyword_indexes(client, args.collection)


if __name__ == "__main__":
    main()
