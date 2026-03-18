"""Qdrant vector database indexer."""
from __future__ import annotations

import logging
import uuid
from typing import List, Set

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id string to a deterministic UUID for Qdrant point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantIndexer:
    """Manages a Qdrant collection: create, upsert, delete stale."""

    def __init__(self, host: str, port: int, collection: str, vector_size: int) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection with cosine similarity if it does not exist."""
        existing = self._client.get_collections().collections
        if any(c.name == self._collection for c in existing):
            logger.info("Collection %s already exists", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection %s (dim=%d, cosine)", self._collection, self._vector_size)

    def upsert_chunks(self, chunks: List[dict], embeddings: List[List[float]]) -> None:
        """Upsert chunk embeddings as Qdrant points."""
        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(models.PointStruct(
                id=chunk_id_to_uuid(chunk["chunk_id"]),
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "page_type": chunk.get("page_type", "general"),
                    "headings_path": chunk.get("headings_path", []),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content_hash": chunk.get("content_hash", ""),
                    "crawl_timestamp": chunk.get("crawl_timestamp", ""),
                    "token_count": chunk.get("token_count", 0),
                    "text": chunk.get("text", ""),
                },
            ))
        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.debug("Upserted %d points to %s", len(points), self._collection)

    def delete_stale_chunks(self, document_id: str, current_chunk_ids: Set[str]) -> int:
        """Delete Qdrant points for document_id not in current_chunk_ids."""
        current_uuids = {chunk_id_to_uuid(cid) for cid in current_chunk_ids}

        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )]
            ),
            limit=10000,
        )

        stale_ids = [p.id for p in results if p.id not in current_uuids]
        if stale_ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=stale_ids),
            )
            logger.info("Deleted %d stale points for doc %s", len(stale_ids), document_id)
        return len(stale_ids)
