"""MinIO storage layer for the chunker service."""
from __future__ import annotations

import io
import json
import logging
from typing import List

from minio import Minio

from chunker.config import ChunkerConfig
from chunker.models import ChunkRecord

logger = logging.getLogger(__name__)

# Maximum size of a single markdown document (10 MB)
MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024


class ChunkerStorage:
    """Reads cleaned markdown from MinIO; writes chunk JSON to MinIO."""

    def __init__(self, config: ChunkerConfig) -> None:
        self._cleaned_bucket = config.minio_cleaned_bucket
        self._chunks_bucket = config.minio_chunks_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self._ensure_bucket(self._chunks_bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info("Created bucket %s", bucket)

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure a bucket exists, creating it if needed (public, mirrors MinioStorage)."""
        self._ensure_bucket(bucket)

    def list_markdown_keys(self) -> List[str]:
        """Return all .md object keys from the cleaned bucket (excludes .meta.json)."""
        return [
            obj.object_name
            for obj in self._client.list_objects(self._cleaned_bucket, recursive=True)
            if obj.object_name.endswith(".md")
        ]

    def download_document(self, key: str) -> str:
        """Download a markdown document from the cleaned bucket.

        Raises ValueError if the document exceeds MAX_DOCUMENT_SIZE_BYTES.
        """
        response = self._client.get_object(self._cleaned_bucket, key)
        try:
            data = response.read()
            if len(data) > MAX_DOCUMENT_SIZE_BYTES:
                raise ValueError(
                    f"Document {key} is {len(data)} bytes, exceeds limit of {MAX_DOCUMENT_SIZE_BYTES}"
                )
            return data.decode("utf-8")
        finally:
            response.close()
            response.release_conn()

    def upload_chunks(self, doc_id: str, chunks: List[ChunkRecord]) -> None:
        """Upload all chunks for a document as chunks/<doc_id>.json."""
        key = f"chunks/{doc_id}.json"
        payload = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2)
        data = payload.encode("utf-8")
        self._client.put_object(
            self._chunks_bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        logger.info("Uploaded %s (%d chunks, %d bytes) to %s", key, len(chunks), len(data), self._chunks_bucket)
