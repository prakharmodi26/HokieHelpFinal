"""MinIO storage layer — reads chunk JSON files from the chunks bucket."""
from __future__ import annotations

import json
import logging
from typing import List

from minio import Minio

from embedder.config import EmbedderConfig

logger = logging.getLogger(__name__)


class EmbedderStorage:
    """Reads chunk JSON files from MinIO."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._bucket = config.minio_chunks_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )

    def list_chunk_keys(self) -> List[str]:
        """Return all chunks/*.json object keys."""
        return [
            obj.object_name
            for obj in self._client.list_objects(self._bucket, prefix="chunks/", recursive=True)
            if obj.object_name.endswith(".json")
        ]

    def download_chunks(self, key: str) -> List[dict]:
        """Download and parse a chunk JSON file. Returns list of chunk dicts."""
        response = self._client.get_object(self._bucket, key)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
