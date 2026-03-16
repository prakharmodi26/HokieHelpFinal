"""MinIO object storage client for uploading crawled documents."""

from __future__ import annotations

import io
import logging

from minio import Minio

from crawler.config import CrawlerConfig

logger = logging.getLogger(__name__)


class MinioStorage:
    """Uploads Markdown documents to MinIO."""

    def __init__(self, config: CrawlerConfig) -> None:
        self._bucket = config.minio_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            logger.info("Created bucket %s", self._bucket)

    def upload_document(self, object_key: str, content: str) -> None:
        """Upload a Markdown document to MinIO.

        Args:
            object_key: The object path within the bucket (e.g. cs.vt.edu/index.md)
            content: The full Markdown document string
        """
        data = content.encode("utf-8")
        self._client.put_object(
            self._bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type="text/markdown",
        )
        logger.info("Uploaded %s (%d bytes)", object_key, len(data))
