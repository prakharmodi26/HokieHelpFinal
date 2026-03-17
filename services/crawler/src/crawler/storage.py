"""MinIO object storage client for uploading crawled documents."""

from __future__ import annotations

import io
import logging

from minio import Minio

from crawler.config import CrawlerConfig
from crawler.metadata import PageMetadata, metadata_key_for

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
        self._ensure_bucket(self._bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info("Created bucket %s", bucket)

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure a bucket exists, creating it if needed."""
        self._ensure_bucket(bucket)

    def upload_document(self, object_key: str, content: str, bucket: str | None = None) -> None:
        """Upload a Markdown document to MinIO."""
        target = bucket or self._bucket
        data = content.encode("utf-8")
        self._client.put_object(
            target,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type="text/markdown",
        )
        logger.info("Uploaded %s (%d bytes) to %s", object_key, len(data), target)

    def list_objects(self, bucket: str | None = None) -> list[str]:
        """List all object keys in the given bucket."""
        target = bucket or self._bucket
        return [
            obj.object_name
            for obj in self._client.list_objects(target, recursive=True)
        ]

    def download_document(self, object_key: str, bucket: str | None = None) -> str:
        """Download a document and return its content as string."""
        target = bucket or self._bucket
        response = self._client.get_object(target, object_key)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()

    def upload_metadata(
        self, object_key: str, metadata: PageMetadata, bucket: str | None = None
    ) -> None:
        """Upload a .meta.json sidecar for a .md object key."""
        target = bucket or self._bucket
        meta_key = metadata_key_for(object_key)
        json_bytes = metadata.to_json().encode("utf-8")
        self._client.put_object(
            target,
            meta_key,
            io.BytesIO(json_bytes),
            length=len(json_bytes),
            content_type="application/json",
        )
        logger.info("Uploaded %s (%d bytes) to %s", meta_key, len(json_bytes), target)

    def download_metadata(
        self, object_key: str, bucket: str | None = None
    ) -> PageMetadata:
        """Download and deserialise a .meta.json sidecar."""
        target = bucket or self._bucket
        meta_key = metadata_key_for(object_key)
        response = self._client.get_object(target, meta_key)
        try:
            return PageMetadata.from_json(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def load_all_content_hashes(
        self, bucket: str | None = None
    ) -> dict[str, str]:
        """Return {content_hash: url} for every stored .meta.json in the bucket."""
        target = bucket or self._bucket
        result: dict[str, str] = {}
        for obj in self._client.list_objects(target, recursive=True):
            if not obj.object_name.endswith(".meta.json"):
                continue
            try:
                meta = self.download_metadata(
                    obj.object_name.replace(".meta.json", ".md"), bucket=target
                )
                result[meta.content_hash] = meta.url
            except Exception as exc:
                logger.warning("Could not load metadata %s: %s", obj.object_name, exc)
        return result
