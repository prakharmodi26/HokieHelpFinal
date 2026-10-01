"""MinIO object storage client for uploading crawled documents.

All MinIO operations are wrapped in a retry-with-backoff layer (``_with_retry``)
that retries transient failures (connection resets, timeouts, 503s, brief disk
pressure) up to ``_MAX_RETRIES`` times with exponential backoff, and re-raises
permanent failures (4xx auth/argument errors, NoSuchBucket/Key) immediately so
callers can log-and-skip. This prevents a single transient blip from killing
the entire crawl run.
"""

from __future__ import annotations

import io
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

from crawler.config import CrawlerConfig
from crawler.metadata import PageMetadata, metadata_key_for

logger = logging.getLogger(__name__)

# ── Retry configuration ────────────────────────────────────────────────────
# Tunable constants. Transient failures are retried with exponential backoff
# (base * 2^attempt) plus up to 25% jitter so concurrent writers don't thunder.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5

# S3 error codes that are safe to retry (server-side transient conditions).
# Anything 5xx or matching these codes is retried; 4xx and NoSuch*/AccessDenied
# are permanent and re-raised immediately.
_RETRYABLE_S3_CODES = frozenset({
    "InternalError",
    "RequestTimeout",
    "SlowDown",
    "ServiceUnavailable",
    "RequestTimeTooSkewed",
    "Throttling",
    "OperationAborted",
})

# S3 error codes (and http status ranges) that are permanent — never retry.
# NoSuchBucket/NoSuchKey/AccessDenied/Invalid* propagate to the caller.
_PERMANENT_S3_CODES = frozenset({
    "NoSuchBucket",
    "NoSuchKey",
    "AccessDenied",
    "InvalidAccessKeyId",
    "InvalidBucketName",
    "InvalidObjectName",
    "InvalidArgument",
    "SignatureDoesNotMatch",
})


# Codes/status meaning the storage backend is out of space. FATAL, not
# transient: retrying can't free the disk, and swallowing it per-item makes the
# whole run "succeed" while storing nothing (the issue-#5 silent green build).
_STORAGE_FULL_CODES = frozenset({"XMinioStorageFull", "InsufficientStorage"})


class StorageFullError(RuntimeError):
    """Raised when MinIO reports the backend is out of space (HTTP 507).

    Deliberately propagated past the engine's per-item poison-pill handler so a
    full backend aborts the run loudly (non-zero exit) instead of being logged
    as one failed item while the run reports success.
    """


def _is_storage_full(exc: Exception) -> bool:
    """True if *exc* is a MinIO 'backend out of space' error (507)."""
    if not isinstance(exc, S3Error):
        return False
    code = getattr(exc, "code", None) or ""
    if code in _STORAGE_FULL_CODES:
        return True
    status = getattr(getattr(exc, "response", None), "status", None)
    return status == 507


def _classify_error(exc: Exception) -> bool:
    """Return True if *exc* is transient (worth retrying), False if permanent.

    Heuristic:
      - ``S3Error``: inspect ``code`` (retryable codes → transient; permanent
        codes → not transient; unknown codes fall back to HTTP status: 5xx →
        transient, 4xx → permanent).
      - Connection-level errors (``ConnectionError``, ``OSError``,
        ``MaxRetryError``) → transient.
      - Anything else → not transient (don't retry unknown failures forever).
    """
    if isinstance(exc, S3Error):
        code = getattr(exc, "code", None) or ""
        if code in _RETRYABLE_S3_CODES:
            return True
        if code in _PERMANENT_S3_CODES:
            return False
        # Unknown code: fall back to the real HTTP status. S3Error exposes it on
        # ``.response.status`` — there is NO ``.http_code`` attribute (that made
        # both branches below dead code, forcing every unknown error transient).
        status = getattr(getattr(exc, "response", None), "status", None) or 0
        if 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False
        # No code and no http_code — treat as transient (one retry won't hurt).
        return True
    if isinstance(exc, (ConnectionError, OSError, MaxRetryError)):
        return True
    return False


def _with_retry(operation_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute *fn* with retry-on-transient-failure.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff
    (base * 2^attempt + jitter). Permanent failures (per ``_classify_error``)
    and any error exhausted on retries are re-raaised to the caller.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_storage_full(exc):
                # Fatal: retrying cannot free the disk. Raise a distinct error
                # so the engine aborts the run instead of marking one item
                # failed and marching on to a false "completed" (issue #5).
                logger.error(
                    "%s hit storage-full — aborting run: %s", operation_name, exc
                )
                raise StorageFullError(str(exc)) from exc
            if not _classify_error(exc):
                # Permanent failure — don't waste retries.
                logger.error(
                    "%s failed permanently (not retryable): %s", operation_name, exc
                )
                raise
            if attempt >= _MAX_RETRIES:
                logger.error(
                    "%s failed after %d retries: %s",
                    operation_name, _MAX_RETRIES, exc,
                )
                raise
            delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.25)
            logger.warning(
                "%s transient failure (attempt %d/%d), retrying in %.2fs: %s",
                operation_name, attempt + 1, _MAX_RETRIES + 1, delay, exc,
            )
            time.sleep(delay)
    # Unreachable — the loop either returns or raises. Defensive only.
    assert last_exc is not None
    raise last_exc


class MinioStorage:
    """Uploads Markdown documents to MinIO."""

    def __init__(self, config: CrawlerConfig) -> None:
        self._bucket = config.minio_data_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self._ensure_bucket(self._bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        exists = _with_retry(
            f"bucket_exists({bucket})", self._client.bucket_exists, bucket,
        )
        if not exists:
            _with_retry(f"make_bucket({bucket})", self._client.make_bucket, bucket)
            logger.info("Created bucket %s", bucket)

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure a bucket exists, creating it if needed."""
        self._ensure_bucket(bucket)

    def upload_document(self, object_key: str, content: str, bucket: str | None = None) -> None:
        """Upload a Markdown document to MinIO."""
        target = bucket or self._bucket
        data = content.encode("utf-8")
        _with_retry(
            f"put_object({target}/{object_key})",
            self._client.put_object,
            target,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type="text/markdown",
        )
        logger.info("Uploaded %s (%d bytes) to %s", object_key, len(data), target)

    def upload_raw_bytes(
        self, object_key: str, data: bytes, content_type: str, bucket: str | None = None
    ) -> None:
        """Upload raw binary data (e.g. a PDF or DOCX file) to MinIO."""
        target = bucket or self._bucket
        _with_retry(
            f"put_object({target}/{object_key})",
            self._client.put_object,
            target,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        logger.info("Uploaded raw %s (%d bytes) to %s", object_key, len(data), target)

    def list_objects(self, bucket: str | None = None) -> list[str]:
        """List all object keys in the given bucket."""
        target = bucket or self._bucket
        objects = _with_retry(
            f"list_objects({target})", self._client.list_objects, target, recursive=True,
        )
        return [obj.object_name for obj in objects]

    def download_document(self, object_key: str, bucket: str | None = None) -> str:
        """Download a document and return its content as string."""
        target = bucket or self._bucket
        response = _with_retry(
            f"get_object({target}/{object_key})", self._client.get_object, target, object_key,
        )
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
        _with_retry(
            f"put_object({target}/{meta_key})",
            self._client.put_object,
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
        response = _with_retry(
            f"get_object({target}/{meta_key})", self._client.get_object, target, meta_key,
        )
        try:
            return PageMetadata.from_json(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
