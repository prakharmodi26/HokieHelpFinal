from __future__ import annotations

import asyncio
import logging
from typing import Any

from minio import Minio
from admin.config import AdminConfig

logger = logging.getLogger(__name__)


def _get_client(config: AdminConfig) -> Minio:
    import urllib3
    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=3.0, read=10.0),
        retries=urllib3.Retry(total=0, connect=0, read=0),
    )
    return Minio(
        config.minio_endpoint,
        access_key=config.minio_access_key,
        secret_key=config.minio_secret_key,
        secure=config.minio_secure,
        http_client=http_client,
    )


def _list_buckets_sync(config: AdminConfig) -> list[str]:
    client = _get_client(config)
    return [b.name for b in client.list_buckets()]


def _browse_sync(
    config: AdminConfig,
    bucket: str,
    prefix: str,
    sort_by: str,
    order: str,
) -> dict[str, Any]:
    client = _get_client(config)
    raw = list(client.list_objects(bucket, prefix=prefix, recursive=False))

    folders: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    for obj in raw:
        stripped = obj.object_name[len(prefix):]
        if obj.is_dir:
            folders.append({
                "name": stripped.rstrip("/"),
                "path": obj.object_name,
                "type": "folder",
                "last_modified": None,
                "size": None,
            })
        else:
            modified = obj.last_modified.isoformat() if obj.last_modified else None
            files.append({
                "name": stripped,
                "path": obj.object_name,
                "type": "file",
                "size": obj.size or 0,
                "last_modified": modified,
                "etag": (obj.etag or "").strip('"'),
            })

    reverse = order == "desc"
    key_name = lambda x: x["name"].lower()
    key_date = lambda x: (x.get("last_modified") or "")

    sort_key = key_date if sort_by == "date" else key_name
    folders.sort(key=sort_key, reverse=reverse)
    files.sort(key=sort_key, reverse=reverse)

    return {
        "bucket": bucket,
        "prefix": prefix,
        "folders": folders,
        "files": files,
        "total": len(folders) + len(files),
    }


def _get_object_sync(config: AdminConfig, bucket: str, key: str):
    """Returns (data_bytes, content_type, size) for the given object."""
    client = _get_client(config)
    response = client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    stat = client.stat_object(bucket, key)
    content_type = stat.content_type or "application/octet-stream"
    return data, content_type, len(data)


async def get_object(config: AdminConfig, bucket: str, key: str):
    """Returns (data_bytes, content_type, size). Streams via asyncio.to_thread."""
    return await asyncio.wait_for(
        asyncio.to_thread(_get_object_sync, config, bucket, key), timeout=60.0
    )


async def list_buckets(config: AdminConfig) -> list[str]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_list_buckets_sync, config), timeout=8.0
        )
    except Exception as exc:
        logger.warning("list_buckets failed: %s", exc)
        return []


async def browse(
    config: AdminConfig,
    bucket: str,
    prefix: str = "",
    sort_by: str = "name",
    order: str = "asc",
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_browse_sync, config, bucket, prefix, sort_by, order),
            timeout=15.0,
        )
    except Exception as exc:
        logger.warning("browse %s/%s failed: %s", bucket, prefix, exc)
        return {"bucket": bucket, "prefix": prefix, "folders": [], "files": [], "total": 0, "error": str(exc)}


async def presign(config: AdminConfig, bucket: str, key: str) -> str:
    """Kept for reference but prefer proxy download for in-cluster deployments."""
    from datetime import timedelta
    return await asyncio.wait_for(
        asyncio.to_thread(
            lambda: _get_client(config).presigned_get_object(bucket, key, expires=timedelta(hours=1))
        ),
        timeout=8.0,
    )
