from __future__ import annotations
import logging, time
from typing import Any
import httpx
from minio import Minio
from qdrant_client import QdrantClient
from admin.config import AdminConfig

logger = logging.getLogger(__name__)

async def check_http_service(name: str, url: str) -> dict[str, Any]:
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return {"name": name, "healthy": resp.status_code < 500,
                    "status_code": resp.status_code, "latency_ms": round((time.monotonic() - start) * 1000)}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc),
                "latency_ms": round((time.monotonic() - start) * 1000)}

def get_minio_stats(config: AdminConfig) -> dict[str, Any]:
    try:
        client = Minio(config.minio_endpoint, access_key=config.minio_access_key,
                       secret_key=config.minio_secret_key, secure=config.minio_secure)
        buckets = client.list_buckets()
        bucket_stats = []
        for bucket in buckets:
            objects = list(client.list_objects(bucket.name, recursive=True))
            total_size = sum(o.size or 0 for o in objects)
            bucket_stats.append({"name": bucket.name, "objects": len(objects),
                                  "size_bytes": total_size, "size_mb": round(total_size / (1024*1024), 2)})
        return {"healthy": True, "buckets": bucket_stats}
    except Exception as exc:
        logger.warning("MinIO health check failed: %s", exc)
        return {"healthy": False, "error": str(exc), "buckets": []}

def get_qdrant_stats(config: AdminConfig) -> dict[str, Any]:
    try:
        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port, timeout=5)
        collections = client.get_collections().collections
        result = []
        for col in collections:
            info = client.get_collection(col.name)
            result.append({"name": col.name, "vectors_count": info.vectors_count or 0,
                           "indexed_vectors_count": info.indexed_vectors_count or 0,
                           "status": info.status.value if info.status else "unknown"})
        return {"healthy": True, "collections": result}
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {"healthy": False, "error": str(exc), "collections": []}

async def get_full_health(config: AdminConfig) -> dict[str, Any]:
    embedder = await check_http_service("embedder", f"{config.embedder_url}/health")
    chatbot = await check_http_service("chatbot", f"{config.chatbot_url}/health")
    ollama = await check_http_service("ollama", f"{config.ollama_url}/api/tags")
    return {
        "services": {"embedder": embedder, "chatbot": chatbot, "ollama": ollama},
        "storage": get_minio_stats(config),
        "vectors": get_qdrant_stats(config),
    }
