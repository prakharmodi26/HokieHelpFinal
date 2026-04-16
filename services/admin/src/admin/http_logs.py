from __future__ import annotations
import logging
from typing import Any
import httpx
from admin.config import AdminConfig
from admin import logbuffer

logger = logging.getLogger(__name__)


def service_endpoints(config: AdminConfig) -> dict[str, dict[str, str]]:
    return {
        "admin": {"url": "self", "kind": "ringbuffer"},
        "chatbot": {"url": f"{config.chatbot_url}/logs", "kind": "http"},
        "embedder": {"url": f"{config.embedder_url}/logs", "kind": "http"},
        "ollama": {"url": f"{config.ollama_url}", "kind": "unsupported"},
        "minio": {"url": config.minio_endpoint, "kind": "unsupported"},
        "qdrant": {"url": f"{config.qdrant_host}:{config.qdrant_port}", "kind": "unsupported"},
    }


async def list_services(config: AdminConfig) -> list[dict[str, Any]]:
    eps = service_endpoints(config)
    return [{"service": name, **info} for name, info in eps.items()]


async def fetch_logs(config: AdminConfig, service: str, lines: int = 200) -> dict[str, Any]:
    eps = service_endpoints(config)
    if service not in eps:
        return {"service": service, "error": "unknown service", "logs": ""}
    info = eps[service]
    if info["kind"] == "ringbuffer":
        return {"service": service, "logs": logbuffer.get_logs(lines)}
    if info["kind"] == "unsupported":
        return {"service": service, "error": f"log endpoint not available for {service} (use kubectl/rancher)", "logs": ""}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(info["url"], params={"lines": lines})
            if r.status_code != 200:
                return {"service": service, "error": f"HTTP {r.status_code}", "logs": r.text[:2000]}
            return {"service": service, "logs": r.text}
    except Exception as exc:
        logger.warning("fetch_logs %s failed: %s", service, exc)
        return {"service": service, "error": str(exc), "logs": ""}
