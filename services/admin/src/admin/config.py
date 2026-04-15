from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AdminConfig:
    port: int
    data_dir: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket: str
    minio_cleaned_bucket: str
    minio_chunks_bucket: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedder_url: str
    chatbot_url: str
    ollama_url: str
    default_seed_url: str
    default_max_depth: int
    default_max_pages: int
    default_schedule: str

    @classmethod
    def from_env(cls) -> AdminConfig:
        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required env var {name} is not set")
            return val

        return cls(
            port=int(os.environ.get("ADMIN_PORT", "8080")),
            data_dir=os.environ.get("ADMIN_DATA_DIR", "/data"),
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            minio_bucket=os.environ.get("MINIO_BUCKET", "crawled-pages"),
            minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            qdrant_host=_require("QDRANT_HOST"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedder_url=_require("EMBEDDER_URL"),
            chatbot_url=os.environ.get("CHATBOT_URL", "http://hokiehelp-chatbot:8000"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
            default_seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
            default_max_depth=int(os.environ.get("CRAWL_MAX_DEPTH", "4")),
            default_max_pages=int(os.environ.get("CRAWL_MAX_PAGES", "9999999")),
            default_schedule=os.environ.get("CRAWL_SCHEDULE", "0 3 * * 0"),
        )
