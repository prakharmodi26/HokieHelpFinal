"""Configuration loaded from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderConfig:
    """Immutable embedder configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_chunks_bucket: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model: str
    embedding_batch_size: int

    @classmethod
    def from_env(cls) -> EmbedderConfig:
        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required environment variable {name} is not set")
            return val

        return cls(
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            qdrant_host=_require("QDRANT_HOST"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
        )
