"""Configuration loaded from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkerConfig:
    """Immutable chunker configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_cleaned_bucket: str
    minio_chunks_bucket: str
    chunk_preferred_tokens: int
    chunk_overlap_tokens: int
    chunk_min_tokens: int

    @classmethod
    def from_env(cls) -> ChunkerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_SECURE, MINIO_CLEANED_BUCKET, MINIO_CHUNKS_BUCKET,
                  CHUNK_PREFERRED_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS
        """
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
            minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            chunk_preferred_tokens=int(os.environ.get("CHUNK_PREFERRED_TOKENS", "400")),
            chunk_overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "64")),
            chunk_min_tokens=int(os.environ.get("CHUNK_MIN_TOKENS", "120")),
        )
