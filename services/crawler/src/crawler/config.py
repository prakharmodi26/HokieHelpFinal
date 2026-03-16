"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CrawlerConfig:
    """Immutable crawler configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool
    seed_url: str
    max_depth: int
    max_pages: int
    allowed_domain: str

    @classmethod
    def from_env(cls) -> CrawlerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_BUCKET, MINIO_SECURE, CRAWL_SEED_URL,
                  CRAWL_MAX_DEPTH, CRAWL_MAX_PAGES
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
            minio_bucket=os.environ.get("MINIO_BUCKET", "crawled-pages"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
            max_depth=int(os.environ.get("CRAWL_MAX_DEPTH", "2")),
            max_pages=int(os.environ.get("CRAWL_MAX_PAGES", "500")),
            allowed_domain=os.environ.get("CRAWL_ALLOWED_DOMAIN", "cs.vt.edu"),
        )
