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
    minio_cleaned_bucket: str
    minio_secure: bool
    seed_url: str
    max_depth: int
    max_pages: int
    allowed_domains: tuple[str, ...]
    blocked_domains: tuple[str, ...]
    blocked_paths: tuple[str, ...]
    prune_threshold: float
    request_delay: float  # seconds between requests to avoid rate limiting

    @classmethod
    def from_env(cls) -> CrawlerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_BUCKET, MINIO_SECURE, CRAWL_SEED_URL,
                  CRAWL_MAX_DEPTH, CRAWL_MAX_PAGES, CRAWL_ALLOWED_DOMAINS,
                  CRAWL_BLOCKED_DOMAINS, CRAWL_PRUNE_THRESHOLD
        """

        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required environment variable {name} is not set")
            return val

        def _domains(name: str, default: str) -> tuple[str, ...]:
            raw = os.environ.get(name, default)
            return tuple(d.strip() for d in raw.split(",") if d.strip())

        return cls(
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_bucket=os.environ.get("MINIO_BUCKET", "crawled-pages"),
            minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
            max_depth=int(os.environ.get("CRAWL_MAX_DEPTH", "4")),
            max_pages=int(os.environ.get("CRAWL_MAX_PAGES", "9999999")),
            allowed_domains=_domains(
                "CRAWL_ALLOWED_DOMAINS",
                "cs.vt.edu",
            ),
            blocked_domains=_domains(
                "CRAWL_BLOCKED_DOMAINS",
                "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,"
                "portal.cs.vt.edu,api.cs.vt.edu,forum.cs.vt.edu,login.cs.vt.edu",
            ),
            blocked_paths=_domains(
                "CRAWL_BLOCKED_PATHS",
                "/content/,/editor.html,/cs-root.html,/cs-source.html",
            ),
            prune_threshold=float(os.environ.get("CRAWL_PRUNE_THRESHOLD", "0.45")),
            request_delay=float(os.environ.get("CRAWL_REQUEST_DELAY", "0.5")),
        )
