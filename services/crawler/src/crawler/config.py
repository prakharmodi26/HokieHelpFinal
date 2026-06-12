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
    minio_external_bucket: str = "crawled-pages-external"
    external_hops: int = 1
    frontier_db_path: str = "/data/frontier.db"
    host_registry_path: str = "config/host-registry.yaml"
    host_blocklist_path: str = "config/host-blocklist.yaml"
    discovery_source: str = "sitemap+cc"
    max_hosts: int | None = None
    max_urls_per_host: int = -1
    batch_size: int = 200
    memory_threshold_percent: float = 70.0
    max_sessions: int = 10
    prune_content_threshold: float = 0.5
    # Watchdog: if an arun_many result stream produces no new result for this
    # many seconds, the batch is treated as wedged (a single page whose render
    # never returns — downloads, SPA login loops). The browser is recycled and
    # the unreturned URLs are failed so they are not re-leased and re-wedged.
    batch_idle_timeout: float = 180.0

    @classmethod
    def from_env(cls) -> CrawlerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_BUCKET, MINIO_SECURE, CRAWL_SEED_URL,
                  CRAWL_MAX_DEPTH, CRAWL_MAX_PAGES, CRAWL_ALLOWED_DOMAINS,
                  CRAWL_BLOCKED_DOMAINS, CRAWL_PRUNE_THRESHOLD,
                  MINIO_EXTERNAL_BUCKET, CRAWL_EXTERNAL_HOPS,
                  FRONTIER_DB_PATH, HOST_REGISTRY_PATH, HOST_BLOCKLIST_PATH,
                  CRAWL_DISCOVERY_SOURCE, CRAWL_MAX_HOSTS,
                  CRAWL_MAX_URLS_PER_HOST, CRAWL_BATCH_SIZE,
                  CRAWL_MEMORY_PCT, CRAWL_MAX_SESSIONS,
                  CRAWL_PRUNE_CONTENT_THRESHOLD
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
                "vt.edu",
            ),
            blocked_domains=_domains(
                "CRAWL_BLOCKED_DOMAINS",
                "",
            ),
            blocked_paths=_domains(
                "CRAWL_BLOCKED_PATHS",
                "/content/,/editor.html,/cs-root.html,/cs-source.html",
            ),
            prune_threshold=float(os.environ.get("CRAWL_PRUNE_THRESHOLD", "0.45")),
            request_delay=float(os.environ.get("CRAWL_REQUEST_DELAY", "0.1")),
            minio_external_bucket=os.environ.get("MINIO_EXTERNAL_BUCKET", "crawled-pages-external"),
            external_hops=int(os.environ.get("CRAWL_EXTERNAL_HOPS", "1")),
            frontier_db_path=os.environ.get("FRONTIER_DB_PATH", "/data/frontier.db"),
            host_registry_path=os.environ.get("HOST_REGISTRY_PATH", "config/host-registry.yaml"),
            host_blocklist_path=os.environ.get("HOST_BLOCKLIST_PATH", "config/host-blocklist.yaml"),
            discovery_source=os.environ.get("CRAWL_DISCOVERY_SOURCE", "sitemap+cc"),
            max_hosts=(int(os.environ["CRAWL_MAX_HOSTS"]) if os.environ.get("CRAWL_MAX_HOSTS") else None),
            max_urls_per_host=int(os.environ.get("CRAWL_MAX_URLS_PER_HOST", "-1")),
            batch_size=int(os.environ.get("CRAWL_BATCH_SIZE", "200")),
            memory_threshold_percent=float(os.environ.get("CRAWL_MEMORY_PCT", "70.0")),
            max_sessions=int(os.environ.get("CRAWL_MAX_SESSIONS", "10")),
            prune_content_threshold=float(os.environ.get("CRAWL_PRUNE_CONTENT_THRESHOLD", "0.5")),
            batch_idle_timeout=float(os.environ.get("CRAWL_BATCH_IDLE_TIMEOUT", "180.0")),
        )
