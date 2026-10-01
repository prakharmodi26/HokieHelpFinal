"""Crawler configuration — the single source of truth.

All crawler configuration lives in this file. Every field has a default value
suitable for production Kubernetes deployment, except for the two MinIO
credential fields (``minio_access_key``, ``minio_secret_key``) which MUST be
provided via environment variables — without them the crawler will not start.

Any field can be overridden via an environment variable. This is how the Admin
service spawns crawler jobs with different settings (different buckets, crawl
depth, registry paths, blocklists, etc.) without rebuilding the image.

If a non-secret env var is malformed (e.g. ``CRAWL_MAX_DEPTH=abc``), the code
falls back to the default rather than crashing. If a required secret is missing,
``from_env()`` raises ``ValueError`` and the entrypoint exits with a clean
error message and non-zero status code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read an int env var; fall back to default on missing/malformed value."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r, using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var; fall back to default on missing/malformed value."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default %f", name, raw, default)
        return default


def _env_str(name: str, default: str) -> str:
    """Read a string env var; fall back to default if unset or empty."""
    return os.environ.get(name) or default


def _env_str_list(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated env var as a tuple of stripped strings."""
    raw = os.environ.get(name, default)
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _env_int_or_none(name: str) -> int | None:
    """Read an optional int env var; returns None if unset/malformed."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r, treating as unset (None)", name, raw)
        return None


@dataclass(frozen=True)
class CrawlerConfig:
    """Immutable crawler configuration.

    This is the single source of truth. Edit defaults here → they change
    everywhere (dev, prod, tests). Override any field via env vars — this is
    how Admin spawns jobs with custom settings.

    Required env vars (no defaults, crawler won't start without them):
        MINIO_ACCESS_KEY, MINIO_SECRET_KEY

    All other fields have sensible production defaults defined below.
    """

    # ── MinIO connection ───────────────────────────────────────────────────
    # Secrets — MUST be provided via env vars. No hardcoded defaults.
    minio_access_key: str
    minio_secret_key: str
    # Endpoint default is the in-cluster K8s service name. Dev overrides to
    # localhost:9000 via env var.
    minio_endpoint: str = "minio:9000"
    minio_data_bucket: str = "data"
    minio_document_bucket: str = "document"
    minio_logs_bucket: str = "logs"
    minio_secure: bool = False

    # ── Crawl engine ───────────────────────────────────────────────────────
    # 10 = production depth. Dev should set CRAWL_MAX_DEPTH=2 for fast local runs.
    max_depth: int = 10
    # Seconds to sleep between DOCUMENT (PDF/DOCX) downloads in the httpx
    # documents pass. The render path does NOT use this — it has crawl4ai's
    # adaptive RateLimiter (crawl_rate_* fields) instead.
    request_delay: float = 0.5
    # Regex fragments matched against URL path+query, merged with YAML patterns.
    # /content/(?!dam/) blocks AEM duplicate HTML but keeps /content/dam/ (PDFs).
    blocked_paths: tuple[str, ...] = (
        "/content/(?!dam/)", "/editor.html", "/cs-root.html", "/cs-source.html",
    )
    # Frontier SQLite database — on an emptyDir (ephemeral, dies with the pod)
    # in K8s, a local path in dev. Every run starts with a fresh frontier.
    frontier_db_path: str = "/data/frontier.db"

    # ── Registry / blocklist / affiliation YAML paths ──────────────────────
    # Admin can override these to inject custom seed hosts, blocklists, or
    # affiliated-host allowlists per job.
    host_registry_path: str = "config/host-registry.yaml"
    host_blocklist_path: str = "config/host-blocklist.yaml"
    vt_affiliated_hosts_path: str = "config/vt-affiliated-hosts.yaml"

    # ── Discovery + per-host caps ──────────────────────────────────────────
    discovery_source: str = "sitemap+cc"
    # Cap number of registry hosts (None = all).
    max_hosts: int | None = None
    # Backstop for BFS-discovered hosts NOT in the registry. Registry hosts
    # tiered -1 (unlimited) are bounded by unlimited_backstop.
    max_urls_per_host: int = 50
    # Runaway guard for unlimited-tier registry hosts. High enough not to
    # truncate real departments, low enough to stop a faceted-search trap.
    unlimited_backstop: int = 1500

    # ── Render engine ──────────────────────────────────────────────────────
    batch_size: int = 200
    memory_threshold_percent: float = 70.0
    max_sessions: int = 10
    prune_content_threshold: float = 0.5
    # Watchdog: if no result for this many seconds, the batch is wedged.
    # Browser is recycled and unreturned URLs are requeued as pending.
    batch_idle_timeout: float = 180.0
    # A wedge idles the whole batch's stream; its un-returned URLs are requeued
    # (not blanket-failed) and only marked failed after wedging this many times,
    # so one poison page can't drop its innocent batch siblings (BUG-A1).
    max_wedge_attempts: int = 3

    # Render-path politeness (crawl4ai RateLimiter on the MemoryAdaptiveDispatcher).
    # Adaptive: a random delay in [min, max] before each request, exponential
    # backoff up to max_delay on 429/503, bounded by max_retries. This governs
    # the BROWSER render path. The documents (httpx) path uses request_delay
    # instead — see that field. Don't confuse the two knobs.
    crawl_rate_base_delay_min: float = 1.0
    crawl_rate_base_delay_max: float = 3.0
    crawl_rate_max_delay: float = 60.0
    crawl_rate_max_retries: int = 3

    # ── Document pass (PDF/DOCX extraction limits) ─────────────────────────
    max_document_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_pdf_pages: int = 500
    max_extracted_text_chars: int = 2_000_000  # 2M characters

    # ── Content validation ─────────────────────────────────────────────────
    # HTML pages larger than this are rejected before storage (prevents
    # runaway tables/faceted-search pages from consuming MinIO + chunker).
    max_html_size_bytes: int = 10 * 1024 * 1024  # 10 MB

    # ── Catalog ────────────────────────────────────────────────────────────
    catalog_edition: str = "2026-2027"

    # Catalog render tunables (CourseLeaf browser-render pass). Defaults mirror
    # render.DEFAULT_RENDER_CHUNK_SIZE / DEFAULT_RENDER_IDLE_TIMEOUT. A retry
    # pass re-renders failed/watchdog-deferred subject/program pages so a whole
    # course set is not lost to one transient render hiccup.
    catalog_render_chunk_size: int = 12
    catalog_render_idle_timeout: float = 120.0
    catalog_render_max_retries: int = 1

    # Pod-local single-runner lock for the catalog job (parity with the main
    # crawler's frontier lock). Guards two catalog processes from racing on
    # bucket keys within a pod; cross-pod concurrency would need a distributed
    # lease. /tmp is always writable (the catalog Job mounts no data volume).
    catalog_lock_path: str = "/tmp/hokiehelp-catalog.lock"

    # ── Observability ──────────────────────────────────────────────────────
    # run_id identifies this job in logs and summary files. Admin sets this
    # per K8s Job so logs from concurrent jobs are distinguishable. If unset,
    # a timestamp-based ID is generated.
    run_id: str = ""
    # Logging level: INFO (default), DEBUG (per-URL detail), WARNING, etc.
    log_level: str = "INFO"
    # Emit a frontier progress log every N batches (0 = disabled).
    progress_log_interval: int = 10

    _SENSITIVE_FIELDS = frozenset({"minio_access_key", "minio_secret_key"})

    def __repr__(self) -> str:
        """Mask secrets in repr to prevent credential leakage in logs."""
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in self._SENSITIVE_FIELDS:
                val = (val[:3] + "***") if val else "***"
            parts.append(f"{f.name}={val!r}")
        return f"CrawlerConfig({', '.join(parts)})"

    def configure_logging(self) -> None:
        """Set log level from config and inject run_id into all log records.

        Call this right after from_env() succeeds. Every subsequent log line
        will carry [run_id] so admin can correlate logs to K8s Jobs.
        """
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(level)

        run_id = self.run_id

        class _RunIdFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.run_id = run_id
                return True

        filt = _RunIdFilter()
        for handler in logging.getLogger().handlers:
            handler.addFilter(filt)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s [%(run_id)s]: %(message)s"
                )
            )

    @classmethod
    def from_env(cls) -> CrawlerConfig:
        """Load configuration from environment variables.

        Raises ValueError if required secrets are missing. The entrypoint
        catches this, logs a clean error, and exits with status 1.

        See field docstrings above for what each env var controls.
        """
        access_key = os.environ.get("MINIO_ACCESS_KEY")
        secret_key = os.environ.get("MINIO_SECRET_KEY")
        if not access_key:
            raise ValueError(
                "Required environment variable MINIO_ACCESS_KEY is not set — "
                "crawler cannot start without MinIO credentials."
            )
        if not secret_key:
            raise ValueError(
                "Required environment variable MINIO_SECRET_KEY is not set — "
                "crawler cannot start without MinIO credentials."
            )

        return cls(
            minio_access_key=access_key,
            minio_secret_key=secret_key,
            minio_endpoint=_env_str("MINIO_ENDPOINT", "minio:9000"),
            minio_data_bucket=_env_str("MINIO_DATA_BUCKET", "data"),
            minio_document_bucket=_env_str("MINIO_DOCUMENT_BUCKET", "document"),
            minio_logs_bucket=_env_str("MINIO_LOGS_BUCKET", "logs"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",

            max_depth=_env_int("CRAWL_MAX_DEPTH", 10),
            request_delay=_env_float("CRAWL_REQUEST_DELAY", 0.5),
            blocked_paths=_env_str_list(
                "CRAWL_BLOCKED_PATHS",
                "/content/(?!dam/),/editor.html,/cs-root.html,/cs-source.html",
            ),
            frontier_db_path=_env_str("FRONTIER_DB_PATH", "/data/frontier.db"),

            host_registry_path=_env_str("HOST_REGISTRY_PATH", "config/host-registry.yaml"),
            host_blocklist_path=_env_str("HOST_BLOCKLIST_PATH", "config/host-blocklist.yaml"),
            vt_affiliated_hosts_path=_env_str(
                "VT_AFFILIATED_HOSTS_PATH", "config/vt-affiliated-hosts.yaml"
            ),

            discovery_source=_env_str("CRAWL_DISCOVERY_SOURCE", "sitemap+cc"),
            max_hosts=_env_int_or_none("CRAWL_MAX_HOSTS"),
            max_urls_per_host=_env_int("CRAWL_MAX_URLS_PER_HOST", 50),
            unlimited_backstop=_env_int("CRAWL_UNLIMITED_BACKSTOP", 1500),

            batch_size=_env_int("CRAWL_BATCH_SIZE", 200),
            memory_threshold_percent=_env_float("CRAWL_MEMORY_PCT", 70.0),
            max_sessions=_env_int("CRAWL_MAX_SESSIONS", 10),
            prune_content_threshold=_env_float("CRAWL_PRUNE_CONTENT_THRESHOLD", 0.5),
            batch_idle_timeout=_env_float("CRAWL_BATCH_IDLE_TIMEOUT", 180.0),
            max_wedge_attempts=_env_int("CRAWL_MAX_WEDGE_ATTEMPTS", 3),
            crawl_rate_base_delay_min=_env_float("CRAWL_RATE_BASE_DELAY_MIN", 1.0),
            crawl_rate_base_delay_max=_env_float("CRAWL_RATE_BASE_DELAY_MAX", 3.0),
            crawl_rate_max_delay=_env_float("CRAWL_RATE_MAX_DELAY", 60.0),
            crawl_rate_max_retries=_env_int("CRAWL_RATE_MAX_RETRIES", 3),

            max_document_size_bytes=_env_int("CRAWL_MAX_DOCUMENT_SIZE_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_env_int("CRAWL_MAX_PDF_PAGES", 500),
            max_extracted_text_chars=_env_int("CRAWL_MAX_EXTRACTED_TEXT_CHARS", 2_000_000),
            max_html_size_bytes=_env_int("CRAWL_MAX_HTML_SIZE_BYTES", 10 * 1024 * 1024),

            catalog_edition=_env_str("CATALOG_EDITION", "2026-2027"),
            catalog_render_chunk_size=_env_int("CATALOG_RENDER_CHUNK_SIZE", 12),
            catalog_render_idle_timeout=_env_float("CATALOG_RENDER_IDLE_TIMEOUT", 120.0),
            catalog_render_max_retries=_env_int("CATALOG_RENDER_MAX_RETRIES", 1),
            catalog_lock_path=_env_str("CATALOG_LOCK_PATH", "/tmp/hokiehelp-catalog.lock"),

            run_id=_env_str(
                "CRAWL_RUN_ID",
                datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S"),
            ),
            log_level=_env_str("LOG_LEVEL", "INFO"),
            progress_log_interval=_env_int("CRAWL_PROGRESS_LOG_INTERVAL", 10),
        )
