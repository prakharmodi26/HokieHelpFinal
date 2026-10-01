"""Crawler entrypoint — registry seed → frontier → render → mirror → documents → summary."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from crawler.affiliated import load_affiliated_matcher
from crawler.config import CrawlerConfig
from crawler.discovery.hostfilter import HostFilter
from crawler.documents import download_and_process_documents
from crawler.engine import run_engine
from crawler.frontier import Frontier
from crawler.registry import host_root_urls, load_host_caps, load_modes, load_seed_hosts
from crawler.runlock import RunnerLockHeld, single_runner_lock
from crawler.scope import Scope, classify_url
from crawler.seed_docs import build_department_info_doc
from crawler.storage import MinioStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Graceful shutdown ──────────────────────────────────────────────────────
# Set by the SIGTERM/SIGINT handler (installed in cli()). The render engine
# and documents pass poll is_set() between batches/URLs and drain cleanly
# rather than dying mid-operation. threading.Event.set() is async-signal-safe
# (it only flips a flag), so it's safe to call from a signal handler.
shutdown_event = threading.Event()


def _request_shutdown(signum, frame) -> None:
    """Signal handler: request a graceful drain on the next batch boundary."""
    if not shutdown_event.is_set():
        logger.warning(
            "Received signal %s — requesting graceful shutdown "
            "(will finish the in-flight batch, then stop)", signum,
        )
        shutdown_event.set()


async def populate_frontier(frontier: Frontier, config: CrawlerConfig) -> dict:
    """Seed the frontier from the committed host registry (no enumeration).

    Resumable: if the frontier already has pending or done rows, seeding is
    skipped and the existing frontier is reused.

    CONTRACT (load-bearing): the returned dict ALWAYS carries a real HostFilter
    under the "host_filter" key — built BEFORE the resume short-circuit — and
    cli() passes it to run_engine so the blocklist filters BFS expansion. If a
    future edit drops that key on any production path, the blocklist silently
    stops applying during the crawl.
    """
    # config.blocked_paths (CRAWL_BLOCKED_PATHS) are treated as regex fragments
    # matched against URL path+query, merged with any path_patterns in the YAML.
    host_filter = HostFilter.from_yaml(
        config.host_blocklist_path,
        extra_path_patterns=list(config.blocked_paths),
    )
    affiliated_matcher = load_affiliated_matcher(config.vt_affiliated_hosts_path)
    logger.info(
        "Loaded VT-affiliated hosts allowlist from %s", config.vt_affiliated_hosts_path
    )

    import os
    counts = frontier.count_by_status()
    total_rows = sum(counts.values())
    # Fresh start if DB file missing or no rows in frontier
    if not os.path.exists(config.frontier_db_path) or total_rows == 0:
        logger.info("Fresh frontier, seeding from registry.")
        # proceed to seeding below
    elif counts.get("pending", 0) > 0:
        logger.info("Resuming interrupted crawl: %s pending URLs to process.", counts.get("pending", 0))
        return {
            "urls_enqueued": 0,
            "resumed": True,
            "host_filter": host_filter,
            "affiliated_matcher": affiliated_matcher,
        }
    else:
        logger.info("Frontier has %d rows but 0 pending — crawl already completed. Exiting.", total_rows)
        return {
            "completed_already": True,
            "host_filter": host_filter,
            "affiliated_matcher": affiliated_matcher,
        }

    hosts = load_seed_hosts(config.host_registry_path)
    if config.max_hosts is not None:
        hosts = hosts[: config.max_hosts]

    seeded = 0
    for url in host_root_urls(hosts):
        scope, hops = classify_url(
            url,
            parent_hops=[],
            host_filter=host_filter,
            affiliated_matcher=affiliated_matcher,
        )
        if scope is Scope.OUT_OF_SCOPE:
            continue
        if frontier.enqueue(
            url,
            scope=scope.value,
            depth=0,
            hop_history=hops,
            discovery_source="registry",
        ):
            seeded += 1
    logger.info(
        "Seeded %d host roots from registry %s", seeded, config.host_registry_path
    )
    return {
        "urls_enqueued": seeded,
        "host_filter": host_filter,
        "affiliated_matcher": affiliated_matcher,
    }


def _render_summary(
    counts: dict,
    stats: dict,
    doc_stats: dict,
    timestamp: str,
    status: str = "completed",
    error: str | None = None,
) -> str:
    """Build the run-summary markdown uploaded to the logs bucket."""
    lines = [
        f"# Crawl Summary — {timestamp}",
        "",
        f"**Status:** {status}",
    ]
    if error:
        lines.append(f"**Error:** {error}")
    lines += ["", "## Frontier"]
    for status_key in sorted(counts):
        lines.append(f"- {status_key}: {counts[status_key]}")
    lines += ["", "## Render"]
    for k in sorted(stats):
        if k == "document_urls":
            continue
        lines.append(f"- {k}: {stats[k]}")
    lines += [
        "",
        "## Documents",
        f"- documents_processed: {doc_stats.get('documents_processed', 0)}",
        f"- documents_failed: {doc_stats.get('documents_failed', 0)}",
    ]
    lines.append("")
    return "\n".join(lines)


def _build_summary_json(
    counts: dict,
    stats: dict,
    doc_stats: dict,
    by_host: dict,
    config: CrawlerConfig,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    status: str = "completed",
    error: str | None = None,
) -> str:
    """Build the machine-readable JSON summary uploaded to the logs bucket."""
    payload = {
        "run_id": config.run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "frontier": counts,
        "render": {k: v for k, v in stats.items() if k != "document_urls"},
        "documents": {
            "documents_processed": doc_stats.get("documents_processed", 0),
            "documents_failed": doc_stats.get("documents_failed", 0),
        },
        "by_host": by_host,
    }
    if error:
        payload["error"] = error
    return json.dumps(payload, indent=2, default=str)


def cli() -> None:
    """CLI entrypoint: seed frontier from registry, render, mirror, documents."""
    logger.info("Starting HokieHelp crawler (registry-seeded frontier)")

    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    config.configure_logging()
    logger.info("Config: %s", repr(config))
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    t0 = time.monotonic()

    # Graceful-shutdown handlers. SIGTERM (K8s activeDeadlineSeconds) drains
    # cleanly: asyncio.run does not touch SIGTERM, so this handler stays live
    # across every phase and run_engine stops at the next batch boundary.
    # SIGINT is best-effort here: during an asyncio.run phase the asyncio Runner
    # owns SIGINT and raises KeyboardInterrupt instead of setting shutdown_event,
    # so a dev Ctrl-C cancels rather than drains. Production uses SIGTERM, so
    # this is a dev-ergonomics gap only. (catalog_main, a single asyncio.run,
    # fixes SIGINT via loop.add_signal_handler — see its _install_shutdown_handlers.
    # main's multi-phase asyncio.run structure can't reuse that without leaving
    # SIGTERM at SIG_DFL between phases, so it keeps the signal.signal handler.)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_shutdown)

    # Single-runner guard: the frontier DB (an ephemeral emptyDir SQLite file) is
    # single-writer and the engine assumes exactly one crawler Job (lease_pending
    # + requeue_leased are not concurrency-safe). Refuse to start if another
    # process holds the lock
    # rather than silently double-leasing the shared frontier. Manual enter/exit
    # keeps the long run body un-indented; the lock is released in the finally.
    run_lock = single_runner_lock(f"{config.frontier_db_path}.lock")
    try:
        run_lock.__enter__()
    except RunnerLockHeld as exc:
        logger.error("Refusing to start — %s", exc)
        sys.exit(1)

    # Partial-run state: initialised before the try so the finally can always
    # build a summary even if setup itself fails. These are overwritten with
    # real values as each phase completes.
    counts: dict = {}
    stats: dict = {"pages_stored": 0, "pages_failed": 0, "needs_js": 0, "documents": 0}
    doc_stats: dict = {"documents_processed": 0, "documents_failed": 0}
    by_host: dict = {}
    run_status = "starting"
    failure_msg: str | None = None
    frontier: Frontier | None = None
    storage: MinioStorage | None = None

    try:
        run_status = "setup"
        storage = MinioStorage(config)
        for bucket in (
            config.minio_data_bucket,
            config.minio_document_bucket,
            config.minio_logs_bucket,
        ):
            storage.ensure_bucket(bucket)

        frontier = Frontier(config.frontier_db_path)

        run_status = "seeding"
        seed_info = asyncio.run(populate_frontier(frontier, config))
        logger.info(
            "Frontier seeding: %s",
            {k: v for k, v in seed_info.items() if k != "host_filter"},
        )

        # CONTRACT: populate_frontier always returns a real HostFilter under
        # "host_filter" in production (see its docstring). None happens only
        # when populate_frontier is mocked out in tests.
        host_filter = seed_info.get("host_filter")
        if host_filter is None:
            logger.warning(
                "BFS blocklist filtering disabled (host_filter=None) — "
                "expected only under test mocks"
            )
        affiliated_matcher = seed_info.get("affiliated_matcher")

        modes = load_modes(config.host_registry_path)
        host_caps = load_host_caps(config.host_registry_path)
        # Backstop: registry "-1" (unlimited) hosts get a high ceiling so an
        # uncaught trap can't replay the wuvt firehose. Real departments are far
        # below this, so legitimate content is never truncated. Set
        # CRAWL_UNLIMITED_BACKSTOP=-1 to restore truly-unlimited behavior.
        if config.unlimited_backstop != -1:
            n_unlimited = sum(1 for c in host_caps.values() if c == -1)
            host_caps = {h: (config.unlimited_backstop if c == -1 else c)
                         for h, c in host_caps.items()}
            logger.info("Applied unlimited backstop %d to %d '-1' hosts",
                        config.unlimited_backstop, n_unlimited)
        if host_caps:
            logger.info("Per-host URL caps loaded for %d hosts", len(host_caps))

        if seed_info.get("completed_already"):
            # Frontier has rows but 0 pending — a prior run in this same pod
            # already drained the frontier. Skip render + documents entirely
            # (no browser startup, no wasted work) and go straight to summary.
            logger.info(
                "Frontier has no pending URLs — crawl already completed, skipping render."
            )
            run_status = "completed"
        else:
            run_status = "render"
            stats = asyncio.run(
                run_engine(
                    frontier,
                    storage,
                    main_bucket=config.minio_data_bucket,
                    max_depth=config.max_depth,
                    batch_size=config.batch_size,
                    memory_pct=config.memory_threshold_percent,
                    max_sessions=config.max_sessions,
                    threshold=config.prune_content_threshold,
                    host_filter=host_filter,
                    affiliated_matcher=affiliated_matcher,
                    max_urls_per_host=config.max_urls_per_host,
                    host_caps=host_caps or None,
                    modes=modes,
                    batch_idle_timeout=config.batch_idle_timeout,
                    max_wedge_attempts=config.max_wedge_attempts,
                    run_id=config.run_id,
                    progress_log_interval=config.progress_log_interval,
                    rate_base_delay=(
                        config.crawl_rate_base_delay_min,
                        config.crawl_rate_base_delay_max,
                    ),
                    rate_max_delay=config.crawl_rate_max_delay,
                    rate_max_retries=config.crawl_rate_max_retries,
                    shutdown_event=shutdown_event,
                    max_html_size_bytes=config.max_html_size_bytes,
                )
            )
            logger.info(
                "Render complete: %s",
                {k: v for k, v in stats.items() if k != "document_urls"},
            )

            # Seed doc so RAG has campus addresses, phones, social links.
            # Non-critical: a failure here must not kill the run.
            try:
                storage.upload_document(
                    "_department-info.md",
                    build_department_info_doc(),
                    bucket=config.minio_data_bucket,
                )
            except Exception as exc:
                logger.warning("Failed to upload _department-info.md (non-fatal): %s", exc)

            # Documents pass: PDFs/DOCX go to the data bucket (markdown) and raw
            # binary is preserved in the document bucket before conversion.
            doc_urls = stats.get("document_urls", set())
            if doc_urls:
                run_status = "documents"
                doc_stats = asyncio.run(
                    download_and_process_documents(
                        doc_urls,
                        storage,
                        request_delay=config.request_delay,
                        target_bucket=config.minio_data_bucket,
                        document_bucket=config.minio_document_bucket,
                        max_document_size_bytes=config.max_document_size_bytes,
                        max_pdf_pages=config.max_pdf_pages,
                        max_extracted_text_chars=config.max_extracted_text_chars,
                        affiliated_matcher=affiliated_matcher,
                        shutdown_event=shutdown_event,
                    )
                )
                logger.info(
                    "Documents: %d processed, %d failed",
                    doc_stats["documents_processed"],
                    doc_stats["documents_failed"],
                )

            # Determine final status: a drained run is a clean exit, not a failure.
            if shutdown_event.is_set():
                run_status = "interrupted"
            else:
                run_status = "completed"

        logger.info(
            "Crawl %s — %d pages stored, %d failed, %d skipped, %.1fs — "
            "data=%s documents=%s",
            run_status,
            counts.get("done", 0) if counts else stats.get("pages_stored", 0),
            counts.get("failed", 0) if counts else stats.get("pages_failed", 0),
            counts.get("skipped", 0) if counts else 0,
            time.monotonic() - t0, config.minio_data_bucket,
            config.minio_document_bucket,
        )
    except Exception as exc:
        # Any unhandled exception (storage failure, render crash, etc.) is
        # captured here so the finally can still write a partial summary and
        # release resources — no more raw tracebacks killing the K8s Job.
        failure_msg = f"{type(exc).__name__}: {exc}"
        run_status = "failed"
        logger.error("Crawl failed during '%s': %s", run_status, failure_msg, exc_info=True)
    finally:
        # Always gather whatever frontier counts are available and write a
        # best-effort summary. If MinIO itself is down, the summary upload will
        # also fail — that's logged but never masks the original error.
        if frontier is not None:
            try:
                counts = frontier.count_by_status()
                by_host = frontier.count_by_host_status()
            except Exception as exc:
                logger.warning("Could not read final frontier counts: %s", exc)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        ended_at = datetime.now(timezone.utc).isoformat()
        duration = time.monotonic() - t0

        if storage is not None:
            try:
                summary_md = _render_summary(
                    counts, stats, doc_stats, timestamp,
                    status=run_status, error=failure_msg,
                )
                summary_key = f"crawl-summary-{timestamp}.md"
                storage.upload_document(
                    summary_key, summary_md, bucket=config.minio_logs_bucket,
                )

                summary_json = _build_summary_json(
                    counts, stats, doc_stats, by_host,
                    config, started_at, ended_at, duration,
                    status=run_status, error=failure_msg,
                )
                json_key = f"crawl-summary-{timestamp}.json"
                storage.upload_document(
                    json_key, summary_json, bucket=config.minio_logs_bucket,
                )
                logger.info(
                    "Run summary written to %s/%s and %s/%s (status=%s)",
                    config.minio_logs_bucket, summary_key,
                    config.minio_logs_bucket, json_key, run_status,
                )
            except Exception as exc:
                logger.warning("Could not upload run summary (non-fatal): %s", exc)

        if frontier is not None:
            try:
                frontier.close()
            except Exception as exc:
                logger.warning("Could not close frontier DB (non-fatal): %s", exc)
        run_lock.__exit__(None, None, None)

    # Exit code: 0 for clean completion OR graceful drain; 1 for failure.
    # A drained run (SIGTERM at the 48h deadline) is a successful clean exit.
    if run_status == "failed":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    cli()
