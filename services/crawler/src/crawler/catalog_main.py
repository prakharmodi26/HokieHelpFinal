"""Standalone CourseLeaf catalog ingestion entrypoint.

Run as:
    python -m crawler.catalog_main

Loads config from environment (MINIO_DATA_BUCKET, CATALOG_EDITION plus the
standard MINIO_* credentials), renders the entire catalog.vt.edu course and
program pages via crawl4ai, and writes edition-tagged markdown cards into the
MAIN corpus bucket (MINIO_DATA_BUCKET — the one the chunker reads) so the
course catalog is ingested alongside every other page. OAI/Dataverse repository
records go to the same bucket; there is no separate course-catalog bucket. The
main web crawl is NOT involved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from crawler.catalog_card import harvest_catalog_records
from crawler.config import CrawlerConfig
from crawler.courseleaf import harvest_courseleaf_catalog
from crawler.runlock import RunnerLockHeld, single_runner_lock
from crawler.storage import MinioStorage, StorageFullError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Graceful shutdown ──────────────────────────────────────────────────────
# Set by the SIGTERM/SIGINT handler (installed in cli()). Checked between the
# OAI and CourseLeaf harvest phases so a catalog job can drain at a phase
# boundary instead of dying mid-harvest.
shutdown_event = threading.Event()


def _request_shutdown(signum, frame) -> None:
    """Signal handler: request a graceful drain at the next phase boundary."""
    if not shutdown_event.is_set():
        logger.warning(
            "Received signal %s — requesting graceful shutdown "
            "(will finish the current phase, then stop)", signum,
        )
        shutdown_event.set()


def _install_shutdown_handlers(loop) -> None:
    """Register the graceful-shutdown handler on the RUNNING event loop.

    A signal.signal(SIGINT, ...) handler set in cli() does NOT survive
    asyncio.run(): the asyncio Runner installs its own SIGINT handler for the
    loop's lifetime, so a SIGINT would raise KeyboardInterrupt / cancel the task
    instead of setting shutdown_event — only SIGTERM drained (BUG-H).
    Registering on the loop makes BOTH signals set shutdown_event and drain at
    the next phase boundary. Falls back to the cli() signal.signal handlers
    where add_signal_handler is unavailable (Windows / non-main thread).
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig, None)
        except (NotImplementedError, RuntimeError):
            pass


def _render_catalog_summary(
    harvest_stats: dict,
    courseleaf_stats: dict,
    timestamp: str,
    target_bucket: str,
    edition: str,
) -> str:
    """Build the catalog-run summary markdown uploaded to the logs bucket."""
    return "\n".join([
        f"# Catalog Ingestion Summary — {timestamp}",
        "",
        "## OAI-PMH / Dataverse harvest",
        f"- records_harvested: {harvest_stats.get('records_harvested', 0)}",
        f"- hosts_harvested: {harvest_stats.get('hosts_harvested', 0)}",
        f"- hosts_failed: {harvest_stats.get('hosts_failed', 0)}",
        "",
        "## CourseLeaf harvest",
        f"- courses_harvested: {courseleaf_stats.get('courses_harvested', 0)}",
        f"- programs_harvested: {courseleaf_stats.get('programs_harvested', 0)}",
        f"- pages_rendered: {courseleaf_stats.get('pages_rendered', 0)}",
        f"- pages_failed: {courseleaf_stats.get('pages_failed', 0)}",
        "",
        "## Config",
        f"- bucket: {target_bucket}",
        f"- edition: {edition}",
        "",
    ])


def _build_catalog_summary_json(
    harvest_stats: dict,
    courseleaf_stats: dict,
    target_bucket: str,
    edition: str,
    config: CrawlerConfig,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
) -> str:
    """Build the machine-readable JSON summary uploaded to the logs bucket."""
    return json.dumps({
        "run_id": config.run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration_seconds, 2),
        "bucket": target_bucket,
        "edition": edition,
        "oai_dataverse_harvest": harvest_stats,
        "courseleaf_harvest": courseleaf_stats,
    }, indent=2, default=str)


async def _run(config: CrawlerConfig, storage: MinioStorage) -> None:
    # Install shutdown handlers on THIS event loop so both SIGINT and SIGTERM set
    # shutdown_event and drain at a phase boundary — the cli() signal.signal
    # handlers don't survive asyncio.run for SIGINT (BUG-H).
    _install_shutdown_handlers(asyncio.get_running_loop())

    # Everything the catalog job produces — CourseLeaf course/program cards AND
    # OAI/Dataverse repository records — goes into the ONE main corpus bucket the
    # chunker reads. There is no separate course-catalog bucket anymore.
    # catalog_main runs no mirror pass, so this targets the cleaned bucket directly.
    target_bucket = config.minio_data_bucket
    edition = config.catalog_edition
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    t0 = time.monotonic()

    # Partial-run state: initialised before the try so the finally can always
    # build a summary even if a phase crashes.
    harvest_stats: dict = {"records_harvested": 0, "hosts_harvested": 0, "hosts_failed": 0}
    courseleaf_stats: dict = {
        "courses_harvested": 0, "programs_harvested": 0,
        "pages_rendered": 0, "pages_failed": 0,
    }
    run_status = "starting"
    failure_msg: str | None = None

    try:
        storage.ensure_bucket(target_bucket)
        logger.info(
            "Catalog ingestion starting — bucket=%s edition=%s",
            target_bucket,
            edition,
        )

        # OAI/Dataverse catalog record harvest (repositories, archives).
        run_status = "oai_harvest"
        try:
            harvest_stats = harvest_catalog_records(
                config.host_registry_path, storage, target_bucket
            )
            logger.info("Catalog record harvest: %s", harvest_stats)
        except StorageFullError:
            raise  # fatal: abort the run + exit non-zero (issue #5)
        except Exception:
            logger.warning("Catalog record harvest failed (non-fatal)", exc_info=True)
            harvest_stats = {"records_harvested": 0, "hosts_harvested": 0, "hosts_failed": 0}

        # Graceful shutdown: skip the CourseLeaf phase if SIGTERM/SIGINT was
        # received during (or before) the OAI harvest. The OAI results already
        # written are kept; the run exits cleanly with status=interrupted.
        if shutdown_event.is_set():
            logger.warning(
                "Shutdown requested — skipping CourseLeaf phase "
                "(OAI harvest already complete)"
            )
            run_status = "interrupted"
        else:
            # CourseLeaf structured ingest: course and program cards from catalog.vt.edu.
            run_status = "courseleaf"
            try:
                courseleaf_stats = await harvest_courseleaf_catalog(
                    storage,
                    target_bucket,
                    catalog_edition=edition,
                    threshold=config.prune_content_threshold,
                    chunk_size=config.catalog_render_chunk_size,
                    idle_timeout=config.catalog_render_idle_timeout,
                    max_retries=config.catalog_render_max_retries,
                )
                logger.info("CourseLeaf harvest: %s", courseleaf_stats)
            except StorageFullError:
                raise  # fatal: abort the run + exit non-zero (issue #5)
            except Exception:
                logger.exception("CourseLeaf harvest failed")
                courseleaf_stats = {
                    "courses_harvested": 0,
                    "programs_harvested": 0,
                    "pages_rendered": 0,
                    "pages_failed": 0,
                }

        if shutdown_event.is_set():
            run_status = "interrupted"
        else:
            run_status = "completed"

        logger.info(
            "Catalog ingestion %s — "
            "courses=%d programs=%d catalog_records=%d -> %s edition=%s",
            run_status,
            courseleaf_stats.get("courses_harvested", 0),
            courseleaf_stats.get("programs_harvested", 0),
            harvest_stats.get("records_harvested", 0),
            target_bucket,
            edition,
        )
    except Exception as exc:
        failure_msg = f"{type(exc).__name__}: {exc}"
        run_status = "failed"
        logger.error(
            "Catalog ingestion failed during '%s': %s", run_status, failure_msg,
            exc_info=True,
        )
    finally:
        # Always write a best-effort summary, even on failure/interruption.
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        ended_at = datetime.now(timezone.utc).isoformat()
        duration = time.monotonic() - t0
        try:
            storage.ensure_bucket(config.minio_logs_bucket)
            summary_md = _render_catalog_summary(
                harvest_stats, courseleaf_stats, timestamp,
                target_bucket, edition,
            )
            summary_key = f"catalog-summary-{timestamp}.md"
            storage.upload_document(summary_key, summary_md, bucket=config.minio_logs_bucket)

            summary_json = _build_catalog_summary_json(
                harvest_stats, courseleaf_stats, target_bucket, edition,
                config, started_at, ended_at, duration,
            )
            json_key = f"catalog-summary-{timestamp}.json"
            storage.upload_document(json_key, summary_json, bucket=config.minio_logs_bucket)

            logger.info(
                "Catalog summary written to %s/%s and %s/%s (status=%s, %.1fs)",
                config.minio_logs_bucket, summary_key,
                config.minio_logs_bucket, json_key, run_status, duration,
            )
        except Exception as exc:
            logger.warning("Could not upload catalog summary (non-fatal): %s", exc)

    # Exit code: 0 for clean completion OR graceful drain; 1 for failure.
    if run_status == "failed":
        sys.exit(1)
    sys.exit(0)


def run_catalog(config: CrawlerConfig, storage: MinioStorage) -> None:
    """Run the catalog pipeline under a single-runner lock.

    Raises ``RunnerLockHeld`` immediately if another catalog process already
    holds the lock — the caller maps that to a clean refusal/exit rather than
    letting two Jobs race on identical bucket keys.
    """
    with single_runner_lock(config.catalog_lock_path):
        asyncio.run(_run(config, storage))


def cli() -> None:
    """CLI entrypoint for standalone catalog ingestion."""
    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    config.configure_logging()
    logger.info("Config: %s", repr(config))

    # Pre-loop fallback handlers so a signal arriving BEFORE asyncio.run starts
    # (config load, lock acquire) still requests a drain. Once the loop is
    # running, _run re-registers both signals via loop.add_signal_handler — that
    # is what makes SIGINT actually drain (asyncio.run's Runner would otherwise
    # own SIGINT and raise KeyboardInterrupt instead). See BUG-H.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_shutdown)

    storage = MinioStorage(config)
    try:
        run_catalog(config, storage)
    except RunnerLockHeld as exc:
        logger.error("Refusing to start — %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    cli()
