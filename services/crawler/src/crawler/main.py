"""Crawler entrypoint — registry seed → frontier → render → mirror → documents → summary."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from crawler.config import CrawlerConfig
from crawler.discovery.hostfilter import HostFilter
from crawler.documents import download_and_process_documents
from crawler.engine import run_engine
from crawler.frontier import Frontier
from crawler.registry import host_root_urls, load_modes, load_seed_hosts
from crawler.scope import Scope, classify_url
from crawler.seed_docs import build_department_info_doc
from crawler.storage import MinioStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_BUCKET = "logs"


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
    host_filter = HostFilter.from_yaml(config.host_blocklist_path)

    counts = frontier.count_by_status()
    if counts.get("pending", 0) > 0 or counts.get("done", 0) > 0:
        logger.info("Resuming existing frontier: %s", counts)
        return {"urls_enqueued": 0, "resumed": True, "host_filter": host_filter}

    hosts = load_seed_hosts(config.host_registry_path)
    if config.max_hosts is not None:
        hosts = hosts[: config.max_hosts]

    seeded = 0
    for url in host_root_urls(hosts):
        scope, hops = classify_url(
            url,
            parent_hops=[],
            external_hops=config.external_hops,
            host_filter=host_filter,
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
    return {"urls_enqueued": seeded, "host_filter": host_filter}


def mirror_vt_pages(storage, src_bucket: str, dst_bucket: str) -> int:
    """Copy every .md page (and its metadata sidecar) from src to dst bucket.

    fit_markdown extraction is generic, so pages are mirrored verbatim — the
    cleaned bucket is simply the bucket downstream consumers (chunker) read.
    Returns the number of pages mirrored.
    """
    mirrored = 0
    for key in storage.list_objects(bucket=src_bucket):
        if not key.endswith(".md"):
            continue  # .meta.json sidecars are mirrored alongside their .md below
        content = storage.download_document(key, bucket=src_bucket)
        storage.upload_document(key, content, bucket=dst_bucket)
        try:
            meta = storage.download_metadata(key, bucket=src_bucket)
            storage.upload_metadata(key, meta, bucket=dst_bucket)
        except Exception as exc:
            logger.warning("Could not mirror metadata sidecar for %s: %s", key, exc)
        mirrored += 1
    return mirrored


def _render_summary(
    counts: dict, stats: dict, mirrored: int, doc_stats: dict, timestamp: str
) -> str:
    """Build the run-summary markdown uploaded to the logs bucket."""
    lines = [
        f"# Crawl Summary — {timestamp}",
        "",
        "## Frontier",
    ]
    for status in sorted(counts):
        lines.append(f"- {status}: {counts[status]}")
    lines += ["", "## Render"]
    for k in sorted(stats):
        if k == "document_urls":
            continue
        lines.append(f"- {k}: {stats[k]}")
    lines += [
        "",
        "## Post-processing",
        f"- pages_mirrored_to_cleaned: {mirrored}",
        f"- documents_processed: {doc_stats.get('documents_processed', 0)}",
        f"- documents_failed: {doc_stats.get('documents_failed', 0)}",
        "",
    ]
    return "\n".join(lines)


def cli() -> None:
    """CLI entrypoint: seed frontier from registry, render, mirror, documents."""
    logger.info("Starting HokieHelp crawler (registry-seeded frontier)")

    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = MinioStorage(config)
    for bucket in (
        config.minio_bucket,
        config.minio_cleaned_bucket,
        config.minio_external_bucket,
        LOGS_BUCKET,
    ):
        storage.ensure_bucket(bucket)

    frontier = Frontier(config.frontier_db_path)
    try:
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

        stats = asyncio.run(
            run_engine(
                frontier,
                storage,
                main_bucket=config.minio_bucket,
                external_bucket=config.minio_external_bucket,
                external_hops=config.external_hops,
                max_depth=config.max_depth,
                batch_size=config.batch_size,
                memory_pct=config.memory_threshold_percent,
                max_sessions=config.max_sessions,
                threshold=config.prune_content_threshold,
                host_filter=host_filter,
                max_urls_per_host=config.max_urls_per_host,
                # Registry host modes: full ingest vs catalog cards vs
                # special handlers (Phase 8 mode routing).
                modes=load_modes(config.host_registry_path),
            )
        )
        logger.info(
            "Render complete: %s",
            {k: v for k, v in stats.items() if k != "document_urls"},
        )

        # Mirror VT pages to the cleaned bucket (chunker reads from there).
        mirrored = mirror_vt_pages(
            storage, config.minio_bucket, config.minio_cleaned_bucket
        )
        logger.info(
            "Mirrored %d pages from %s to %s",
            mirrored,
            config.minio_bucket,
            config.minio_cleaned_bucket,
        )

        # Seed doc so RAG has campus addresses, phones, social links.
        storage.upload_document(
            "_department-info.md",
            build_department_info_doc(),
            bucket=config.minio_cleaned_bucket,
        )

        # Documents pass: PDFs/DOCX discovered during the crawl go straight to
        # the cleaned bucket so they reach the chunker.
        doc_stats = {"documents_processed": 0, "documents_failed": 0}
        doc_urls = stats.get("document_urls", set())
        if doc_urls:
            hashes = storage.load_all_content_hashes(config.minio_cleaned_bucket)
            doc_stats = asyncio.run(
                download_and_process_documents(
                    doc_urls,
                    storage,
                    seen_content_hashes=set(hashes.keys()),
                    stored_urls=set(hashes.values()),
                    request_delay=config.request_delay,
                    target_bucket=config.minio_cleaned_bucket,
                )
            )
            logger.info(
                "Documents: %d processed, %d failed",
                doc_stats["documents_processed"],
                doc_stats["documents_failed"],
            )

        # Run summary to the logs bucket.
        counts = frontier.count_by_status()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        summary_key = f"crawl-summary-{timestamp}.md"
        storage.upload_document(
            summary_key,
            _render_summary(counts, stats, mirrored, doc_stats, timestamp),
            bucket=LOGS_BUCKET,
        )
        logger.info("Crawl summary uploaded to %s/%s", LOGS_BUCKET, summary_key)
    finally:
        frontier.close()


if __name__ == "__main__":
    cli()
