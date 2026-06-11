#!/usr/bin/env python3
"""Verify ingested markdown quality + completeness; write a report to the logs bucket.

    uv run python scripts/verify_ingestion.py --bucket crawled-pages-cleaned [--frontier f.db]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import logging
from datetime import datetime, timezone

from crawler.config import CrawlerConfig
from crawler.frontier import Frontier
from crawler.storage import MinioStorage
from crawler.verify import assess_document, render_report, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="crawled-pages-cleaned")
    p.add_argument("--frontier", default=None)
    args = p.parse_args()

    config = CrawlerConfig.from_env()
    storage = MinioStorage(config)

    rows = []
    for key in storage.list_objects(args.bucket):
        if not key.endswith(".md"):
            continue
        rows.append(assess_document(key, storage.download_document(key, bucket=args.bucket)))

    summary = summarize(rows)
    failures = [r for r in rows if not r.ok]
    report = render_report(summary, failures)

    if args.frontier:
        fr = Frontier(args.frontier)
        report += f"\n## Frontier status\n\n```\n{fr.count_by_status()}\n```\n"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    storage.ensure_bucket("logs")
    storage.upload_document(f"verification-{ts}.md", report, bucket="logs")
    logger.info("Verification: %s", summary)
    print(report)


if __name__ == "__main__":
    main()
