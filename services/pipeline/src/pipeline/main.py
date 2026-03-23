"""Pipeline CLI — orchestrate crawler → chunker → embedder sequentially."""
from __future__ import annotations

import argparse
import logging
import sys
import uuid

from kubernetes import client, config

from pipeline.config import PipelineConfig
from pipeline.jobs import (
    build_chunker_job,
    build_crawler_job,
    build_embedder_job,
    cleanup_job,
    wait_for_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STAGES = [
    ("crawler", build_crawler_job),
    ("chunker", build_chunker_job),
    ("embedder", build_embedder_job),
]


def _load_k8s_config() -> None:
    """Load kubeconfig — in-cluster if available, else local ~/.kube/config."""
    try:
        config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes config")
    except config.ConfigException:
        config.load_kube_config()
        logger.info("Loaded local kubeconfig (~/.kube/config)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HokieHelp ingestion pipeline — crawl → chunk → embed",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Crawl max depth (default: from env/configmap or 2)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Crawl max pages (default: from env/configmap or 500)",
    )
    parser.add_argument(
        "--seed-url",
        type=str,
        default=None,
        help="Crawl seed URL (default: from env/configmap or https://website.cs.vt.edu)",
    )
    return parser.parse_args(argv)


def run_pipeline(cfg: PipelineConfig) -> bool:
    """Run all 3 stages sequentially. Returns True if all succeed."""
    _load_k8s_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()

    run_id = uuid.uuid4().hex[:8]
    logger.info("Pipeline run %s starting", run_id)
    logger.info(
        "Config: seed_url=%s, max_depth=%d, max_pages=%d",
        cfg.crawl_seed_url,
        cfg.crawl_max_depth,
        cfg.crawl_max_pages,
    )

    for stage_name, build_fn in STAGES:
        logger.info("=== Stage: %s ===", stage_name)

        job_spec = build_fn(cfg, run_id)
        job_name = job_spec.metadata.name

        try:
            batch_api.create_namespaced_job(
                namespace=cfg.namespace, body=job_spec
            )
            logger.info("Created job %s", job_name)
        except client.ApiException as exc:
            logger.error(
                "Failed to create job %s: %s %s",
                job_name,
                exc.status,
                exc.reason,
            )
            return False

        success = wait_for_job(
            batch_api,
            core_api,
            name=job_name,
            namespace=cfg.namespace,
            poll_interval=cfg.poll_interval,
        )

        if not success:
            logger.error(
                "Stage %s failed — aborting pipeline run %s",
                stage_name,
                run_id,
            )
            return False

        cleanup_job(batch_api, job_name, cfg.namespace)

    logger.info("Pipeline run %s completed successfully", run_id)
    return True


def cli(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)

    cfg = PipelineConfig.build(
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        seed_url=args.seed_url,
    )

    success = run_pipeline(cfg)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    cli()
