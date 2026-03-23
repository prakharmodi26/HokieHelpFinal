# Ingestion Pipeline Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3 independent time-offset CronJobs with a single Python orchestrator that creates K8s Jobs sequentially (crawler → chunker → embedder), waitable from laptop or schedulable via CronJob.

**Architecture:** A lightweight Python script uses the `kubernetes` client to create Jobs for each pipeline stage, polls until completion, then proceeds to the next stage. It auto-detects whether it's running in-cluster or from a local kubeconfig. CLI args override crawl config (max-depth, max-pages, seed-url); defaults come from a ConfigMap or env vars.

**Tech Stack:** Python 3.11, `kubernetes` Python client, `argparse`, Kubernetes RBAC (ServiceAccount + Role + RoleBinding)

---

## File Structure

| File | Responsibility |
|---|---|
| `services/pipeline/src/pipeline/main.py` | CLI entrypoint — parse args, run 3-stage orchestration |
| `services/pipeline/src/pipeline/jobs.py` | K8s Job creation, polling, log streaming, cleanup |
| `services/pipeline/src/pipeline/config.py` | Config dataclass — merge env vars + CLI overrides |
| `services/pipeline/src/pipeline/__init__.py` | Package init |
| `services/pipeline/pyproject.toml` | Dependencies: `kubernetes>=29.0` |
| `services/pipeline/Dockerfile` | Tiny image: python:3.11-slim + kubernetes client |
| `services/pipeline/tests/test_config.py` | Unit tests for config merging and defaults |
| `services/pipeline/tests/test_jobs.py` | Unit tests for job spec building (no cluster needed) |
| `k8s/pipeline-cronjob.yaml` | CronJob: Sunday 3AM, runs pipeline with defaults |
| `k8s/pipeline-configmap.yaml` | Default crawl overrides for scheduled runs |
| `k8s/pipeline-rbac.yaml` | ServiceAccount + Role + RoleBinding for Job management |
| `.github/workflows/pipeline-ci.yaml` | Build & push pipeline image to GHCR |

**Files to delete:**
| File | Reason |
|---|---|
| `k8s/crawler-cronjob.yaml` | Replaced by pipeline-cronjob.yaml |
| `k8s/chunker-cronjob.yaml` | Replaced by pipeline-cronjob.yaml |
| `k8s/embedder-cronjob.yaml` | Replaced by pipeline-cronjob.yaml |

**Files unchanged:** All existing service code, Dockerfiles, configmaps (crawler-config, chunker-config, embedder-config), secrets, deployments.

---

### Task 1: Pipeline config module

**Files:**
- Create: `services/pipeline/src/pipeline/__init__.py`
- Create: `services/pipeline/src/pipeline/config.py`
- Create: `services/pipeline/tests/__init__.py`
- Create: `services/pipeline/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `services/pipeline/tests/test_config.py`:

```python
"""Tests for pipeline config merging."""
import os
import pytest
from pipeline.config import PipelineConfig


class TestPipelineConfigDefaults:
    def test_defaults(self):
        cfg = PipelineConfig.build()
        assert cfg.namespace == "test"
        assert cfg.crawl_max_depth == 2
        assert cfg.crawl_max_pages == 500
        assert cfg.crawl_seed_url == "https://website.cs.vt.edu"
        assert cfg.poll_interval == 30

    def test_cli_overrides(self):
        cfg = PipelineConfig.build(
            max_depth=4,
            max_pages=1000,
            seed_url="https://example.com",
        )
        assert cfg.crawl_max_depth == 4
        assert cfg.crawl_max_pages == 1000
        assert cfg.crawl_seed_url == "https://example.com"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_NAMESPACE", "prod")
        monkeypatch.setenv("CRAWL_MAX_DEPTH", "5")
        monkeypatch.setenv("CRAWL_MAX_PAGES", "2000")
        monkeypatch.setenv("CRAWL_SEED_URL", "https://other.vt.edu")
        cfg = PipelineConfig.build()
        assert cfg.namespace == "prod"
        assert cfg.crawl_max_depth == 5
        assert cfg.crawl_max_pages == 2000
        assert cfg.crawl_seed_url == "https://other.vt.edu"

    def test_cli_beats_env(self, monkeypatch):
        monkeypatch.setenv("CRAWL_MAX_DEPTH", "5")
        cfg = PipelineConfig.build(max_depth=3)
        assert cfg.crawl_max_depth == 3
```

Create `services/pipeline/tests/__init__.py` (empty).

- [ ] **Step 2: Create the config module**

Create `services/pipeline/src/pipeline/__init__.py` (empty).

Create `services/pipeline/src/pipeline/config.py`:

```python
"""Pipeline configuration — merge env vars with CLI overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    namespace: str
    crawl_seed_url: str
    crawl_max_depth: int
    crawl_max_pages: int
    poll_interval: int  # seconds between job status checks

    # Docker images for each stage
    crawler_image: str
    chunker_image: str
    embedder_image: str

    @staticmethod
    def build(
        *,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        seed_url: Optional[str] = None,
    ) -> PipelineConfig:
        """Build config: CLI args > env vars > hardcoded defaults."""
        return PipelineConfig(
            namespace=os.getenv("PIPELINE_NAMESPACE", "test"),
            crawl_seed_url=(
                seed_url
                or os.getenv("CRAWL_SEED_URL", "https://website.cs.vt.edu")
            ),
            crawl_max_depth=(
                max_depth
                if max_depth is not None
                else int(os.getenv("CRAWL_MAX_DEPTH", "2"))
            ),
            crawl_max_pages=(
                max_pages
                if max_pages is not None
                else int(os.getenv("CRAWL_MAX_PAGES", "500"))
            ),
            poll_interval=int(os.getenv("PIPELINE_POLL_INTERVAL", "30")),
            crawler_image=os.getenv(
                "CRAWLER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-crawler:latest",
            ),
            chunker_image=os.getenv(
                "CHUNKER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-chunker:latest",
            ),
            embedder_image=os.getenv(
                "EMBEDDER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-embedder:latest",
            ),
        )
```

- [ ] **Step 3: Create pyproject.toml**

Create `services/pipeline/pyproject.toml`:

```toml
[project]
name = "hokiehelp-pipeline"
version = "0.1.0"
description = "Sequential ingestion pipeline orchestrator for HokieHelp"
requires-python = ">=3.11"
dependencies = [
    "kubernetes>=29.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[project.scripts]
hokiehelp-pipeline = "pipeline.main:cli"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/pipeline && pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/pipeline/
git commit -m "feat(pipeline): add config module with env + CLI override merging"
```

---

### Task 2: Job builder and poller (core orchestration)

**Files:**
- Create: `services/pipeline/src/pipeline/jobs.py`
- Create: `services/pipeline/tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

Create `services/pipeline/tests/test_jobs.py`:

```python
"""Tests for K8s Job spec building (no cluster required)."""
import pytest
from pipeline.config import PipelineConfig
from pipeline.jobs import build_crawler_job, build_chunker_job, build_embedder_job


@pytest.fixture
def config():
    return PipelineConfig.build(max_depth=3, max_pages=100000)


class TestBuildCrawlerJob:
    def test_job_name_prefix(self, config):
        job = build_crawler_job(config, run_id="abc123")
        assert job.metadata.name == "pipeline-crawler-abc123"
        assert job.metadata.namespace == "test"

    def test_crawl_overrides_in_env(self, config):
        job = build_crawler_job(config, run_id="abc123")
        container = job.spec.template.spec.containers[0]
        env_dict = {e.name: e.value for e in container.env if e.value is not None}
        assert env_dict["CRAWL_MAX_DEPTH"] == "3"
        assert env_dict["CRAWL_MAX_PAGES"] == "100000"

    def test_image(self, config):
        job = build_crawler_job(config, run_id="abc123")
        container = job.spec.template.spec.containers[0]
        assert container.image == "ghcr.io/prakharmodi26/hokiehelp-crawler:latest"

    def test_minio_secrets_present(self, config):
        job = build_crawler_job(config, run_id="abc123")
        container = job.spec.template.spec.containers[0]
        secret_envs = {
            e.name: e.value_from.secret_key_ref.key
            for e in container.env
            if e.value_from and e.value_from.secret_key_ref
        }
        assert secret_envs["MINIO_ACCESS_KEY"] == "access-key"
        assert secret_envs["MINIO_SECRET_KEY"] == "secret-key"


class TestBuildChunkerJob:
    def test_job_name_prefix(self, config):
        job = build_chunker_job(config, run_id="abc123")
        assert job.metadata.name == "pipeline-chunker-abc123"

    def test_uses_chunker_config(self, config):
        job = build_chunker_job(config, run_id="abc123")
        container = job.spec.template.spec.containers[0]
        refs = [e.config_map_ref.name for e in container.env_from]
        assert "chunker-config" in refs


class TestBuildEmbedderJob:
    def test_job_name_prefix(self, config):
        job = build_embedder_job(config, run_id="abc123")
        assert job.metadata.name == "pipeline-embedder-abc123"

    def test_gpu_requested(self, config):
        job = build_embedder_job(config, run_id="abc123")
        container = job.spec.template.spec.containers[0]
        assert container.resources.requests["nvidia.com/gpu"] == "1"
        assert container.resources.limits["nvidia.com/gpu"] == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/pipeline && pytest tests/test_jobs.py -v`
Expected: FAIL (ImportError — `pipeline.jobs` not found)

- [ ] **Step 3: Implement the jobs module**

Create `services/pipeline/src/pipeline/jobs.py`:

```python
"""Build and manage Kubernetes Jobs for each pipeline stage."""
from __future__ import annotations

import logging
import time
from typing import Optional

from kubernetes import client

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def _minio_secret_envs() -> list[client.V1EnvVar]:
    """Common MinIO credential env vars from the minio-credentials secret."""
    return [
        client.V1EnvVar(
            name="MINIO_ACCESS_KEY",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name="minio-credentials", key="access-key"
                )
            ),
        ),
        client.V1EnvVar(
            name="MINIO_SECRET_KEY",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name="minio-credentials", key="secret-key"
                )
            ),
        ),
    ]


def _pipeline_labels(stage: str, run_id: str) -> dict[str, str]:
    """Standard labels for pipeline jobs."""
    return {
        "app": f"hokiehelp-{stage}",
        "pipeline-run": run_id,
        "managed-by": "hokiehelp-pipeline",
    }


def build_crawler_job(config: PipelineConfig, run_id: str) -> client.V1Job:
    """Build a K8s Job spec for the crawler stage."""
    env_overrides = [
        client.V1EnvVar(name="CRAWL_MAX_DEPTH", value=str(config.crawl_max_depth)),
        client.V1EnvVar(name="CRAWL_MAX_PAGES", value=str(config.crawl_max_pages)),
        client.V1EnvVar(name="CRAWL_SEED_URL", value=config.crawl_seed_url),
    ]

    container = client.V1Container(
        name="crawler",
        image=config.crawler_image,
        image_pull_policy="Always",
        env_from=[
            client.V1EnvFromSource(
                config_map_ref=client.V1ConfigMapEnvSource(name="crawler-config")
            )
        ],
        env=env_overrides + _minio_secret_envs(),
        resources=client.V1ResourceRequirements(
            requests={"memory": "512Mi", "cpu": "250m"},
            limits={"memory": "2Gi", "cpu": "1000m"},
        ),
    )

    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=f"pipeline-crawler-{run_id}",
            namespace=config.namespace,
            labels=_pipeline_labels("crawler", run_id),
        ),
        spec=client.V1JobSpec(
            backoff_limit=2,
            active_deadline_seconds=7200,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=_pipeline_labels("crawler", run_id),
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                ),
            ),
        ),
    )


def build_chunker_job(config: PipelineConfig, run_id: str) -> client.V1Job:
    """Build a K8s Job spec for the chunker stage."""
    container = client.V1Container(
        name="chunker",
        image=config.chunker_image,
        image_pull_policy="Always",
        env_from=[
            client.V1EnvFromSource(
                config_map_ref=client.V1ConfigMapEnvSource(name="chunker-config")
            )
        ],
        env=_minio_secret_envs(),
        resources=client.V1ResourceRequirements(
            requests={"memory": "256Mi", "cpu": "250m"},
            limits={"memory": "1Gi", "cpu": "500m"},
        ),
    )

    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=f"pipeline-chunker-{run_id}",
            namespace=config.namespace,
            labels=_pipeline_labels("chunker", run_id),
        ),
        spec=client.V1JobSpec(
            backoff_limit=2,
            active_deadline_seconds=3600,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=_pipeline_labels("chunker", run_id),
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                ),
            ),
        ),
    )


def build_embedder_job(config: PipelineConfig, run_id: str) -> client.V1Job:
    """Build a K8s Job spec for the embedder stage."""
    container = client.V1Container(
        name="embedder",
        image=config.embedder_image,
        image_pull_policy="Always",
        env_from=[
            client.V1EnvFromSource(
                config_map_ref=client.V1ConfigMapEnvSource(name="embedder-config")
            )
        ],
        env=_minio_secret_envs(),
        resources=client.V1ResourceRequirements(
            requests={"memory": "4Gi", "cpu": "500m", "nvidia.com/gpu": "1"},
            limits={"memory": "8Gi", "cpu": "2000m", "nvidia.com/gpu": "1"},
        ),
    )

    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=f"pipeline-embedder-{run_id}",
            namespace=config.namespace,
            labels=_pipeline_labels("embedder", run_id),
        ),
        spec=client.V1JobSpec(
            backoff_limit=2,
            active_deadline_seconds=14400,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=_pipeline_labels("embedder", run_id),
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                ),
            ),
        ),
    )


def wait_for_job(
    batch_api: client.BatchV1Api,
    core_api: client.CoreV1Api,
    name: str,
    namespace: str,
    poll_interval: int = 30,
    timeout: Optional[int] = None,
) -> bool:
    """Poll a Job until it succeeds or fails. Stream logs from the pod.

    Returns True if the job succeeded, False otherwise.
    """
    start = time.time()
    while True:
        job = batch_api.read_namespaced_job(name=name, namespace=namespace)
        status = job.status

        # Stream latest logs from the job's pod on every poll cycle
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={name}",
        )
        for pod in pods.items:
            if pod.status.phase in ("Running", "Succeeded", "Failed"):
                _stream_pod_logs(core_api, pod.metadata.name, namespace)
                break

        if status.succeeded and status.succeeded > 0:
            logger.info("Job %s succeeded", name)
            return True

        if status.failed and status.failed > 0:
            logger.error("Job %s failed", name)
            _log_pod_events(core_api, name, namespace)
            return False

        if timeout and (time.time() - start) > timeout:
            logger.error("Job %s timed out after %ds", name, timeout)
            return False

        elapsed = int(time.time() - start)
        logger.info("Job %s still running (%ds elapsed)...", name, elapsed)
        time.sleep(poll_interval)


def _stream_pod_logs(
    core_api: client.CoreV1Api, pod_name: str, namespace: str
) -> None:
    """Fetch and log the latest pod logs (non-blocking)."""
    try:
        logs = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=50,
        )
        if logs:
            for line in logs.strip().split("\n"):
                logger.info("  [pod] %s", line)
    except Exception as exc:
        logger.debug("Could not read logs for pod %s: %s", pod_name, exc)


def _log_pod_events(
    core_api: client.CoreV1Api, job_name: str, namespace: str
) -> None:
    """Log pod events for a failed job to aid debugging."""
    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
        for pod in pods.items:
            events = core_api.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod.metadata.name}",
            )
            for event in events.items:
                logger.error(
                    "  [event] %s: %s", event.reason, event.message
                )
    except Exception as exc:
        logger.debug("Could not fetch events for job %s: %s", job_name, exc)


def cleanup_job(
    batch_api: client.BatchV1Api, name: str, namespace: str
) -> None:
    """Delete a completed job and its pods."""
    try:
        batch_api.delete_namespaced_job(
            name=name,
            namespace=namespace,
            body=client.V1DeleteOptions(propagation_policy="Background"),
        )
        logger.info("Cleaned up job %s", name)
    except Exception as exc:
        logger.warning("Could not clean up job %s: %s", name, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/pipeline && pytest tests/test_jobs.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/pipeline/src/pipeline/jobs.py services/pipeline/tests/test_jobs.py
git commit -m "feat(pipeline): add K8s job builders and poll/wait logic"
```

---

### Task 3: CLI entrypoint (main.py)

**Files:**
- Create: `services/pipeline/src/pipeline/main.py`

- [ ] **Step 1: Write the CLI entrypoint**

Create `services/pipeline/src/pipeline/main.py`:

```python
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
```

- [ ] **Step 2: Verify it parses args correctly (quick smoke test)**

Run: `cd services/pipeline && python -c "from pipeline.main import parse_args; a = parse_args(['--max-depth', '3', '--max-pages', '1000']); print(a)"`
Expected: `Namespace(max_depth=3, max_pages=1000, seed_url=None)`

- [ ] **Step 3: Commit**

```bash
git add services/pipeline/src/pipeline/main.py
git commit -m "feat(pipeline): add CLI entrypoint with argparse and sequential orchestration"
```

---

### Task 4: Dockerfile and CI workflow

**Files:**
- Create: `services/pipeline/Dockerfile`
- Create: `.github/workflows/pipeline-ci.yaml`

- [ ] **Step 1: Create the Dockerfile**

Create `services/pipeline/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

CMD ["python", "-m", "pipeline.main"]
```

- [ ] **Step 2: Create the CI workflow**

Create `.github/workflows/pipeline-ci.yaml`:

```yaml
name: Pipeline CI

on:
  push:
    branches: [main]
    paths:
      - "services/pipeline/**"
      - ".github/workflows/pipeline-ci.yaml"
  pull_request:
    branches: [main]
    paths:
      - "services/pipeline/**"

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: prakharmodi26/hokiehelp-pipeline

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/pipeline
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ".[dev]"
      - run: pytest -v

  build-and-push:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest

      - uses: docker/build-push-action@v5
        with:
          context: services/pipeline
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 3: Commit**

```bash
git add services/pipeline/Dockerfile .github/workflows/pipeline-ci.yaml
git commit -m "feat(pipeline): add Dockerfile and CI workflow"
```

---

### Task 5: Kubernetes manifests (RBAC + ConfigMap + CronJob)

**Files:**
- Create: `k8s/pipeline-rbac.yaml`
- Create: `k8s/pipeline-configmap.yaml`
- Create: `k8s/pipeline-cronjob.yaml`
- Delete: `k8s/crawler-cronjob.yaml`
- Delete: `k8s/chunker-cronjob.yaml`
- Delete: `k8s/embedder-cronjob.yaml`

- [ ] **Step 1: Create the RBAC manifest**

Create `k8s/pipeline-rbac.yaml`:

```yaml
# ServiceAccount for the pipeline orchestrator
apiVersion: v1
kind: ServiceAccount
metadata:
  name: hokiehelp-pipeline
  namespace: test
  labels:
    app: hokiehelp-pipeline
---
# Role: create, get, watch, list, delete Jobs and read pod logs/events
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hokiehelp-pipeline-role
  namespace: test
  labels:
    app: hokiehelp-pipeline
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["get", "list"]
---
# Bind the role to the service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hokiehelp-pipeline-binding
  namespace: test
  labels:
    app: hokiehelp-pipeline
subjects:
  - kind: ServiceAccount
    name: hokiehelp-pipeline
    namespace: test
roleRef:
  kind: Role
  name: hokiehelp-pipeline-role
  apiGroup: rbac.authorization.k8s.io
```

- [ ] **Step 2: Create the ConfigMap**

Create `k8s/pipeline-configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pipeline-config
  namespace: test
  labels:
    app: hokiehelp-pipeline
data:
  # Default crawl settings for scheduled runs
  CRAWL_SEED_URL: "https://website.cs.vt.edu"
  CRAWL_MAX_DEPTH: "3"
  CRAWL_MAX_PAGES: "100000"
  # Pipeline behavior
  PIPELINE_NAMESPACE: "test"
  PIPELINE_POLL_INTERVAL: "30"
```

- [ ] **Step 3: Create the CronJob**

Create `k8s/pipeline-cronjob.yaml`:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hokiehelp-pipeline
  namespace: test
  labels:
    app: hokiehelp-pipeline
spec:
  schedule: "0 3 * * 0"  # Weekly on Sunday at 3:00 AM UTC
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 0  # Pipeline handles retries per-stage; don't re-run the whole thing
      activeDeadlineSeconds: 28800  # 8 hour overall timeout
      template:
        metadata:
          labels:
            app: hokiehelp-pipeline
        spec:
          serviceAccountName: hokiehelp-pipeline
          restartPolicy: Never
          containers:
            - name: pipeline
              image: ghcr.io/prakharmodi26/hokiehelp-pipeline:latest
              imagePullPolicy: Always
              envFrom:
                - configMapRef:
                    name: pipeline-config
              resources:
                requests:
                  memory: "64Mi"
                  cpu: "50m"
                limits:
                  memory: "128Mi"
                  cpu: "200m"
```

- [ ] **Step 4: Delete the old CronJobs**

```bash
git rm k8s/crawler-cronjob.yaml k8s/chunker-cronjob.yaml k8s/embedder-cronjob.yaml
```

- [ ] **Step 5: Commit**

```bash
git add k8s/pipeline-rbac.yaml k8s/pipeline-configmap.yaml k8s/pipeline-cronjob.yaml
git commit -m "feat(pipeline): add K8s RBAC, configmap, and cronjob; remove old per-stage cronjobs"
```

---

### Task 6: Deploy and verify

- [ ] **Step 1: Check kubectl context**

```bash
kubectl config current-context
```

- [ ] **Step 2: Apply RBAC and ConfigMap**

```bash
kubectl apply -f k8s/pipeline-rbac.yaml
kubectl apply -f k8s/pipeline-configmap.yaml
kubectl apply -f k8s/pipeline-cronjob.yaml
```

- [ ] **Step 3: Delete old CronJobs from cluster**

```bash
kubectl delete cronjob hokiehelp-crawler -n test --ignore-not-found
kubectl delete cronjob hokiehelp-chunker -n test --ignore-not-found
kubectl delete cronjob hokiehelp-embedder -n test --ignore-not-found
```

- [ ] **Step 4: Verify resources**

```bash
kubectl get serviceaccount hokiehelp-pipeline -n test
kubectl get role hokiehelp-pipeline-role -n test
kubectl get rolebinding hokiehelp-pipeline-binding -n test
kubectl get cronjob hokiehelp-pipeline -n test
kubectl get configmap pipeline-config -n test
```

---

## Usage Reference

### Scheduled runs (automatic)

The CronJob `hokiehelp-pipeline` fires every Sunday at 3:00 AM UTC with these defaults from `pipeline-config` ConfigMap:

| Setting | Value |
|---|---|
| `CRAWL_MAX_DEPTH` | `3` |
| `CRAWL_MAX_PAGES` | `100000` |
| `CRAWL_SEED_URL` | `https://website.cs.vt.edu` |

To change the cron defaults, edit `k8s/pipeline-configmap.yaml` and `kubectl apply -f`.

### Manual run from your laptop

**Prerequisites:** `pip install kubernetes` and `kubectl` logged in to the cluster.

```bash
# Install the pipeline package locally
cd services/pipeline && pip install .

# Run with defaults (reads from env or hardcoded defaults)
hokiehelp-pipeline

# Override crawl config for a one-off deep crawl
hokiehelp-pipeline --max-depth 4 --max-pages 2000

# Override seed URL for a different site
hokiehelp-pipeline --seed-url https://cs.vt.edu/other --max-depth 2 --max-pages 100
```

### Manual run via kubectl (no local install)

```bash
# Create a one-off Job from the CronJob template (uses ConfigMap defaults)
kubectl create job pipeline-manual-$(date +%s) --from=cronjob/hokiehelp-pipeline -n test

# Watch it
kubectl logs -f -l managed-by=hokiehelp-pipeline -n test
```
