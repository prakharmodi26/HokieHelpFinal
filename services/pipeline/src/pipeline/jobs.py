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
