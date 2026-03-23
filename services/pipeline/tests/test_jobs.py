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
