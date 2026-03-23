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
