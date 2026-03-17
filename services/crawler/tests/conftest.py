"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock
from crawler.config import CrawlerConfig


@pytest.fixture
def crawler_config():
    """Minimal crawler config for testing."""
    return CrawlerConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="test-bucket",
        minio_secure=False,
        seed_url="https://website.cs.vt.edu",
        max_depth=2,
        max_pages=100,
        allowed_domains=("website.cs.vt.edu", "students.cs.vt.edu"),
        blocked_domains=("git.cs.vt.edu", "login.cs.vt.edu"),
        prune_threshold=0.45,
    )


@pytest.fixture
def mock_minio_client():
    """A MagicMock standing in for minio.Minio."""
    client = MagicMock()
    client.bucket_exists.return_value = True
    client.put_object.return_value = MagicMock(
        object_name="cs.vt.edu/index.md", etag="abc123"
    )
    return client
