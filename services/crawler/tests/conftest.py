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
        seed_url="https://cs.vt.edu",
        max_depth=2,
        max_pages=100,
        allowed_domain="cs.vt.edu",
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
