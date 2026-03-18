"""Shared test fixtures for the chunker service."""
import pytest
from chunker.config import ChunkerConfig


@pytest.fixture
def chunker_config():
    return ChunkerConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        minio_cleaned_bucket="crawled-pages-cleaned",
        minio_chunks_bucket="chunks",
        chunk_preferred_tokens=400,
        chunk_overlap_tokens=64,
        chunk_min_tokens=120,
    )
