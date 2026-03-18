"""Shared test fixtures for the embedder service."""
import pytest
from embedder.config import EmbedderConfig


@pytest.fixture
def embedder_config():
    return EmbedderConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        minio_chunks_bucket="chunks",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_chunks",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_batch_size=32,
    )
