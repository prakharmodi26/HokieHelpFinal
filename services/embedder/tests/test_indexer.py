"""Tests for Qdrant indexer using mocked client."""
import uuid
from unittest.mock import MagicMock, patch, call
import pytest

from embedder.indexer import QdrantIndexer, chunk_id_to_uuid


# --- chunk_id_to_uuid ---

def test_chunk_id_to_uuid_deterministic():
    u1 = chunk_id_to_uuid("abc_0000")
    u2 = chunk_id_to_uuid("abc_0000")
    assert u1 == u2


def test_chunk_id_to_uuid_different_ids_differ():
    assert chunk_id_to_uuid("abc_0000") != chunk_id_to_uuid("abc_0001")


def test_chunk_id_to_uuid_is_valid_uuid():
    result = chunk_id_to_uuid("abc_0000")
    uuid.UUID(result)  # raises if invalid


# --- QdrantIndexer ---

@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    # Collection does not exist yet
    collection_info = MagicMock()
    collection_info.collections = []
    client.get_collections.return_value = collection_info
    return client


@pytest.fixture
def indexer(mock_qdrant):
    with patch("embedder.indexer.QdrantClient", return_value=mock_qdrant):
        idx = QdrantIndexer(
            host="localhost",
            port=6333,
            collection="test_chunks",
            vector_size=1024,
        )
    idx._client = mock_qdrant
    return idx


def test_ensure_collection_creates_if_missing(mock_qdrant, indexer):
    mock_qdrant.create_collection.assert_called_once()
    call_args = mock_qdrant.create_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_chunks"


def test_ensure_collection_creates_payload_index(mock_qdrant, indexer):
    mock_qdrant.create_payload_index.assert_called_once()
    call_args = mock_qdrant.create_payload_index.call_args
    assert call_args.kwargs["field_name"] == "document_id"


def test_upsert_chunks(indexer, mock_qdrant):
    chunks = [
        {
            "chunk_id": "abc_0000",
            "document_id": "abc",
            "url": "https://example.com",
            "title": "Test",
            "page_type": "general",
            "headings_path": [],
            "chunk_index": 0,
            "content_hash": "deadbeef",
            "crawl_timestamp": "2026-03-16T00:00:00",
            "token_count": 10,
        }
    ]
    embeddings = [[0.1] * 1024]
    indexer.upsert_chunks(chunks, embeddings)
    mock_qdrant.upsert.assert_called_once()
    call_args = mock_qdrant.upsert.call_args
    assert call_args.kwargs["collection_name"] == "test_chunks"
    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == chunk_id_to_uuid("abc_0000")
    assert points[0].payload["document_id"] == "abc"
    assert points[0].payload["chunk_id"] == "abc_0000"


def test_delete_stale_chunks_removes_old(indexer, mock_qdrant):
    # Simulate Qdrant returning 3 points for this doc, but only 2 are current
    old_point = MagicMock()
    old_point.id = chunk_id_to_uuid("abc_0002")
    current_point_1 = MagicMock()
    current_point_1.id = chunk_id_to_uuid("abc_0000")
    current_point_2 = MagicMock()
    current_point_2.id = chunk_id_to_uuid("abc_0001")

    mock_qdrant.scroll.return_value = ([old_point, current_point_1, current_point_2], None)

    deleted = indexer.delete_stale_chunks("abc", {"abc_0000", "abc_0001"})
    assert deleted == 1
    mock_qdrant.delete.assert_called_once()


def test_delete_stale_chunks_nothing_to_delete(indexer, mock_qdrant):
    point = MagicMock()
    point.id = chunk_id_to_uuid("abc_0000")
    mock_qdrant.scroll.return_value = ([point], None)

    deleted = indexer.delete_stale_chunks("abc", {"abc_0000"})
    assert deleted == 0
    mock_qdrant.delete.assert_not_called()
