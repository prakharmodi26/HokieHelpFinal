"""Tests for EmbedderStorage using mocked MinIO client."""
import json
from unittest.mock import MagicMock, patch
import pytest

from embedder.storage import EmbedderStorage


@pytest.fixture
def mock_minio():
    client = MagicMock()
    client.bucket_exists.return_value = True
    return client


@pytest.fixture
def storage(embedder_config, mock_minio):
    with patch("embedder.storage.Minio", return_value=mock_minio):
        s = EmbedderStorage(embedder_config)
    s._client = mock_minio
    return s


SAMPLE_CHUNKS = [
    {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Hello world",
        "url": "https://example.com",
        "title": "Test",
        "page_type": "general",
        "headings_path": [],
        "content_hash": "deadbeef01234567",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 2,
    }
]


def test_list_chunk_keys(storage, mock_minio):
    obj1 = MagicMock()
    obj1.object_name = "chunks/abc.json"
    obj2 = MagicMock()
    obj2.object_name = "chunks/def.json"
    obj3 = MagicMock()
    obj3.object_name = "other/file.txt"
    mock_minio.list_objects.return_value = [obj1, obj2, obj3]
    keys = storage.list_chunk_keys()
    assert keys == ["chunks/abc.json", "chunks/def.json"]


def test_download_chunks(storage, mock_minio):
    data = json.dumps(SAMPLE_CHUNKS).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = data
    mock_minio.get_object.return_value = resp
    chunks = storage.download_chunks("chunks/abc.json")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "abc_0000"
    resp.close.assert_called_once()
    resp.release_conn.assert_called_once()


def test_download_chunks_empty_file(storage, mock_minio):
    resp = MagicMock()
    resp.read.return_value = b"[]"
    mock_minio.get_object.return_value = resp
    chunks = storage.download_chunks("chunks/empty.json")
    assert chunks == []
