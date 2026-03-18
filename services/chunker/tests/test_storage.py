"""Tests for ChunkerStorage using a mocked Minio client."""
import json
import io
from unittest.mock import MagicMock, patch, call
import pytest

from chunker.storage import ChunkerStorage
from chunker.models import ChunkRecord


@pytest.fixture
def mock_minio():
    client = MagicMock()
    client.bucket_exists.return_value = True
    return client


@pytest.fixture
def storage(chunker_config, mock_minio):
    with patch("chunker.storage.Minio", return_value=mock_minio):
        s = ChunkerStorage(chunker_config)
    s._client = mock_minio
    return s


def test_list_markdown_keys_filters_md_only(storage, mock_minio):
    obj1 = MagicMock(); obj1.object_name = "website.cs.vt.edu/about.md"
    obj2 = MagicMock(); obj2.object_name = "website.cs.vt.edu/about.meta.json"
    obj3 = MagicMock(); obj3.object_name = "website.cs.vt.edu/people.md"
    mock_minio.list_objects.return_value = [obj1, obj2, obj3]
    keys = storage.list_markdown_keys()
    assert keys == ["website.cs.vt.edu/about.md", "website.cs.vt.edu/people.md"]


def test_download_document_returns_content(storage, mock_minio):
    resp = MagicMock()
    resp.read.return_value = b"# Hello\n\nWorld"
    mock_minio.get_object.return_value = resp
    content = storage.download_document("website.cs.vt.edu/about.md")
    assert content == "# Hello\n\nWorld"
    resp.close.assert_called_once()
    resp.release_conn.assert_called_once()


def test_upload_chunks_serialises_to_json(storage, mock_minio):
    chunks = [
        ChunkRecord(
            chunk_id="abc_0000", document_id="abc", chunk_index=0,
            text="Hello", url="https://example.com", title="Test",
            page_type="general", headings_path=[], content_hash="deadbeef01234567",
            crawl_timestamp="2026-03-16T00:00:00", token_count=1,
        )
    ]
    storage.upload_chunks("abc", chunks)
    assert mock_minio.put_object.called
    call_args = mock_minio.put_object.call_args
    bucket = call_args[0][0]
    key = call_args[0][1]
    assert bucket == "chunks"
    assert key == "chunks/abc.json"
    # Verify content is valid JSON array
    data_stream = call_args[0][2]
    content = json.loads(data_stream.read().decode())
    assert isinstance(content, list)
    assert content[0]["chunk_id"] == "abc_0000"


def test_init_creates_bucket_if_missing(chunker_config, mock_minio):
    """Verify bucket is created during __init__ when it does not exist."""
    mock_minio.bucket_exists.return_value = False
    with patch("chunker.storage.Minio", return_value=mock_minio):
        ChunkerStorage(chunker_config)
    mock_minio.make_bucket.assert_called_once_with("chunks")


def test_ensure_bucket_public_method(storage, mock_minio):
    """ChunkerStorage exposes ensure_bucket() for parity with MinioStorage."""
    mock_minio.bucket_exists.return_value = False
    storage.ensure_bucket("new-bucket")
    mock_minio.make_bucket.assert_called_with("new-bucket")
