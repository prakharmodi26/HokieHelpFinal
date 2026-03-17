import io
import json as json_module
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

from crawler.storage import MinioStorage
from crawler.metadata import PageMetadata


def test_init_creates_bucket_if_missing(crawler_config, mock_minio_client):
    """Storage creates the bucket when it doesn't exist."""
    mock_minio_client.bucket_exists.return_value = False

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)

    mock_minio_client.make_bucket.assert_called_once_with("test-bucket")


def test_init_skips_bucket_creation_if_exists(crawler_config, mock_minio_client):
    """Storage does not recreate an existing bucket."""
    mock_minio_client.bucket_exists.return_value = True

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)

    mock_minio_client.make_bucket.assert_not_called()


def test_upload_document(crawler_config, mock_minio_client):
    """Uploads markdown content to the correct object key."""
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_document("cs.vt.edu/index.md", "# Hello")

    mock_minio_client.put_object.assert_called_once()
    args, kwargs = mock_minio_client.put_object.call_args
    assert args[0] == "test-bucket"
    assert args[1] == "cs.vt.edu/index.md"
    assert kwargs["content_type"] == "text/markdown"


def test_upload_document_content_bytes(crawler_config, mock_minio_client):
    """Uploaded data matches the input content."""
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_document("cs.vt.edu/page.md", "# Test Content")

    args, kwargs = mock_minio_client.put_object.call_args
    data_stream = args[2]
    data_length = args[3]
    content = data_stream.read()
    assert content == b"# Test Content"
    assert data_length == len(b"# Test Content")


def test_upload_document_to_custom_bucket(crawler_config, mock_minio_client):
    """upload_document can target a different bucket."""
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_document("key.md", "# Hi", bucket="other-bucket")

    args, _ = mock_minio_client.put_object.call_args
    assert args[0] == "other-bucket"


def test_list_objects(crawler_config, mock_minio_client):
    """list_objects returns object names from the bucket."""
    mock_obj = MagicMock()
    mock_obj.object_name = "website.cs.vt.edu/index.md"
    mock_minio_client.list_objects.return_value = [mock_obj]

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        keys = storage.list_objects()

    assert keys == ["website.cs.vt.edu/index.md"]


def test_download_document(crawler_config, mock_minio_client):
    """download_document returns file content as string."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"# Hello"
    mock_minio_client.get_object.return_value = mock_response

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        content = storage.download_document("key.md")

    assert content == "# Hello"
    mock_response.close.assert_called_once()
    mock_response.release_conn.assert_called_once()


def test_ensure_bucket_creates_if_missing(crawler_config, mock_minio_client):
    """ensure_bucket creates a new bucket when it doesn't exist."""
    mock_minio_client.bucket_exists.return_value = False

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.ensure_bucket("new-bucket")

    mock_minio_client.make_bucket.assert_called_with("new-bucket")


# --- Metadata methods ---

def _make_test_metadata() -> PageMetadata:
    return PageMetadata(
        doc_id="abcd1234abcd1234",
        url="https://website.cs.vt.edu/about",
        title="About",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
        content_hash="deadbeef" * 8,
        markdown_size_bytes=512,
        status_code=200,
        response_headers={"content-type": "text/html"},
        internal_links=[],
        external_links=[],
        last_modified=None,
        etag=None,
    )


def test_upload_metadata_uses_meta_json_key(crawler_config, mock_minio_client):
    meta = _make_test_metadata()
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_metadata("website.cs.vt.edu/about.md", meta)
    args, kwargs = mock_minio_client.put_object.call_args
    assert args[1] == "website.cs.vt.edu/about.meta.json"
    assert kwargs["content_type"] == "application/json"


def test_upload_metadata_content_is_valid_json(crawler_config, mock_minio_client):
    meta = _make_test_metadata()
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_metadata("website.cs.vt.edu/about.md", meta)
    args, _ = mock_minio_client.put_object.call_args
    parsed = json_module.loads(args[2].read().decode("utf-8"))
    assert parsed["doc_id"] == "abcd1234abcd1234"


def test_download_metadata_round_trips(crawler_config, mock_minio_client):
    meta = _make_test_metadata()
    json_bytes = meta.to_json().encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio_client.get_object.return_value = mock_response

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        restored = storage.download_metadata("website.cs.vt.edu/about.md")

    assert restored.doc_id == "abcd1234abcd1234"
    assert restored.content_hash == "deadbeef" * 8
    mock_minio_client.get_object.assert_called_once_with(
        "test-bucket", "website.cs.vt.edu/about.meta.json"
    )


def test_load_all_content_hashes(crawler_config, mock_minio_client):
    meta = _make_test_metadata()

    mock_obj = MagicMock()
    mock_obj.object_name = "website.cs.vt.edu/about.meta.json"
    mock_minio_client.list_objects.return_value = iter([mock_obj])

    json_bytes = meta.to_json().encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio_client.get_object.return_value = mock_response

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        hashes = storage.load_all_content_hashes()

    assert ("deadbeef" * 8) in hashes
    assert hashes["deadbeef" * 8] == "https://website.cs.vt.edu/about"
