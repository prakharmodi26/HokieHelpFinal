import io
import pytest
from unittest.mock import MagicMock, patch, call

from crawler.storage import MinioStorage


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
