"""Tests for document URL collection and text extraction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawler.documents import (
    is_document_url,
    collect_document_links,
    download_and_process_documents,
    _title_from_url,
)


class TestIsDocumentUrl:
    def test_pdf(self):
        assert is_document_url("https://example.com/file.pdf") is True

    def test_docx(self):
        assert is_document_url("https://example.com/report.docx") is True

    def test_doc(self):
        assert is_document_url("https://example.com/legacy.doc") is True

    def test_html(self):
        assert is_document_url("https://example.com/page.html") is False

    def test_no_extension(self):
        assert is_document_url("https://example.com/page") is False

    def test_pdf_case_insensitive(self):
        assert is_document_url("https://example.com/File.PDF") is True

    def test_pdf_with_query(self):
        assert is_document_url("https://example.com/file.pdf?v=1") is True  # still a PDF


class TestCollectDocumentLinks:
    def test_finds_internal_pdf(self):
        links = {
            "internal": [{"href": "https://cs.vt.edu/doc.pdf"}],
            "external": [],
        }
        result = collect_document_links(links)
        assert "https://cs.vt.edu/doc.pdf" in result

    def test_finds_external_docx(self):
        links = {
            "internal": [],
            "external": [{"href": "https://other.edu/report.docx"}],
        }
        result = collect_document_links(links)
        assert "https://other.edu/report.docx" in result

    def test_skips_html(self):
        links = {
            "internal": [{"href": "https://cs.vt.edu/page.html"}],
            "external": [],
        }
        result = collect_document_links(links)
        assert len(result) == 0

    def test_handles_none_links(self):
        assert collect_document_links(None) == set()

    def test_deduplicates(self):
        links = {
            "internal": [{"href": "https://cs.vt.edu/doc.pdf"}],
            "external": [{"href": "https://cs.vt.edu/doc.pdf"}],
        }
        result = collect_document_links(links)
        assert len(result) == 1


def _run_documents_pass(**kwargs):
    """Drive download_and_process_documents against a mocked httpx client.

    Returns (stats, storage_mock). Extraction is stubbed so only the
    download/store plumbing is exercised.
    """
    storage = MagicMock()

    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"%PDF-1.4 fake"
    resp.headers = {}
    resp.raise_for_status = MagicMock()

    client = MagicMock()
    client.head = AsyncMock(side_effect=Exception("HEAD not supported"))
    client.get = AsyncMock(return_value=resp)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.documents.httpx.AsyncClient", return_value=ctx), \
         patch("crawler.documents._extract_pdf_text", return_value="Extracted text"):
        stats = asyncio.run(
            download_and_process_documents(
                {"https://cs.vt.edu/grad-handbook.pdf"},
                storage,
                seen_content_hashes=set(),
                stored_urls=set(),
                request_delay=0.0,
                **kwargs,
            )
        )
    return stats, storage


class TestDownloadAndProcessDocumentsTargetBucket:
    def test_uploads_to_target_bucket(self):
        stats, storage = _run_documents_pass(target_bucket="crawled-pages-cleaned")

        assert stats["documents_processed"] == 1
        _, doc_kwargs = storage.upload_document.call_args
        assert doc_kwargs["bucket"] == "crawled-pages-cleaned"
        _, meta_kwargs = storage.upload_metadata.call_args
        assert meta_kwargs["bucket"] == "crawled-pages-cleaned"

    def test_default_target_bucket_is_none(self):
        """No target_bucket → bucket=None → MinioStorage falls back to its default."""
        stats, storage = _run_documents_pass()

        assert stats["documents_processed"] == 1
        _, doc_kwargs = storage.upload_document.call_args
        assert doc_kwargs["bucket"] is None
        _, meta_kwargs = storage.upload_metadata.call_args
        assert meta_kwargs["bucket"] is None


class TestTitleFromUrl:
    def test_simple_filename(self):
        assert _title_from_url("https://cs.vt.edu/grad-handbook.pdf") == "Grad Handbook"

    def test_underscores(self):
        assert _title_from_url("https://cs.vt.edu/annual_report.pdf") == "Annual Report"

    def test_nested_path(self):
        assert _title_from_url("https://cs.vt.edu/docs/cs-curriculum-2024.docx") == "Cs Curriculum 2024"
