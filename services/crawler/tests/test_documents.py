"""Tests for document URL collection and text extraction."""

import pytest
from crawler.documents import is_document_url, collect_document_links, _title_from_url


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


class TestTitleFromUrl:
    def test_simple_filename(self):
        assert _title_from_url("https://cs.vt.edu/grad-handbook.pdf") == "Grad Handbook"

    def test_underscores(self):
        assert _title_from_url("https://cs.vt.edu/annual_report.pdf") == "Annual Report"

    def test_nested_path(self):
        assert _title_from_url("https://cs.vt.edu/docs/cs-curriculum-2024.docx") == "Cs Curriculum 2024"
