"""Download and extract text from PDF and Word documents found during crawl."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.storage import MinioStorage

logger = logging.getLogger(__name__)

# File extensions we treat as documents
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}

# Safety limits
MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_PDF_PAGES = 500
MAX_EXTRACTED_TEXT_CHARS = 2_000_000  # 2M characters
ALLOWED_SCHEMES = {"http", "https"}


def is_document_url(url: str) -> bool:
    """Return True if the URL points to a downloadable document."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ALLOWED_SCHEMES:
        return False
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def collect_document_links(links: dict | None) -> set[str]:
    """Extract document URLs from a CrawlResult's links dict."""
    urls: set[str] = set()
    if not links:
        return urls
    for category in ("internal", "external"):
        for link in links.get(category, []):
            href = link.get("href", "")
            if href and is_document_url(href):
                urls.add(href)
    return urls


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from PDF bytes using pymupdf.

    Limits extraction to MAX_PDF_PAGES pages and MAX_EXTRACTED_TEXT_CHARS total.
    """
    try:
        import pymupdf
    except ImportError:
        logger.error("pymupdf not installed — cannot extract PDF text")
        return ""

    text_parts = []
    total_chars = 0
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        page_count = min(len(doc), MAX_PDF_PAGES)
        if len(doc) > MAX_PDF_PAGES:
            logger.warning(
                "PDF has %d pages, limiting extraction to %d", len(doc), MAX_PDF_PAGES
            )
        for i in range(page_count):
            page_text = doc[i].get_text()
            text_parts.append(page_text)
            total_chars += len(page_text)
            if total_chars >= MAX_EXTRACTED_TEXT_CHARS:
                logger.warning("Extracted text exceeds %d chars, truncating", MAX_EXTRACTED_TEXT_CHARS)
                break
        doc.close()
    except Exception as exc:
        logger.error("Failed to extract PDF text: %s", exc)
        return ""

    result = "\n\n".join(text_parts).strip()
    return result[:MAX_EXTRACTED_TEXT_CHARS]


def _extract_docx_text(data: bytes) -> str:
    """Extract text from Word .docx bytes using python-docx.

    Limits extraction to MAX_EXTRACTED_TEXT_CHARS total.
    """
    try:
        import docx
    except ImportError:
        logger.error("python-docx not installed — cannot extract Word text")
        return ""

    try:
        doc = docx.Document(io.BytesIO(data))
        parts = []
        total_chars = 0
        for p in doc.paragraphs:
            if p.text.strip():
                parts.append(p.text)
                total_chars += len(p.text)
                if total_chars >= MAX_EXTRACTED_TEXT_CHARS:
                    logger.warning("Extracted text exceeds %d chars, truncating", MAX_EXTRACTED_TEXT_CHARS)
                    break
        return "\n\n".join(parts)[:MAX_EXTRACTED_TEXT_CHARS]
    except Exception as exc:
        logger.error("Failed to extract Word text: %s", exc)
        return ""


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a document URL."""
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    # Remove extension and replace separators with spaces
    name = re.sub(r"\.[^.]+$", "", filename)
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def _validate_document_url(url: str) -> bool:
    """Validate that a document URL is safe to fetch."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        logger.warning("Rejected document URL with scheme %r: %s", parsed.scheme, url)
        return False
    if not parsed.hostname:
        logger.warning("Rejected document URL with no hostname: %s", url)
        return False
    return True


async def download_and_process_documents(
    document_urls: set[str],
    storage: MinioStorage,
    seen_content_hashes: set[str],
    stored_urls: set[str],
    request_delay: float = 0.5,
) -> dict:
    """Download documents, extract text, and store as markdown in MinIO.

    Returns stats dict with documents_processed and documents_failed counts.
    """
    stats = {"documents_processed": 0, "documents_failed": 0}

    if not document_urls:
        return stats

    logger.info("Phase 3: processing %d document URLs", len(document_urls))

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "HokieHelp-Crawler/1.0"},
    ) as client:
        for url in document_urls:
            if url in stored_urls:
                continue

            if not _validate_document_url(url):
                stats["documents_failed"] += 1
                continue

            try:
                # HEAD request first to check size
                head_resp = await client.head(url)
                content_length = head_resp.headers.get("content-length")
                if content_length and int(content_length) > MAX_DOCUMENT_SIZE_BYTES:
                    logger.warning(
                        "Document too large (%s bytes), skipping: %s",
                        content_length, url,
                    )
                    stats["documents_failed"] += 1
                    continue
            except Exception:
                pass  # HEAD may not be supported; proceed with GET

            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to download document %s: %s", url, exc)
                stats["documents_failed"] += 1
                continue

            data = resp.content

            # Enforce size limit on actual response body
            if len(data) > MAX_DOCUMENT_SIZE_BYTES:
                logger.warning(
                    "Downloaded document exceeds %d bytes (%d), skipping: %s",
                    MAX_DOCUMENT_SIZE_BYTES, len(data), url,
                )
                stats["documents_failed"] += 1
                continue

            path = urlparse(url).path.lower()

            if path.endswith(".pdf"):
                text = _extract_pdf_text(data)
            elif path.endswith(".docx"):
                text = _extract_docx_text(data)
            elif path.endswith(".doc"):
                logger.warning("Legacy .doc format not supported, skipping: %s", url)
                stats["documents_failed"] += 1
                continue
            else:
                stats["documents_failed"] += 1
                continue

            if not text.strip():
                logger.warning("No text extracted from document: %s", url)
                stats["documents_failed"] += 1
                continue

            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if content_hash in seen_content_hashes:
                logger.info("Duplicate document content: %s, skipping", url)
                continue
            seen_content_hashes.add(content_hash)

            title = _title_from_url(url)
            doc_id = doc_id_for_url(url)
            now = datetime.now(timezone.utc)

            document = build_markdown_document(
                url=url,
                title=title,
                markdown_content=text,
                crawl_depth=0,
                crawl_timestamp=now,
                doc_id=doc_id,
                content_hash=content_hash,
            )

            object_key = url_to_object_key(url)
            storage.upload_document(object_key, document)

            metadata = PageMetadata(
                doc_id=doc_id,
                url=url,
                title=title,
                crawl_depth=0,
                crawl_timestamp=now,
                content_hash=content_hash,
                markdown_size_bytes=len(document.encode()),
                status_code=resp.status_code,
                response_headers=dict(resp.headers),
                internal_links=[],
                external_links=[],
                last_modified=resp.headers.get("last-modified"),
                etag=resp.headers.get("etag"),
            )
            storage.upload_metadata(object_key, metadata)

            stored_urls.add(url)
            stats["documents_processed"] += 1
            logger.info("Stored document: %s (%d chars)", url, len(text))

    return stats
