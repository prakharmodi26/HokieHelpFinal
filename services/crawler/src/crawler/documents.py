"""Download and extract text from PDF and Word documents found during crawl."""

from __future__ import annotations

import asyncio
import hashlib
import io
import ipaddress
import logging
import re
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.urls import safe_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.scope import is_vt_affiliated
from crawler.storage import MinioStorage, StorageFullError
from crawler.urls import is_ingestible_url
from crawler.validate import validate_document

logger = logging.getLogger(__name__)

# File extensions we treat as documents
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
ALLOWED_SCHEMES = {"http", "https"}


def is_document_url(url: str) -> bool:
    """Return True if the URL points to a downloadable document."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ALLOWED_SCHEMES:
        return False
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def collect_document_links(
    links: dict | None,
    *,
    affiliated_matcher=None,
    host_filter=None,
) -> set[str]:
    """Extract VT-affiliated document URLs from a CrawlResult's links dict.

    Only documents whose host is VT-affiliated (vt.edu registered domain OR
    present in *affiliated_matcher*) are collected.  URLs on blocklisted hosts
    (per *host_filter*) and URLs matching a host's path_exclude prefixes are
    dropped.

    When *affiliated_matcher* is None the affiliation check degrades to
    vt.edu-only — a missing allowlist file must never open the floodgate to
    arbitrary external documents.
    """
    urls: set[str] = set()
    if not links:
        return urls
    for category in ("internal", "external"):
        for link in links.get(category, []):
            href = link.get("href", "")
            if not href:
                continue
            if not is_ingestible_url(href):
                continue
            if not is_document_url(href):
                continue
            # Gate 1: affiliation — drop non-VT-affiliated docs.
            if not is_vt_affiliated(href, affiliated_matcher):
                continue
            # Gate 2: blocklist — drop hosts/paths the HostFilter rejects.
            if host_filter is not None and host_filter.is_blocked_url(href):
                continue
            # Gate 3: per-host path_exclude prefixes from the allowlist.
            if affiliated_matcher is not None:
                _parsed = urlparse(href)
                _host = (_parsed.hostname or "").lower()
                _path = _parsed.path
                excludes = affiliated_matcher.path_excludes_for(_host)
                if any(_path.startswith(pfx) for pfx in excludes):
                    continue
            urls.add(href)
    return urls


def _extract_pdf_text_legacy(
    data: bytes,
    *,
    max_pages: int = 500,
    max_chars: int = 2_000_000,
) -> str:
    """Extract plain text from PDF bytes using pymupdf.get_text().

    Legacy fallback for the markdown-producing _extract_pdf_text path.
    Limits extraction to *max_pages* pages and *max_chars* total characters.
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
        page_count = min(len(doc), max_pages)
        if len(doc) > max_pages:
            logger.warning(
                "PDF has %d pages, limiting extraction to %d", len(doc), max_pages
            )
        for i in range(page_count):
            page_text = doc[i].get_text()
            text_parts.append(page_text)
            total_chars += len(page_text)
            if total_chars >= max_chars:
                logger.warning("Extracted text exceeds %d chars, truncating", max_chars)
                break
        doc.close()
    except Exception as exc:
        logger.error("Failed to extract PDF text: %s", exc)
        return ""

    result = "\n\n".join(text_parts).strip()
    return result[:max_chars]


def _extract_pdf_text(
    data: bytes,
    *,
    max_pages: int = 500,
    max_chars: int = 2_000_000,
) -> str:
    """Extract structured markdown from PDF bytes using pymupdf4llm.

    Falls back to the legacy plain-text path on ANY exception (including
    ImportError), so a malformed PDF never raises out of this function.
    Limits extraction to *max_pages* pages and *max_chars* total characters.
    """
    try:
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(stream=data, filetype="pdf")
        page_count = min(len(doc), max_pages)
        if len(doc) > max_pages:
            logger.warning(
                "PDF has %d pages, limiting extraction to %d", len(doc), max_pages
            )
        md = pymupdf4llm.to_markdown(
            doc,
            pages=list(range(page_count)),
            ignore_images=True,
            show_progress=False,
        )
        doc.close()
        return md.strip()[:max_chars]
    except Exception as exc:
        logger.warning(
            "pymupdf4llm extraction failed (%s); falling back to legacy path", exc
        )
        return _extract_pdf_text_legacy(data, max_pages=max_pages, max_chars=max_chars)


def _extract_docx_text_legacy(
    data: bytes,
    *,
    max_chars: int = 2_000_000,
) -> str:
    """Extract plain text from Word .docx bytes using python-docx paragraphs.

    Legacy fallback for the markdown-producing _extract_docx_text path.
    Note: this path drops tables. Limits extraction to *max_chars* characters.
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
                if total_chars >= max_chars:
                    logger.warning("Extracted text exceeds %d chars, truncating", max_chars)
                    break
        return "\n\n".join(parts)[:max_chars]
    except Exception as exc:
        logger.error("Failed to extract Word text: %s", exc)
        return ""


def _extract_docx_text(
    data: bytes,
    *,
    max_chars: int = 2_000_000,
) -> str:
    """Extract structured markdown from Word .docx bytes via mammoth + markdownify.

    mammoth converts the .docx to HTML (preserving tables, lists, headings),
    then markdownify flattens to ATX-style markdown. Falls back to the legacy
    paragraph-only path on ANY exception (including ImportError).
    """
    try:
        import mammoth
        from markdownify import markdownify as _md, ATX

        result = mammoth.convert_to_html(io.BytesIO(data))
        html = result.value or ""
        markdown = _md(html, heading_style=ATX)
        return markdown.strip()[:max_chars]
    except Exception as exc:
        logger.warning(
            "mammoth extraction failed (%s); falling back to legacy path", exc
        )
        return _extract_docx_text_legacy(data, max_chars=max_chars)


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a document URL."""
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    # Remove extension and replace separators with spaces
    name = re.sub(r"\.[^.]+$", "", filename)
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def _is_private_or_reserved_ip(host: str) -> bool:
    """Return True if *host* is an IP literal in a private/reserved range.

    Blocks SSRF vectors (e.g. http://10.0.0.1/secret.pdf,
    http://169.254.169.254/latest/meta-data). Hostnames are not resolved —
    the affiliation re-check in _validate_document_url covers that vector.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_document_url(url: str, *, affiliated_matcher=None) -> bool:
    """Validate that a document URL is safe to fetch.

    Defense-in-depth: even though collect_document_links already gates on
    affiliation and blocklist, this re-checks both plus rejects IP literals
    in private/reserved ranges (SSRF defense).
    """
    if not is_ingestible_url(url):
        logger.warning("Rejected non-ingestible document URL: %s", url)
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        logger.warning("Rejected document URL with scheme %r: %s", parsed.scheme, url)
        return False
    host = parsed.hostname
    if not host:
        logger.warning("Rejected document URL with no hostname: %s", url)
        return False
    if _is_private_or_reserved_ip(host):
        logger.warning("Rejected document URL with private/reserved IP: %s", url)
        return False
    if not is_vt_affiliated(url, affiliated_matcher):
        logger.warning("Rejected non-VT-affiliated document URL: %s", url)
        return False
    return True


async def download_and_process_documents(
    document_urls: set[str],
    storage: MinioStorage,
    request_delay: float = 0.5,
    target_bucket: str | None = None,
    document_bucket: str | None = None,
    max_document_size_bytes: int = 50 * 1024 * 1024,
    max_pdf_pages: int = 500,
    max_extracted_text_chars: int = 2_000_000,
    affiliated_matcher=None,
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Download documents, extract text, and store as markdown in MinIO.

    `target_bucket` selects the destination bucket (None = storage default);
    production passes the cleaned bucket so documents reach the chunker.
    Returns stats dict with documents_processed and documents_failed counts.
    """
    stats = {"documents_processed": 0, "documents_failed": 0}

    if not document_urls:
        return stats

    logger.info("Phase 3: processing %d document URLs", len(document_urls))

    # Within-run URL dedup only: if the same PDF is linked from two pages in
    # this crawl, process it once. No cross-run state — each run re-downloads
    # and overwrites at deterministic MinIO keys.
    seen_urls: set[str] = set()

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "HokieHelp-Crawler/1.0"},
    ) as client:
        for url in document_urls:
            # Graceful shutdown: stop fetching new documents when SIGTERM/SIGINT
            # was received. The in-flight download (if any) completes first.
            if shutdown_event is not None and shutdown_event.is_set():
                logger.warning(
                    "Shutdown requested — draining documents pass "
                    "(%d processed, %d failed)",
                    stats["documents_processed"], stats["documents_failed"],
                )
                break

            if url in seen_urls:
                logger.info("Within-run duplicate document URL skipped: %s", url)
                continue
            seen_urls.add(url)

            if not _validate_document_url(url, affiliated_matcher=affiliated_matcher):
                stats["documents_failed"] += 1
                continue

            # Legacy binary .doc (Word 97-2003) can't be converted to markdown,
            # so skip it BEFORE downloading — otherwise the fetch is wasted and
            # the raw binary is stored to the document bucket before the dispatch
            # discovers it can't convert, leaving an orphan binary with no
            # markdown (BUG-E). .docx / .pdf continue to download + convert.
            if urlparse(url).path.lower().endswith(".doc"):
                logger.info("Legacy .doc format not supported, skipping: %s", url)
                stats["documents_failed"] += 1
                continue

            # Politeness: space out document fetches (the httpx path has no
            # adaptive limiter, unlike the render path). Applied only to URLs
            # that pass validation, so we never sleep on rejects.
            if request_delay:
                await asyncio.sleep(request_delay)

            try:
                # HEAD request first to check size — short timeout: a slow
                # HEAD server shouldn't block the whole documents pass.
                head_resp = await client.head(url, timeout=10.0)
                content_length = head_resp.headers.get("content-length")
                if content_length and int(content_length) > max_document_size_bytes:
                    logger.warning(
                        "Document too large (%s bytes), skipping: %s",
                        content_length, url,
                    )
                    stats["documents_failed"] += 1
                    continue
            except Exception:
                pass  # HEAD may not be supported; proceed with GET

            # Stream the body with a hard cap so a document that is oversized or
            # advertises no Content-Length can never buffer the whole body into
            # memory (OOM). httpx follows redirects, so the FINAL host is
            # re-validated here (SSRF defense-in-depth): a VT URL can 30x-redirect
            # to an internal IP or an off-affiliation host, which the pre-request
            # check on the original URL alone would miss.
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    final_url = str(resp.url)
                    if final_url != url:
                        final_host = (urlparse(final_url).hostname or "")
                        if _is_private_or_reserved_ip(final_host) or not is_vt_affiliated(
                            final_url, affiliated_matcher
                        ):
                            logger.warning(
                                "Document redirected to a disallowed URL, skipping: "
                                "%s -> %s", url, final_url,
                            )
                            stats["documents_failed"] += 1
                            continue
                    chunks: list[bytes] = []
                    total = 0
                    oversized = False
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > max_document_size_bytes:
                            logger.warning(
                                "Downloaded document exceeds %d bytes, skipping: %s",
                                max_document_size_bytes, url,
                            )
                            oversized = True
                            break
                        chunks.append(chunk)
                    if oversized:
                        stats["documents_failed"] += 1
                        continue
                    data = b"".join(chunks)
                    # Magic byte validation: catch HTML 404s served as .pdf, ZIP bombs
                    # served as .docx, etc. Skips storage and extraction on mismatch.
                    doc_result = validate_document(data, url)
                    if not doc_result.valid:
                        logger.warning("Rejected document (%s): %s", doc_result.reason, url)
                        stats["documents_failed"] += 1
                        continue

                    status_code = resp.status_code
                    resp_headers = dict(resp.headers)
            except Exception as exc:
                logger.warning("Failed to download document %s: %s", url, exc)
                stats["documents_failed"] += 1
                continue

            path = urlparse(url).path.lower()

            # Save the raw binary to the document bucket before conversion.
            if document_bucket:
                try:
                    raw_key = safe_object_key(url, suffix="")
                    content_type = (
                        "application/pdf" if path.endswith(".pdf")
                        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    storage.upload_raw_bytes(raw_key, data, content_type, bucket=document_bucket)
                except StorageFullError:
                    raise  # fatal: abort the run (issue #5)
                except Exception as exc:
                    logger.warning("Failed to store raw document %s: %s", url, exc)

            if path.endswith(".pdf"):
                text = _extract_pdf_text(
                    data, max_pages=max_pdf_pages, max_chars=max_extracted_text_chars
                )
            elif path.endswith(".docx"):
                text = _extract_docx_text(data, max_chars=max_extracted_text_chars)
            else:
                # .doc is skipped pre-download above; any other extension can't
                # reach here (the frontier only routes .pdf/.doc/.docx), so this
                # is a defensive skip, not a live path.
                stats["documents_failed"] += 1
                continue

            if not text.strip():
                logger.warning("No text extracted from document: %s", url)
                stats["documents_failed"] += 1
                continue

            content_hash = hashlib.sha256(text.encode()).hexdigest()

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

            # Store defensively: a single document whose key/upload raises
            # (MinIO-illegal key, transient storage error) fails just this URL,
            # never the whole documents pass.
            try:
                object_key = url_to_object_key(url)
                storage.upload_document(object_key, document, bucket=target_bucket)

                metadata = PageMetadata(
                    doc_id=doc_id,
                    url=url,
                    title=title,
                    crawl_depth=0,
                    crawl_timestamp=now,
                    content_hash=content_hash,
                    markdown_size_bytes=len(document.encode()),
                    status_code=status_code,
                    response_headers=resp_headers,
                    internal_links=[],
                    external_links=[],
                    last_modified=resp_headers.get("last-modified"),
                    etag=resp_headers.get("etag"),
                    # Provenance: documents reach this pass only when classify_url
                    # assigned them Scope.DOCUMENT (VT-affiliated host).  Record
                    # the true scope so downstream consumers can distinguish doc
                    # objects from page objects without re-inspecting the URL.
                    scope="document",
                    is_external=False,
                )
                storage.upload_metadata(object_key, metadata, bucket=target_bucket)
            except StorageFullError:
                raise  # fatal: abort the run (issue #5)
            except Exception as exc:
                logger.warning("Failed to store document %s: %s", url, exc)
                stats["documents_failed"] += 1
                continue

            stats["documents_processed"] += 1
            logger.info("Stored document: %s (%d chars)", url, len(text))

    return stats
