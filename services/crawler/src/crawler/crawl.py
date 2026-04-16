"""Deep crawl orchestration using Crawl4AI."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from crawler.config import CrawlerConfig
from crawler.documents import collect_document_links, download_and_process_documents
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.storage import MinioStorage

logger = logging.getLogger(__name__)


@dataclass
class VisitEntry:
    url: str
    status: str
    reason: str = ""
    depth: int = 0
    timestamp: str = ""


@dataclass
class VisitLog:
    entries: list[VisitEntry] = field(default_factory=list)

    def add(self, url: str, status: str, reason: str = "", depth: int = 0) -> None:
        self.entries.append(VisitEntry(
            url=url, status=status, reason=reason, depth=depth,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        ))

    def render(self) -> str:
        lines = ["# Websites Visited Log", ""]
        for e in self.entries:
            detail = f" — {e.reason}" if e.reason else ""
            lines.append(f"[{e.timestamp}] [{e.status:>10}] (depth={e.depth}) {e.url}{detail}")
        lines.append("")
        lines.append(self._summary())
        return "\n".join(lines)

    def _summary(self) -> str:
        total = len(self.entries)
        status_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for e in self.entries:
            status_counts[e.status] = status_counts.get(e.status, 0) + 1
            host = urlparse(e.url).hostname or "unknown"
            domain_counts[host] = domain_counts.get(host, 0) + 1
        parts = [
            "## Summary", "",
            f"Total URLs encountered: {total}",
            f"Total domains discovered: {len(domain_counts)}", "",
            "### By Status", "",
        ]
        for status, count in sorted(status_counts.items()):
            parts.append(f"  {status}: {count}")
        parts.extend(["", "### By Domain", ""])
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            parts.append(f"  {domain}: {count}")
        return "\n".join(parts)


def _is_allowed_host(host: str | None, allowed_domains: tuple[str, ...]) -> bool:
    """Return True if host exactly matches or is a subdomain of any allowed domain.

    Works at any nesting depth: students.website.cs.vt.edu matches cs.vt.edu
    because the string ends with '.cs.vt.edu'.
    """
    if not host:
        return False
    for domain in allowed_domains:
        if host == domain or host.endswith("." + domain):
            return True
    return False


def _is_blocked_path(url: str, blocked_paths: tuple[str, ...]) -> bool:
    """Return True if the URL path starts with any blocked prefix."""
    path = urlparse(url).path
    return any(path.startswith(prefix) for prefix in blocked_paths)


def _store_result(
    result,
    storage: MinioStorage,
    stats: dict,
    seen_content_hashes: set[str],
    stored_urls: set[str],
    visit_log: VisitLog,
    depth: int = 0,
) -> None:
    """Store a successful crawl result; update stats in-place."""
    title = result.metadata.get("title") if result.metadata else None
    now = datetime.now(timezone.utc)

    if result.markdown is None or not result.markdown.raw_markdown:
        logger.warning("Empty or missing markdown for %s, skipping", result.url)
        stats["pages_failed"] += 1
        visit_log.add(result.url, "EMPTY", "empty or missing markdown", depth)
        return
    markdown_content = result.markdown.raw_markdown

    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()
    if content_hash in seen_content_hashes:
        logger.info("Duplicate/unchanged content for %s, skipping", result.url)
        stats["pages_skipped_duplicate"] += 1
        visit_log.add(result.url, "DUPLICATE", "content hash already seen", depth)
        return
    seen_content_hashes.add(content_hash)

    # Build metadata from all available CrawlResult fields
    response_headers = getattr(result, "response_headers", None) or {}
    doc_id = doc_id_for_url(result.url)
    metadata = PageMetadata(
        doc_id=doc_id,
        url=result.url,
        title=title,
        crawl_depth=depth,
        crawl_timestamp=now,
        content_hash=content_hash,
        markdown_size_bytes=len(markdown_content.encode()),
        status_code=getattr(result, "status_code", None),
        response_headers=response_headers if response_headers else None,
        internal_links=[
            lnk["href"] for lnk in (result.links or {}).get("internal", [])
            if lnk.get("href")
        ],
        external_links=[
            lnk["href"] for lnk in (result.links or {}).get("external", [])
            if lnk.get("href")
        ],
        last_modified=response_headers.get("last-modified"),
        etag=response_headers.get("etag"),
    )

    document = build_markdown_document(
        url=result.url,
        title=title,
        markdown_content=markdown_content,
        crawl_depth=depth,
        crawl_timestamp=now,
        doc_id=doc_id,
        content_hash=content_hash,
    )
    object_key = url_to_object_key(result.url)
    storage.upload_document(object_key, document)
    storage.upload_metadata(object_key, metadata)
    stored_urls.add(result.url)
    stats["pages_crawled"] += 1
    logger.info("Stored page %d: %s (depth %d)", stats["pages_crawled"], result.url, depth)
    visit_log.add(result.url, "SAVED", f"stored as {object_key}", depth)


async def run_crawl(config: CrawlerConfig, storage: MinioStorage) -> dict:
    """Execute a deep crawl and upload each page to storage.

    Returns a stats dict with pages_crawled, pages_failed, pages_skipped_duplicate counts.

    BFS crawl with include_external=True seeded from config.seed_url. DomainFilter
    restricts BFS to config.allowed_domains and their subdomains at any nesting depth,
    so cross-subdomain links (e.g. website.cs.vt.edu → students.cs.vt.edu) are followed
    automatically while links to unrelated domains (google.com, etc.) are skipped by
    the filter before fetching.

    Documents (PDF, Word) linked from any crawled page are downloaded regardless of the
    document host — the trust anchor is the cs.vt.edu page that linked them.
    """
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=list(config.allowed_domains),
            blocked_domains=list(config.blocked_domains),
        ),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=config.max_depth,
        include_external=True,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
    )

    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )

    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=config.request_delay,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        page_timeout=30000,
        semaphore_count=5,
        markdown_generator=DefaultMarkdownGenerator(),
    )

    stats = {"pages_crawled": 0, "pages_failed": 0, "pages_skipped_duplicate": 0,
             "documents_processed": 0, "documents_failed": 0}
    seen_content_hashes: set[str] = set(storage.load_all_content_hashes().keys())
    stored_urls: set[str] = set()
    document_urls: set[str] = set()
    visit_log = VisitLog()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun(url=config.seed_url, config=bfs_config):
            depth = result.metadata.get("depth", 0) if result.metadata else 0

            if not result.success:
                error_msg = getattr(result, "error_message", "unknown error")
                logger.warning("Failed to crawl %s: %s", result.url, error_msg)
                stats["pages_failed"] += 1
                visit_log.add(result.url, "FAILED", error_msg, depth)
                continue

            parsed_url = urlparse(result.url)
            if parsed_url.scheme not in ("http", "https"):
                logger.warning("Skipping non-HTTP URL: %s", result.url)
                stats["pages_failed"] += 1
                visit_log.add(result.url, "SKIPPED", "non-HTTP scheme", depth)
                continue

            if _is_blocked_path(result.url, config.blocked_paths):
                logger.debug("Blocked path, skipping: %s", result.url)
                stats["pages_failed"] += 1
                visit_log.add(result.url, "BLOCKED", "blocked path", depth)
                continue

            final_host = parsed_url.hostname

            if not _is_allowed_host(final_host, config.allowed_domains):
                logger.info(
                    "Skipping %s — host %s not in allowed domains",
                    result.url,
                    final_host,
                )
                stats["pages_failed"] += 1
                visit_log.add(result.url, "BLOCKED", f"host {final_host} not in allowed domains", depth)
                continue

            _store_result(result, storage, stats, seen_content_hashes, stored_urls, visit_log, depth)

            document_urls.update(collect_document_links(result.links))

    # Download and process documents (PDFs, Word docs) from any host
    if document_urls:
        doc_stats = await download_and_process_documents(
            document_urls, storage, seen_content_hashes, stored_urls,
            request_delay=config.request_delay,
        )
        stats["documents_processed"] = doc_stats["documents_processed"]
        stats["documents_failed"] = doc_stats["documents_failed"]

    logger.info(
        "Deduplication: %d duplicate pages skipped",
        stats["pages_skipped_duplicate"],
    )
    logger.info(
        "Documents: %d processed, %d failed",
        stats["documents_processed"],
        stats["documents_failed"],
    )
    return stats, visit_log
