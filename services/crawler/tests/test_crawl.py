import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from crawler.crawl import run_crawl


def _make_crawl_result(url: str, title: str, markdown: str, depth: int = 0):
    """Create a mock CrawlResult."""
    result = MagicMock()
    result.success = True
    result.url = url
    result.metadata = {"title": title, "depth": depth}
    result.markdown = MagicMock()
    result.markdown.raw_markdown = markdown
    return result


@pytest.mark.asyncio
async def test_run_crawl_processes_results(crawler_config):
    """run_crawl uploads each successful crawl result to storage."""
    mock_storage = MagicMock()
    results = [
        _make_crawl_result("https://cs.vt.edu", "Home", "# Home Page", depth=0),
        _make_crawl_result("https://cs.vt.edu/about", "About", "# About", depth=1),
    ]

    # Mock AsyncWebCrawler as async context manager returning async iterator
    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            for r in results:
                yield r
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 2
    assert stats["pages_crawled"] == 2
    assert stats["pages_failed"] == 0
    assert stats["pages_skipped_duplicate"] == 0


@pytest.mark.asyncio
async def test_run_crawl_skips_failed_results(crawler_config):
    """run_crawl skips results where success is False."""
    mock_storage = MagicMock()

    failed_result = MagicMock()
    failed_result.success = False
    failed_result.url = "https://cs.vt.edu/broken"
    failed_result.error_message = "404 Not Found"

    ok_result = _make_crawl_result("https://cs.vt.edu", "Home", "# Home", depth=0)

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield failed_result
            yield ok_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 1
    assert stats["pages_crawled"] == 1
    assert stats["pages_failed"] == 1


@pytest.mark.asyncio
async def test_run_crawl_skips_empty_markdown(crawler_config):
    """run_crawl skips results with empty markdown content."""
    mock_storage = MagicMock()

    empty_result = _make_crawl_result("https://cs.vt.edu/empty", "Empty", "", depth=0)

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield empty_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 0
    assert stats["pages_crawled"] == 0
    assert stats["pages_failed"] == 1


@pytest.mark.asyncio
async def test_run_crawl_deduplicates_identical_content(crawler_config):
    """run_crawl skips pages with identical markdown content (e.g. redirect mirrors)."""
    mock_storage = MagicMock()

    same_content = "# Same Page\n\nIdentical body content."
    results = [
        _make_crawl_result("https://cs.vt.edu", "Home", same_content, depth=0),
        _make_crawl_result("https://website.cs.vt.edu/index.html", "Home", same_content, depth=1),
    ]

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            for r in results:
                yield r
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 1
    assert stats["pages_crawled"] == 1
    assert stats["pages_skipped_duplicate"] == 1
