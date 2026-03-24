import pytest
import hashlib
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from crawler.crawl import run_crawl


def _make_crawl_result(
    url: str,
    title: str,
    markdown: str,
    depth: int = 0,
    links: dict | None = None,
    status_code: int = 200,
    response_headers: dict | None = None,
):
    """Create a mock CrawlResult with all fields used by _store_result."""
    result = MagicMock()
    result.success = True
    result.url = url
    result.status_code = status_code
    result.response_headers = response_headers or {"content-type": "text/html"}
    result.metadata = {"title": title, "depth": depth}
    result.markdown = MagicMock()
    result.markdown.fit_markdown = markdown
    result.markdown.raw_markdown = markdown
    result.links = links or {"internal": [], "external": []}
    return result


def _make_mock_storage():
    """Create a mock storage with load_all_content_hashes returning empty."""
    mock = MagicMock()
    mock.load_all_content_hashes.return_value = {}
    return mock


@pytest.mark.asyncio
async def test_run_crawl_processes_results(crawler_config):
    """run_crawl uploads each successful crawl result to storage."""
    mock_storage = _make_mock_storage()
    results = [
        _make_crawl_result("https://website.cs.vt.edu", "Home", "# Home Page", depth=0),
        _make_crawl_result("https://website.cs.vt.edu/about", "About", "# About", depth=1),
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

    assert mock_storage.upload_document.call_count == 2
    assert mock_storage.upload_metadata.call_count == 2
    assert stats["pages_crawled"] == 2
    assert stats["pages_failed"] == 0
    assert stats["pages_skipped_duplicate"] == 0


@pytest.mark.asyncio
async def test_run_crawl_skips_failed_results(crawler_config):
    """run_crawl skips results where success is False."""
    mock_storage = _make_mock_storage()

    failed_result = MagicMock()
    failed_result.success = False
    failed_result.url = "https://website.cs.vt.edu/broken"
    failed_result.error_message = "404 Not Found"

    ok_result = _make_crawl_result("https://website.cs.vt.edu", "Home", "# Home", depth=0)

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
    mock_storage = _make_mock_storage()

    empty_result = _make_crawl_result("https://website.cs.vt.edu/empty", "Empty", "", depth=0)

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
    mock_storage = _make_mock_storage()

    same_content = "# Same Page\n\nIdentical body content."
    results = [
        _make_crawl_result("https://website.cs.vt.edu", "Home", same_content, depth=0),
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


@pytest.mark.asyncio
async def test_run_crawl_rewrites_cs_vt_edu_redirect_and_fetches(crawler_config):
    """When BFS lands on cs.vt.edu, the path is rewritten to website.cs.vt.edu and re-fetched."""
    mock_storage = _make_mock_storage()

    bfs_cs_result = _make_crawl_result("https://cs.vt.edu/about", "About", "# About CS", depth=1)
    ok_result = _make_crawl_result("https://website.cs.vt.edu/people", "People", "# People", depth=1)
    phase2_result = _make_crawl_result("https://website.cs.vt.edu/about", "About VT", "# About VT CS", depth=0)

    mock_crawler_instance = AsyncMock()
    call_count = {"n": 0}

    async def fake_arun(url, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            async def _gen():
                yield bfs_cs_result
                yield ok_result
            return _gen()
        else:
            return phase2_result

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 2
    assert stats["pages_crawled"] == 2
    assert stats["pages_failed"] == 1


@pytest.mark.asyncio
async def test_run_crawl_queues_cs_subdomain_external_links(crawler_config):
    """External links to *.cs.vt.edu subdomains are queued and fetched in phase 2."""
    mock_storage = _make_mock_storage()

    bfs_result = _make_crawl_result(
        "https://website.cs.vt.edu",
        "Home",
        "# Home",
        depth=0,
        links={"external": [{"href": "https://people.cs.vt.edu", "text": "People"}]},
    )
    subdomain_result = _make_crawl_result("https://cs.vt.edu/people", "People", "# People CS", depth=0)
    rewritten_result = _make_crawl_result("https://website.cs.vt.edu/people", "People VT", "# People VT", depth=0)

    mock_crawler_instance = AsyncMock()
    call_count = {"n": 0}

    async def fake_arun(url, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            async def _gen():
                yield bfs_result
            return _gen()
        elif call_count["n"] == 2:
            return subdomain_result
        else:
            return rewritten_result

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 2
    assert stats["pages_crawled"] == 2


# --- New metadata and recrawl tests ---

@pytest.mark.asyncio
async def test_run_crawl_uploads_metadata_sidecar(crawler_config):
    """Each stored page gets a .meta.json sidecar via upload_metadata."""
    mock_storage = _make_mock_storage()

    results = [_make_crawl_result("https://website.cs.vt.edu", "Home", "# Home", depth=0)]
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
    assert mock_storage.upload_metadata.call_count == 1
    assert stats["pages_crawled"] == 1


@pytest.mark.asyncio
async def test_run_crawl_skips_unchanged_pages_via_stored_hash(crawler_config):
    """Pages whose content hash was already stored are skipped (incremental recrawl)."""
    same_content = "# Stable\n\nNo changes."
    existing_hash = hashlib.sha256(same_content.encode()).hexdigest()

    mock_storage = MagicMock()
    mock_storage.load_all_content_hashes.return_value = {existing_hash: "https://website.cs.vt.edu"}

    results = [_make_crawl_result("https://website.cs.vt.edu", "Home", same_content, depth=0)]
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

    assert mock_storage.upload_document.call_count == 0
    assert stats["pages_skipped_duplicate"] == 1


@pytest.mark.asyncio
async def test_run_crawl_updates_changed_page(crawler_config):
    """A page whose content changed since last crawl is re-stored."""
    old_hash = "a" * 64
    mock_storage = MagicMock()
    mock_storage.load_all_content_hashes.return_value = {old_hash: "https://website.cs.vt.edu"}

    results = [_make_crawl_result("https://website.cs.vt.edu", "Home", "# Updated!", depth=0)]
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
    assert mock_storage.upload_metadata.call_count == 1
    assert stats["pages_crawled"] == 1


@pytest.mark.asyncio
async def test_run_crawl_metadata_fields_populated(crawler_config):
    """PageMetadata passed to upload_metadata has all fields from CrawlResult."""
    mock_storage = _make_mock_storage()

    results = [_make_crawl_result(
        "https://website.cs.vt.edu/about", "About", "# About", depth=2,
        status_code=200,
        response_headers={"last-modified": "Mon, 16 Mar 2026 10:00:00 GMT", "etag": '"abc"'},
        links={
            "internal": [{"href": "https://website.cs.vt.edu/people"}],
            "external": [{"href": "https://eng.vt.edu"}],
        },
    )]
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
        await run_crawl(crawler_config, mock_storage)

    from crawler.metadata import PageMetadata
    _, meta = mock_storage.upload_metadata.call_args[0]
    assert isinstance(meta, PageMetadata)
    assert meta.url == "https://website.cs.vt.edu/about"
    assert meta.crawl_depth == 2
    assert meta.status_code == 200
    assert meta.last_modified == "Mon, 16 Mar 2026 10:00:00 GMT"
    assert meta.etag == '"abc"'
    assert "https://website.cs.vt.edu/people" in meta.internal_links
    assert "https://eng.vt.edu" in meta.external_links
    assert len(meta.doc_id) == 16
    assert len(meta.content_hash) == 64


from crawler.crawl import _is_blocked_path


def test_blocked_path_matches_content_prefix():
    blocked = ("/content/", "/editor.html")
    assert _is_blocked_path("https://website.cs.vt.edu/content/tags/foo", blocked) is True


def test_blocked_path_matches_editor_page():
    blocked = ("/content/", "/editor.html")
    assert _is_blocked_path("https://website.cs.vt.edu/editor.html/content/foo", blocked) is True


def test_blocked_path_matches_cs_root():
    blocked = ("/cs-root.html",)
    assert _is_blocked_path("https://website.cs.vt.edu/cs-root.html", blocked) is True


def test_blocked_path_allows_faculty_page():
    blocked = ("/content/", "/editor.html", "/cs-root.html", "/cs-source.html")
    assert _is_blocked_path("https://website.cs.vt.edu/people/faculty/denis-gracanin.html", blocked) is False


def test_blocked_path_allows_seminar_page():
    blocked = ("/content/", "/editor.html", "/cs-root.html", "/cs-source.html")
    assert _is_blocked_path("https://website.cs.vt.edu/research/Seminars/Ali_Butt.html", blocked) is False


def test_blocked_path_empty_blocks_nothing():
    assert _is_blocked_path("https://website.cs.vt.edu/content/anything", ()) is False
