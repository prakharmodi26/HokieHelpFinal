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
async def test_run_crawl_accepts_any_cs_vt_edu_subdomain(crawler_config):
    """BFS results from any *.cs.vt.edu subdomain are stored directly — no rewrite."""
    mock_storage = _make_mock_storage()
    results = [
        _make_crawl_result("https://cs.vt.edu/about", "About", "# About CS", depth=0),
        _make_crawl_result("https://website.cs.vt.edu/people", "People", "# People", depth=1),
        _make_crawl_result("https://students.cs.vt.edu/portal", "Portal", "# Portal", depth=1),
        _make_crawl_result("https://wiki.cs.vt.edu/courses", "Courses", "# Courses", depth=1),
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

    assert mock_storage.upload_document.call_count == 4
    assert stats["pages_crawled"] == 4
    assert stats["pages_failed"] == 0


@pytest.mark.asyncio
async def test_run_crawl_skips_non_cs_vt_edu_domains(crawler_config):
    """Pages on non-cs.vt.edu domains are skipped."""
    mock_storage = _make_mock_storage()
    results = [
        _make_crawl_result("https://eng.vt.edu/page", "Eng", "# Eng", depth=0),
        _make_crawl_result("https://cs.vt.edu", "CS", "# CS VT", depth=0),
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

    assert stats["pages_crawled"] == 1   # only cs.vt.edu
    assert stats["pages_failed"] == 1    # eng.vt.edu skipped


@pytest.mark.asyncio
async def test_run_crawl_bfs_uses_include_external_true(crawler_config):
    """BFS strategy is configured with include_external=True so cross-subdomain links are followed."""
    mock_storage = _make_mock_storage()

    results = [
        _make_crawl_result("https://website.cs.vt.edu", "Home", "# Home", depth=0),
        _make_crawl_result("https://students.cs.vt.edu/grad", "Grad", "# Grad", depth=1),
    ]

    mock_crawler_instance = AsyncMock()
    captured_configs = []

    async def fake_arun(*args, **kwargs):
        # arun is called as arun(url=..., config=...) — capture the config
        cfg = kwargs.get("config")
        if cfg is None and len(args) > 1:
            cfg = args[1]
        captured_configs.append(cfg)
        async def _gen():
            for r in results:
                yield r
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    # Both pages stored — BFS follows cross-subdomain links
    assert stats["pages_crawled"] == 2
    # Verify include_external=True on the BFS config
    assert len(captured_configs) == 1
    bfs_cfg = captured_configs[0]
    assert bfs_cfg is not None, "BFS config was not captured"
    assert bfs_cfg.deep_crawl_strategy.include_external is True


@pytest.mark.asyncio
async def test_run_crawl_collects_external_host_documents(crawler_config):
    """Documents linked from a cs.vt.edu page are collected even if hosted elsewhere."""
    mock_storage = _make_mock_storage()

    bfs_result = _make_crawl_result(
        "https://cs.vt.edu/research",
        "Research",
        "# Research",
        depth=0,
        links={
            "internal": [],
            "external": [{"href": "https://arxiv.org/pdf/2301.00001.pdf", "text": "Paper"}],
        },
    )

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield bfs_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    collected_doc_urls: list[set] = []

    async def fake_download(doc_urls, *args, **kwargs):
        collected_doc_urls.append(set(doc_urls))
        return {"documents_processed": 0, "documents_failed": 0}

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance), \
         patch("crawler.crawl.download_and_process_documents", side_effect=fake_download):
        await run_crawl(crawler_config, mock_storage)

    assert len(collected_doc_urls) == 1
    assert "https://arxiv.org/pdf/2301.00001.pdf" in collected_doc_urls[0]


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


from crawler.crawl import _is_allowed_host


def test_is_allowed_host_exact_match():
    assert _is_allowed_host("cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_subdomain():
    assert _is_allowed_host("website.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_deep_subdomain():
    assert _is_allowed_host("students.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_sub_sub_subdomain():
    # students.website.cs.vt.edu ends with .cs.vt.edu — must be accepted
    assert _is_allowed_host("students.website.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_unrelated_domain():
    assert _is_allowed_host("eng.vt.edu", ("cs.vt.edu",)) is False


def test_is_allowed_host_partial_suffix_no_match():
    # "notcs.vt.edu" must NOT match "cs.vt.edu"
    assert _is_allowed_host("notcs.vt.edu", ("cs.vt.edu",)) is False


def test_is_allowed_host_none_returns_false():
    assert _is_allowed_host(None, ("cs.vt.edu",)) is False


def test_is_allowed_host_empty_domains_returns_false():
    assert _is_allowed_host("cs.vt.edu", ()) is False
