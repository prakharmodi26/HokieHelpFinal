# services/crawler/tests/test_engine_loop.py
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from crawler.engine import run_engine
from crawler.frontier import Frontier


def _result(url, fit):
    md = SimpleNamespace(fit_markdown=fit, raw_markdown=fit)
    return SimpleNamespace(url=url, success=True, markdown=md,
                           metadata={"title": "T"},
                           links={"internal": [], "external": []},
                           status_code=200, response_headers={}, error_message=None)


def test_run_engine_processes_until_frontier_drains(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    fr.enqueue("https://cs.vt.edu/a", scope="vt_page", depth=0,
               hop_history=["https://cs.vt.edu/a"], discovery_source="seed")
    storage = MagicMock()

    async def fake_arun_many(urls, config, dispatcher):
        async def gen():
            for u in urls:
                yield _result(u, "# A\n" + "word " * 40)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        stats = asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=2, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
        ))
    assert stats["pages_stored"] >= 1
    assert fr.count_by_status().get("pending", 0) == 0


def test_run_engine_respects_max_urls_per_host(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    fr.enqueue("https://cs.vt.edu/", scope="vt_page", depth=0,
               hop_history=["https://cs.vt.edu/"], discovery_source="registry")
    storage = MagicMock()

    # Root page links to 5 children on the same host.
    def _root_result(url):
        md = SimpleNamespace(fit_markdown="# A\n" + "word " * 40,
                             raw_markdown="# A\n" + "word " * 40)
        return SimpleNamespace(
            url=url, success=True, markdown=md, metadata={"title": "T"},
            links={"internal": [
                {"href": f"https://cs.vt.edu/p{i}"} for i in range(5)
            ], "external": []},
            status_code=200, response_headers={}, error_message=None)

    async def fake_arun_many(urls, config, dispatcher):
        async def gen():
            for u in urls:
                if u == "https://cs.vt.edu/":
                    yield _root_result(u)
                else:
                    yield _result(u, "# C\n" + "page " * 40 + u)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=3, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
            max_urls_per_host=3,
        ))
    # cap of 3 per host: the root + at most 2 enqueued children
    total_rows = sum(fr.count_by_status().values())
    assert total_rows == 3


def _catalog_result(url):
    md = SimpleNamespace(fit_markdown="# Repo\n" + "word " * 40,
                         raw_markdown="# Repo\n" + "word " * 40)
    return SimpleNamespace(
        url=url, success=True, markdown=md, metadata={"title": "Repo"},
        html=('<html><head><meta name="citation_title" content="Record One">'
              "</head><body>body</body></html>"),
        links={"internal": [{"href": f"{url}items/1"},
                            {"href": f"{url}items/2"}],
               "external": []},
        status_code=200, response_headers={}, error_message=None)


def test_run_engine_routes_catalog_host_to_card_and_skips_expansion(tmp_path):
    """A catalog-mode host produces a catalog_card doc and enqueues zero
    children even though the page has links; a full-mode host in the same
    batch behaves exactly as before (full ingest + BFS expansion)."""
    fr = Frontier(str(tmp_path / "f.db"))
    fr.enqueue("https://vtechworks.lib.vt.edu/", scope="vt_page", depth=0,
               hop_history=["https://vtechworks.lib.vt.edu/"],
               discovery_source="registry")
    fr.enqueue("https://cs.vt.edu/", scope="vt_page", depth=0,
               hop_history=["https://cs.vt.edu/"], discovery_source="registry")
    storage = MagicMock()

    def _full_result(url):
        r = _result(url, "# Full\n" + "word " * 40)
        if url == "https://cs.vt.edu/":
            r.links = {"internal": [{"href": "https://cs.vt.edu/child"}],
                       "external": []}
        return r

    async def fake_arun_many(urls, config, dispatcher):
        async def gen():
            for u in urls:
                if "vtechworks" in u:
                    yield _catalog_result(u)
                else:
                    yield _full_result(u)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=3, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
            modes={"vtechworks.lib.vt.edu": "catalog", "cs.vt.edu": "full"},
        ))

    # catalog host: card doc stored, none of its 2 links enqueued
    uploads = {args[0][0]: args[0][1]
               for args in storage.upload_document.call_args_list}
    card = uploads["vtechworks.lib.vt.edu/index.md"]
    assert "page_type: 'catalog_card'" in card
    assert "Record One" in card
    # full host: normal page + its child enqueued; catalog children absent
    assert "page_type: 'catalog_card'" not in uploads["cs.vt.edu/index.md"]
    urls = {row[0] for status in ("done", "pending")
            for row in fr.iter_status(status)}
    assert "https://cs.vt.edu/child" in urls
    assert not any("items" in u for u in urls)
    # 2 roots done + 1 full-mode child pending = 3 rows total
    assert sum(fr.count_by_status().values()) == 3


def test_run_engine_recovers_from_wedged_batch(tmp_path):
    """A single URL whose render never returns must not wedge the whole crawl.

    Reproduces the production hang (cpes.vt.edu/library/download/* and SPA login
    pages stayed 'leased' for 13h while the engine blocked forever on the
    arun_many stream). The idle watchdog must abort the stuck batch, mark the
    unreturned URL failed (so it is not re-leased and re-wedged), recycle the
    browser, and drain the rest of the frontier.
    """
    fr = Frontier(str(tmp_path / "f.db"))
    # download URL leased last: the good pages stream out first (as they do in
    # production, by completion order), then the download wedges the stream.
    for u in ("https://cs.vt.edu/ok1", "https://cs.vt.edu/ok2",
              "https://cpes.vt.edu/library/download/1"):
        fr.enqueue(u, scope="vt_page", depth=1, hop_history=[u],
                   discovery_source="seed")
    storage = MagicMock()

    async def fake_arun_many(urls, config, dispatcher):
        async def gen():
            for u in urls:
                if "download" in u:
                    await asyncio.sleep(30)  # never returns within idle timeout
                yield _result(u, "# OK\n" + "word " * 40)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        stats = asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=1, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
            batch_idle_timeout=0.3,
        ))

    counts = fr.count_by_status()
    assert counts.get("pending", 0) == 0   # drained, did not hang
    assert counts.get("leased", 0) == 0    # nothing left stuck in-flight
    assert counts.get("done", 0) == 2      # the two good pages still ingested
    assert counts.get("failed", 0) == 1    # only the wedged download URL
    failed = {row[0] for row in fr.iter_status("failed")}
    assert failed == {"https://cpes.vt.edu/library/download/1"}


def test_run_engine_skips_special_courseleaf_host_without_rendering(tmp_path):
    """special-mode hosts (courseleaf_catalog) are checkpointed as skipped and
    never rendered/full-text ingested — Task 8.4 adds their dedicated pass.
    (dining_tool hosts are never seeded, so the engine never sees them.)"""
    fr = Frontier(str(tmp_path / "f.db"))
    fr.enqueue("https://catalog.vt.edu/", scope="vt_page", depth=0,
               hop_history=["https://catalog.vt.edu/"],
               discovery_source="registry")
    storage = MagicMock()
    rendered_urls = []

    async def fake_arun_many(urls, config, dispatcher):
        rendered_urls.extend(urls)

        async def gen():
            for u in urls:
                yield _result(u, "# X\n" + "word " * 40)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=3, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
            modes={"catalog.vt.edu": "special"},
        ))

    assert rendered_urls == []  # never sent to the browser
    assert not storage.upload_document.called  # nothing ingested
    counts = fr.count_by_status()
    assert counts.get("skipped", 0) == 1
    assert counts.get("pending", 0) == 0  # drained, no infinite lease loop
