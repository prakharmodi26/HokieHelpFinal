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
