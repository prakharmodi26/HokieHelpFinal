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
