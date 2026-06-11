# services/crawler/tests/test_engine.py
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from crawler.engine import process_result
from crawler.frontier import Frontier, FrontierItem
from crawler.scope import Scope


def _fake_result(url, fit="# Title\n" + "word " * 40, success=True):
    md = SimpleNamespace(fit_markdown=fit, raw_markdown=fit + "\n[nav]")
    return SimpleNamespace(
        url=url, success=success, markdown=md,
        metadata={"title": "T"}, links={"internal": [], "external": []},
        status_code=200, response_headers={"content-type": "text/html"},
        error_message=None,
    )


def test_process_result_stores_vt_page_and_enqueues_children(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    storage = MagicMock()
    item = FrontierItem(
        url="https://cs.vt.edu/people", scope="vt_page", depth=0,
        hop_history=["https://cs.vt.edu/people"], discovery_source="seed",
    )
    result = _fake_result("https://cs.vt.edu/people")
    result.links = {
        "internal": [{"href": "https://cs.vt.edu/people/jane.html"}],
        "external": [{"href": "https://linkedin.com/in/jane"}],
    }
    new_docs = process_result(
        result, item, fr, storage,
        main_bucket="crawled-pages", external_bucket="crawled-pages-external",
        external_hops=1, max_depth=4,
    )
    # stored to main bucket
    assert storage.upload_document.called
    args, kwargs = storage.upload_document.call_args
    assert kwargs.get("bucket", args[2] if len(args) > 2 else None) in (None, "crawled-pages")
    # frontier marked done
    assert fr.count_by_status()["done"] == 1
    # internal child + external child enqueued (external within 1 hop)
    counts = fr.count_by_status()
    assert counts["pending"] == 2
    # documents collected from links returned for later processing
    assert isinstance(new_docs, set)


def test_process_result_empty_returns_needs_js(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    storage = MagicMock()
    item = FrontierItem(
        url="https://dining.vt.edu/menus", scope="vt_page", depth=0,
        hop_history=["https://dining.vt.edu/menus"], discovery_source="seed",
    )
    result = _fake_result("https://dining.vt.edu/menus", fit="loading...")
    needs_js = process_result(
        result, item, fr, storage,
        main_bucket="crawled-pages", external_bucket="crawled-pages-external",
        external_hops=1, max_depth=4, tier="fast",
    )
    # fast tier + empty -> signalled for JS retry; nothing stored, not marked done
    assert needs_js == "needs_js"
    assert fr.count_by_status().get("done", 0) == 0
