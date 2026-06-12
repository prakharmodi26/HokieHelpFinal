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


def test_process_result_catalog_mode_stores_card_and_no_children(tmp_path):
    """Catalog-mode hosts: metadata-only card, NEVER body ingest, NO BFS expansion."""
    fr = Frontier(str(tmp_path / "f.db"))
    storage = MagicMock()
    item = FrontierItem(
        url="https://vtechworks.lib.vt.edu/items/abc", scope="vt_page", depth=0,
        hop_history=["https://vtechworks.lib.vt.edu/items/abc"],
        discovery_source="registry",
    )
    result = _fake_result("https://vtechworks.lib.vt.edu/items/abc")
    result.html = (
        "<html><head>"
        '<meta name="citation_title" content="A Thesis">'
        '<meta name="citation_author" content="Doe, Jane">'
        '<meta name="citation_publication_date" content="2021-05-01">'
        "</head><body>full text that must NOT be ingested</body></html>"
    )
    result.links = {
        "internal": [{"href": "https://vtechworks.lib.vt.edu/items/def"}],
        "external": [{"href": "https://example.com/x"}],
    }
    new_docs = process_result(
        result, item, fr, storage,
        main_bucket="crawled-pages", external_bucket="crawled-pages-external",
        external_hops=1, max_depth=4, mode="catalog",
    )
    # card stored with page_type=catalog_card; body markdown not ingested
    content = storage.upload_document.call_args[0][1]
    assert "page_type: 'catalog_card'" in content
    assert "A Thesis" in content
    assert "Doe, Jane" in content
    assert "full text that must NOT be ingested" not in content
    # marked done, and ZERO children enqueued despite links on the page
    counts = fr.count_by_status()
    assert counts["done"] == 1
    assert counts.get("pending", 0) == 0
    # no document URLs harvested from catalog pages either
    assert new_docs == set()


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
