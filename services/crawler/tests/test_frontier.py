# services/crawler/tests/test_frontier.py
import hashlib

from crawler.frontier import Frontier, FrontierItem


def _digest(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def test_enqueue_dedup_and_counts(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    added1 = db.enqueue("https://vt.edu/", scope="vt_page", depth=0,
                        hop_history=["https://vt.edu/"], discovery_source="seed")
    added2 = db.enqueue("https://vt.edu/", scope="vt_page", depth=0,
                        hop_history=["https://vt.edu/"], discovery_source="seed")
    assert added1 is True
    assert added2 is False  # same URL digest -> not re-added
    assert db.count_by_status()["pending"] == 1


def test_lease_and_mark_done(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=1,
               hop_history=["https://vt.edu/", "https://a.vt.edu/x"],
               discovery_source="bfs")
    batch = db.lease_pending(limit=10)
    assert len(batch) == 1
    item = batch[0]
    assert isinstance(item, FrontierItem)
    assert item.url == "https://a.vt.edu/x"
    assert item.hop_history[-1] == "https://a.vt.edu/x"
    # leased items are no longer pending
    assert db.count_by_status().get("pending", 0) == 0
    assert db.count_by_status()["leased"] == 1

    db.mark_done("https://a.vt.edu/x", content_hash="abc", http_status=200)
    counts = db.count_by_status()
    assert counts["done"] == 1
    assert counts.get("leased", 0) == 0


def test_mark_failed_and_requeue(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    db.lease_pending(limit=10)
    db.mark_failed("https://a.vt.edu/x", error="timeout")
    assert db.count_by_status()["failed"] == 1
    # requeue resets failed -> pending for a retry pass
    n = db.requeue_failed()
    assert n == 1
    assert db.count_by_status()["pending"] == 1


def test_content_hash_seen_dedup(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    db.lease_pending(limit=10)
    db.mark_done("https://a.vt.edu/x", content_hash="HASH1", http_status=200)
    assert db.content_hash_seen("HASH1") is True
    assert db.content_hash_seen("NOPE") is False


def test_resumability_reopen(tmp_path):
    path = str(tmp_path / "f.db")
    db = Frontier(path)
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    del db
    db2 = Frontier(path)  # reopen same file
    assert db2.count_by_status()["pending"] == 1
