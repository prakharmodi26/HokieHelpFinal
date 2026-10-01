# services/crawler/src/crawler/frontier.py
"""Ephemeral per-job crawl frontier backed by SQLite.

One row per URL (keyed by a 16-hex digest). Status lifecycle:
  pending -> leased -> done | failed | skipped

Within-run URL dedup only: enqueue() returns False on digest conflict (URL
already seen this run). No cross-run content dedup — the DB lives on an
emptyDir in k8s and dies with the pod, so every run starts fresh.

Stores the hop_history (referrer chain) and content_hash of stored pages for
run-summary reporting only; content_hash is NOT used to skip duplicate content.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass


def url_digest(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@dataclass
class FrontierItem:
    url: str
    scope: str
    depth: int
    hop_history: list[str]
    discovery_source: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    digest           TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    host             TEXT,
    scope            TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    depth            INTEGER NOT NULL DEFAULT 0,
    hop_history      TEXT NOT NULL DEFAULT '[]',
    discovery_source TEXT,
    content_hash     TEXT,
    http_status      INTEGER,
    error            TEXT,
    wedge_count      INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON frontier(status);
CREATE INDEX IF NOT EXISTS idx_content_hash ON frontier(content_hash);
"""


class Frontier:
    def __init__(self, path: str) -> None:
        # timeout=5.0: if another process holds the SQLite lock, wait up to 5s
        # before raising OperationalError instead of failing immediately. This
        # is a defensive guardrail — admin should still give each Job a unique
        # FRONTIER_DB_PATH to avoid concurrent access on the same DB file (the
        # frontier lives on an ephemeral emptyDir, one per pod).
        self._conn = sqlite3.connect(path, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after the initial schema to a pre-existing DB.

        The frontier normally lives on an ephemeral emptyDir, so CREATE TABLE
        builds the current schema (incl. wedge_count) fresh every run. But a dev
        reusing a persistent CRAWL_FRONTIER_DB_PATH from an older build keeps the
        old table — CREATE TABLE IF NOT EXISTS is a no-op — and would hit
        'no such column: wedge_count' on the first render wedge. Add any missing
        column idempotently.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(frontier)")}
        if "wedge_count" not in cols:
            self._conn.execute(
                "ALTER TABLE frontier ADD COLUMN wedge_count INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def enqueue(
        self,
        url: str,
        *,
        scope: str,
        depth: int,
        hop_history: list[str],
        discovery_source: str,
    ) -> bool:
        """Insert a new URL. Returns False if the digest already exists."""
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        try:
            self._conn.execute(
                "INSERT INTO frontier(digest, url, host, scope, depth, hop_history, discovery_source) "
                "VALUES(?,?,?,?,?,?,?)",
                (url_digest(url), url, host, scope, depth,
                 json.dumps(hop_history), discovery_source),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def lease_pending(self, limit: int) -> list[FrontierItem]:
        """Atomically move up to `limit` pending rows to 'leased' and return them."""
        cur = self._conn.execute(
            "SELECT digest, url, scope, depth, hop_history, discovery_source "
            "FROM frontier WHERE status='pending' ORDER BY depth ASC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        digests = [r[0] for r in rows]
        if digests:
            self._conn.executemany(
                "UPDATE frontier SET status='leased', updated_at=datetime('now') WHERE digest=?",
                [(d,) for d in digests],
            )
            self._conn.commit()
        return [
            FrontierItem(
                url=r[1], scope=r[2], depth=r[3],
                hop_history=json.loads(r[4]), discovery_source=r[5],
            )
            for r in rows
        ]

    def mark_done(self, url: str, *, content_hash: str, http_status: int) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='done', content_hash=?, http_status=?, "
            "updated_at=datetime('now') WHERE digest=?",
            (content_hash, http_status, url_digest(url)),
        )
        self._conn.commit()

    def mark_failed(self, url: str, *, error: str, http_status: int | None = None) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='failed', error=?, http_status=?, "
            "updated_at=datetime('now') WHERE digest=?",
            (error[:500], http_status, url_digest(url)),
        )
        self._conn.commit()

    def mark_skipped(self, url: str, *, reason: str) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='skipped', error=?, updated_at=datetime('now') WHERE digest=?",
            (reason[:500], url_digest(url)),
        )
        self._conn.commit()

    def requeue_leased(self) -> int:
        """Reset rows stranded in 'leased' (by a crashed prior run) to 'pending'."""
        cur = self._conn.execute(
            "UPDATE frontier SET status='pending', updated_at=datetime('now') "
            "WHERE status='leased'"
        )
        self._conn.commit()
        return cur.rowcount

    def requeue_failed(self) -> int:
        cur = self._conn.execute(
            "UPDATE frontier SET status='pending', error=NULL WHERE status='failed'"
        )
        self._conn.commit()
        return cur.rowcount

    def requeue_wedged(self, url: str, *, max_wedges: int, error: str) -> str:
        """Account for a URL that a render-watchdog wedge left un-returned.

        Increments the URL's wedge counter. While the counter is below
        `max_wedges` the row is returned to 'pending' so it is re-leased and
        re-rendered on a later pass — a single poison page can no longer
        permanently drop its (up to batch_size-1) innocent batch siblings that
        simply had not streamed yet (BUG-A1). Once the URL has wedged
        `max_wedges` times it is marked 'failed' permanently (it is the genuine
        straggler). Returns the resulting status: 'pending' or 'failed'.
        """
        digest = url_digest(url)
        row = self._conn.execute(
            "SELECT wedge_count FROM frontier WHERE digest=?", (digest,)
        ).fetchone()
        count = (row[0] if row else 0) + 1
        if count >= max_wedges:
            self._conn.execute(
                "UPDATE frontier SET status='failed', error=?, wedge_count=?, "
                "updated_at=datetime('now') WHERE digest=?",
                (error[:500], count, digest),
            )
            status = "failed"
        else:
            self._conn.execute(
                "UPDATE frontier SET status='pending', error=NULL, wedge_count=?, "
                "updated_at=datetime('now') WHERE digest=?",
                (count, digest),
            )
            status = "pending"
        self._conn.commit()
        return status

    def count_by_host(self) -> dict[str, int]:
        """Rows per host (any status) — used to enforce per-host URL caps."""
        cur = self._conn.execute(
            "SELECT host, COUNT(*) FROM frontier GROUP BY host"
        )
        return {row[0]: row[1] for row in cur.fetchall() if row[0]}

    def count_by_host_status(self) -> dict[str, dict[str, int]]:
        """Per-host status breakdown: {host: {done: N, failed: N, ...}}.

        Used in the run summary so admin can see which hosts succeeded or
        failed without mining logs.
        """
        cur = self._conn.execute(
            "SELECT host, status, COUNT(*) FROM frontier "
            "WHERE host IS NOT NULL AND host != '' "
            "GROUP BY host, status"
        )
        result: dict[str, dict[str, int]] = {}
        for host, status, count in cur.fetchall():
            result.setdefault(host, {})[status] = count
        return result

    def count_by_status(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM frontier GROUP BY status"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def iter_status(self, status: str):
        """Yield (url, host, scope, content_hash, http_status) for a status."""
        cur = self._conn.execute(
            "SELECT url, host, scope, content_hash, http_status FROM frontier WHERE status=?",
            (status,),
        )
        yield from cur.fetchall()

    def close(self) -> None:
        self._conn.close()
