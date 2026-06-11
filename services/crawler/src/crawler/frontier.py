# services/crawler/src/crawler/frontier.py
"""Durable, resumable crawl frontier backed by SQLite.

One row per URL (keyed by a 16-hex digest). Status lifecycle:
  pending -> leased -> done | failed
Stores the full hop_history (referrer chain) and the content_hash of stored
pages so a re-run skips unchanged content. The DB file lives on a PVC in k8s so
a restarted crawl Job resumes instead of starting over.
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
    updated_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON frontier(status);
CREATE INDEX IF NOT EXISTS idx_content_hash ON frontier(content_hash);
"""


class Frontier:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
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

    def content_hash_seen(self, content_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM frontier WHERE content_hash=? LIMIT 1", (content_hash,)
        )
        return cur.fetchone() is not None

    def count_by_host(self) -> dict[str, int]:
        """Rows per host (any status) — used to enforce per-host URL caps."""
        cur = self._conn.execute(
            "SELECT host, COUNT(*) FROM frontier GROUP BY host"
        )
        return {row[0]: row[1] for row in cur.fetchall() if row[0]}

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
