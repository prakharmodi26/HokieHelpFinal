from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite


class Store:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running', stage TEXT NOT NULL DEFAULT 'crawler',
                stats TEXT, settings TEXT)"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)"""
            )
            await db.commit()

    async def create_run(self, settings: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO runs (id, started_at, status, stage, settings) VALUES (?, ?, 'running', 'crawler', ?)",
                (run_id, now, json.dumps(settings)),
            )
            await db.commit()
        return run_id

    async def update_run(self, run_id: str, **kwargs: Any) -> None:
        allowed = {"status", "stage", "finished_at", "stats"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "stats" in updates and isinstance(updates["stats"], dict):
            updates["stats"] = json.dumps(updates["stats"])
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE runs SET {set_clause} WHERE id = ?",
                list(updates.values()) + [run_id],
            )
            await db.commit()

    async def complete_run(self, run_id: str, stats: dict[str, Any]) -> None:
        await self.update_run(
            run_id,
            status="completed",
            stage="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
            stats=stats,
        )

    async def fail_run(self, run_id: str, stage: str) -> None:
        await self.update_run(
            run_id,
            status="failed",
            stage=stage,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def set_setting(self, key: str, value: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await db.commit()

    async def get_setting(
        self, key: str, default: str | None = None
    ) -> str | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else default

    async def get_all_settings(self) -> dict[str, str]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT key, value FROM settings") as cur:
                return {r[0]: r[1] for r in await cur.fetchall()}
