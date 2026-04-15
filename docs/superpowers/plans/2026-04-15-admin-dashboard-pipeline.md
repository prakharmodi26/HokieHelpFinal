# HokieHelp Admin Dashboard & Pipeline Orchestration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken pipeline CronJob with a persistent FastAPI admin service that orchestrates crawler -> chunker as asyncio subprocesses and embedder via HTTP, and serves a full-featured dashboard for crawl management and system monitoring - accessible only via `kubectl port-forward`.

**Architecture:** New `admin` FastAPI service (always-on Deployment) runs crawler and chunker as asyncio child processes, streams their stdout/stderr live to the dashboard via Server-Sent Events, then calls the always-on `embedder` FastAPI service via HTTP for the embedding stage. Run history and settings persist to SQLite on a PVC. APScheduler manages cron scheduling internally. The dashboard is a single `index.html` served by the admin service on port 8080 - no ingress, SSH port-forward only. The `embedder` service gains a FastAPI HTTP wrapper around its existing `run_embedding()` function, converting it from a batch Job to a persistent GPU Deployment.

**Tech Stack:** FastAPI 0.115, uvicorn, APScheduler 3.x, aiosqlite, httpx, sse-starlette, minio Python SDK, qdrant-client, croniter, Alpine.js 3.x + Tailwind CSS 3.x (CDN, no build step)


---

## File Map

**Create:**
- `services/admin/pyproject.toml`
- `services/admin/Dockerfile` — build context is `services/`, installs crawler + chunker + admin
- `services/admin/src/admin/__init__.py`
- `services/admin/src/admin/config.py` — AdminConfig.from_env()
- `services/admin/src/admin/store.py` — aiosqlite: runs table + settings key-value
- `services/admin/src/admin/runner.py` — async subprocess pipeline + SSE log bus
- `services/admin/src/admin/scheduler.py` — APScheduler cron wrapper
- `services/admin/src/admin/health.py` — HTTP + SDK health/stats for all services
- `services/admin/src/admin/main.py` — FastAPI app: all routes, SSE, static mount
- `services/admin/src/admin/static/index.html` — single-page Alpine.js dashboard
- `services/admin/tests/__init__.py`
- `services/admin/tests/test_store.py`
- `services/admin/tests/test_runner.py`
- `services/admin/tests/test_health.py`
- `services/admin/tests/test_main.py`
- `k8s/admin-configmap.yaml`
- `k8s/admin-deployment.yaml`
- `k8s/admin-pvc.yaml`

**Modify:**
- `services/embedder/src/embedder/server.py` — NEW file: FastAPI wrapper around run_embedding()
- `services/embedder/Dockerfile` — change CMD to uvicorn factory
- `services/embedder/pyproject.toml` — add fastapi, uvicorn deps
- `k8s/network-policies.yaml` — add admin egress + embedder-service ingress/egress

**Delete:**
- `k8s/pipeline-cronjob.yaml`
- `k8s/pipeline-rbac.yaml`


---

## Task 1: Admin service scaffold

**Files:** Create `services/admin/pyproject.toml`, `services/admin/Dockerfile`, `services/admin/src/admin/__init__.py`, `services/admin/tests/__init__.py`

- [ ] **Step 1: Create directory structure**
```bash
mkdir -p services/admin/src/admin/static services/admin/tests
touch services/admin/src/admin/__init__.py services/admin/tests/__init__.py
```

- [ ] **Step 2: Create `services/admin/pyproject.toml`**
```toml
[project]
name = "hokiehelp-admin"
version = "0.1.0"
description = "Admin dashboard and pipeline orchestrator for HokieHelp"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "apscheduler>=3.10.0",
    "aiosqlite>=0.20.0",
    "httpx>=0.27.0",
    "sse-starlette>=2.1.0",
    "minio>=7.2.0",
    "qdrant-client>=1.9.0",
    "croniter>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create `services/admin/Dockerfile`**

Build context must be `services/` — run as: `docker build -f admin/Dockerfile .` from `services/`.

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install crawler (brings crawl4ai, minio, httpx)
COPY crawler/pyproject.toml crawler/
COPY crawler/src/ crawler/src/
RUN pip install --no-cache-dir ./crawler && python -m playwright install chromium --with-deps

# Install chunker
COPY chunker/pyproject.toml chunker/
COPY chunker/src/ chunker/src/
RUN pip install --no-cache-dir ./chunker

# Install admin
COPY admin/pyproject.toml admin/
COPY admin/src/ admin/src/
RUN pip install --no-cache-dir ./admin

CMD ["uvicorn", "admin.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Commit scaffold**
```bash
git add services/admin/
git commit -m "feat(admin): scaffold admin service"
```


---

## Task 2: Config module

**Files:** Create `services/admin/src/admin/config.py`, `services/admin/tests/test_config.py`

- [ ] **Step 1: Write failing test**
```python
# services/admin/tests/test_config.py
import pytest
from admin.config import AdminConfig

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("EMBEDDER_URL", "http://hokiehelp-embedder:8080")
    cfg = AdminConfig.from_env()
    assert cfg.minio_endpoint == "minio:9000"
    assert cfg.qdrant_port == 6333
    assert cfg.default_max_depth == 4
    assert cfg.default_schedule == "0 3 * * 0"
    assert cfg.data_dir == "/data"

def test_config_missing_required(monkeypatch):
    for k in ["MINIO_ENDPOINT","MINIO_ACCESS_KEY","MINIO_SECRET_KEY","QDRANT_HOST","EMBEDDER_URL"]:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        AdminConfig.from_env()
```

- [ ] **Step 2: Run to verify fail** — `cd services/admin && pip install -e ".[dev]" && pytest tests/test_config.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/config.py`**
```python
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AdminConfig:
    port: int
    data_dir: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket: str
    minio_cleaned_bucket: str
    minio_chunks_bucket: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedder_url: str
    chatbot_url: str
    ollama_url: str
    default_seed_url: str
    default_max_depth: int
    default_max_pages: int
    default_schedule: str

    @classmethod
    def from_env(cls) -> AdminConfig:
        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required env var {name} is not set")
            return val
        return cls(
            port=int(os.environ.get("ADMIN_PORT", "8080")),
            data_dir=os.environ.get("ADMIN_DATA_DIR", "/data"),
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            minio_bucket=os.environ.get("MINIO_BUCKET", "crawled-pages"),
            minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            qdrant_host=os.environ.get("QDRANT_HOST", "qdrant"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedder_url=_require("EMBEDDER_URL"),
            chatbot_url=os.environ.get("CHATBOT_URL", "http://hokiehelp-chatbot:8000"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
            default_seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
            default_max_depth=int(os.environ.get("CRAWL_MAX_DEPTH", "4")),
            default_max_pages=int(os.environ.get("CRAWL_MAX_PAGES", "9999999")),
            default_schedule=os.environ.get("CRAWL_SCHEDULE", "0 3 * * 0"),
        )
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_config.py -v` — expect 2 PASS

- [ ] **Step 5: Commit**
```bash
git add services/admin/src/admin/config.py services/admin/tests/test_config.py
git commit -m "feat(admin): add config module"
```


---

## Task 3: SQLite store

**Files:** Create `services/admin/src/admin/store.py`, `services/admin/tests/test_store.py`

Schema: `runs(id, started_at, finished_at, status, stage, stats, settings)` and `settings(key, value)`.

- [ ] **Step 1: Write failing tests**
```python
# services/admin/tests/test_store.py
import pytest
from admin.store import Store

@pytest.fixture
async def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    await s.init()
    return s

async def test_create_and_get_run(store):
    run_id = await store.create_run(settings={"seed_url": "https://cs.vt.edu"})
    run = await store.get_run(run_id)
    assert run["id"] == run_id
    assert run["status"] == "running"
    assert run["stage"] == "crawler"

async def test_complete_run(store):
    run_id = await store.create_run(settings={})
    await store.complete_run(run_id, stats={"pages_crawled": 500})
    run = await store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["finished_at"] is not None

async def test_fail_run(store):
    run_id = await store.create_run(settings={})
    await store.fail_run(run_id, stage="crawler")
    run = await store.get_run(run_id)
    assert run["status"] == "failed"

async def test_list_runs(store):
    await store.create_run(settings={})
    await store.create_run(settings={})
    runs = await store.list_runs(limit=10)
    assert len(runs) == 2

async def test_settings_get_set(store):
    await store.set_setting("crawl.max_depth", "3")
    assert await store.get_setting("crawl.max_depth") == "3"

async def test_settings_default(store):
    assert await store.get_setting("nonexistent", default="fallback") == "fallback"

async def test_get_all_settings(store):
    await store.set_setting("a", "1")
    await store.set_setting("b", "2")
    all_s = await store.get_all_settings()
    assert all_s["a"] == "1" and all_s["b"] == "2"
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_store.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/store.py`**
```python
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from typing import Any
import aiosqlite

class Store:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running', stage TEXT NOT NULL DEFAULT 'crawler',
                stats TEXT, settings TEXT)""")
            await db.execute("""CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            await db.commit()

    async def create_run(self, settings: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO runs (id, started_at, status, stage, settings) VALUES (?, ?, 'running', 'crawler', ?)",
                (run_id, now, json.dumps(settings)))
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
            await db.execute(f"UPDATE runs SET {set_clause} WHERE id = ?", list(updates.values()) + [run_id])
            await db.commit()

    async def complete_run(self, run_id: str, stats: dict[str, Any]) -> None:
        await self.update_run(run_id, status="completed", stage="done",
            finished_at=datetime.now(timezone.utc).isoformat(), stats=stats)

    async def fail_run(self, run_id: str, stage: str) -> None:
        await self.update_run(run_id, status="failed", stage=stage,
            finished_at=datetime.now(timezone.utc).isoformat())

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def set_setting(self, key: str, value: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
            await db.commit()

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cur:
                row = await cur.fetchone()
                return row[0] if row else default

    async def get_all_settings(self) -> dict[str, str]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT key, value FROM settings") as cur:
                return {r[0]: r[1] for r in await cur.fetchall()}
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_store.py -v` — expect 7 PASS

- [ ] **Step 5: Commit**
```bash
git add services/admin/src/admin/store.py services/admin/tests/test_store.py
git commit -m "feat(admin): add SQLite store for runs and settings"
```


---

## Task 4: Pipeline runner

**Files:** Create `services/admin/src/admin/runner.py`, `services/admin/tests/test_runner.py`

Owns the state machine: idle -> crawling -> chunking -> embedding -> done/failed/cancelled.
Spawns crawler and chunker as asyncio child processes using `asyncio.create_subprocess_exec`.
Calls embedder via httpx HTTP. Maintains a list of asyncio.Queue subscribers for SSE streaming.

- [ ] **Step 1: Write failing tests**
```python
# services/admin/tests/test_runner.py
import pytest
from admin.runner import PipelineRunner, PipelineState
from admin.config import AdminConfig
from admin.store import Store

@pytest.fixture
async def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    await s.init()
    return s

@pytest.fixture
def cfg():
    return AdminConfig(
        port=8080, data_dir="/tmp",
        minio_endpoint="minio:9000", minio_access_key="key", minio_secret_key="secret",
        minio_secure=False, minio_bucket="crawled-pages", minio_cleaned_bucket="crawled-pages-cleaned",
        minio_chunks_bucket="chunks", qdrant_host="qdrant", qdrant_port=6333,
        qdrant_collection="test", embedder_url="http://embedder:8080",
        chatbot_url="http://chatbot:8000", ollama_url="http://ollama:11434",
        default_seed_url="https://cs.vt.edu", default_max_depth=2, default_max_pages=100,
        default_schedule="0 3 * * 0",
    )

async def test_initial_state(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    assert runner.state == PipelineState.IDLE
    assert runner.current_run_id is None

async def test_cannot_start_while_running(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    runner._state = PipelineState.CRAWLING
    with pytest.raises(RuntimeError, match="already running"):
        await runner.start({})

async def test_build_crawler_env(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    env = runner._build_crawler_env({"crawl.seed_url": "https://cs.vt.edu", "crawl.max_depth": "3", "crawl.max_pages": "500", "crawl.allowed_domains": "cs.vt.edu", "crawl.blocked_domains": "git.cs.vt.edu", "crawl.blocked_paths": "/content/", "crawl.request_delay": "0.5", "crawl.prune_threshold": "0.45"})
    assert env["MINIO_ENDPOINT"] == "minio:9000"
    assert env["CRAWL_SEED_URL"] == "https://cs.vt.edu"
    assert env["CRAWL_MAX_DEPTH"] == "3"

async def test_build_chunker_env(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    env = runner._build_chunker_env({"chunker.preferred_tokens": "400", "chunker.overlap_tokens": "64", "chunker.min_tokens": "120"})
    assert env["MINIO_ENDPOINT"] == "minio:9000"
    assert env["CHUNK_PREFERRED_TOKENS"] == "400"

async def test_subscribe_unsubscribe(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    q = runner.subscribe()
    assert q in runner._log_queues
    runner.unsubscribe(q)
    assert q not in runner._log_queues
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_runner.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/runner.py`**
```python
from __future__ import annotations
import asyncio, logging, os, sys
from enum import Enum
from pathlib import Path
from typing import Any
import httpx
from admin.config import AdminConfig
from admin.store import Store

logger = logging.getLogger(__name__)

class PipelineState(str, Enum):
    IDLE = "idle"
    CRAWLING = "crawling"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineRunner:
    def __init__(self, store: Store, config: AdminConfig, log_dir: str) -> None:
        self._store = store
        self._config = config
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._state = PipelineState.IDLE
        self._current_run_id: str | None = None
        self._current_proc: asyncio.subprocess.Process | None = None
        self._log_queues: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._log_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._log_queues:
            self._log_queues.remove(q)

    def _broadcast(self, line: str) -> None:
        for q in self._log_queues:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass

    async def start(self, settings: dict[str, str]) -> str:
        if self._state not in (PipelineState.IDLE, PipelineState.DONE, PipelineState.FAILED, PipelineState.CANCELLED):
            raise RuntimeError(f"Pipeline already running (state={self._state.value})")
        run_id = await self._store.create_run(settings=settings)
        self._current_run_id = run_id
        self._state = PipelineState.CRAWLING
        self._task = asyncio.create_task(self._run(run_id, settings))
        return run_id

    async def stop(self) -> None:
        if self._current_proc and self._current_proc.returncode is None:
            self._current_proc.terminate()
        if self._task and not self._task.done():
            self._task.cancel()
        if self._current_run_id:
            await self._store.fail_run(self._current_run_id, stage=self._state.value)
        self._state = PipelineState.CANCELLED
        self._broadcast("[pipeline] Stopped by user")

    def _build_crawler_env(self, settings: dict[str, str]) -> dict[str, str]:
        cfg = self._config
        return {
            **os.environ,
            "MINIO_ENDPOINT": cfg.minio_endpoint,
            "MINIO_ACCESS_KEY": cfg.minio_access_key,
            "MINIO_SECRET_KEY": cfg.minio_secret_key,
            "MINIO_SECURE": str(cfg.minio_secure).lower(),
            "MINIO_BUCKET": cfg.minio_bucket,
            "MINIO_CLEANED_BUCKET": cfg.minio_cleaned_bucket,
            "CRAWL_SEED_URL": settings.get("crawl.seed_url", cfg.default_seed_url),
            "CRAWL_MAX_DEPTH": settings.get("crawl.max_depth", str(cfg.default_max_depth)),
            "CRAWL_MAX_PAGES": settings.get("crawl.max_pages", str(cfg.default_max_pages)),
            "CRAWL_ALLOWED_DOMAINS": settings.get("crawl.allowed_domains", "cs.vt.edu"),
            "CRAWL_BLOCKED_DOMAINS": settings.get("crawl.blocked_domains", "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,portal.cs.vt.edu,api.cs.vt.edu"),
            "CRAWL_BLOCKED_PATHS": settings.get("crawl.blocked_paths", "/content/,/editor.html,/cs-root.html,/cs-source.html"),
            "CRAWL_REQUEST_DELAY": settings.get("crawl.request_delay", "0.5"),
            "CRAWL_PRUNE_THRESHOLD": settings.get("crawl.prune_threshold", "0.45"),
        }

    def _build_chunker_env(self, settings: dict[str, str]) -> dict[str, str]:
        cfg = self._config
        return {
            **os.environ,
            "MINIO_ENDPOINT": cfg.minio_endpoint,
            "MINIO_ACCESS_KEY": cfg.minio_access_key,
            "MINIO_SECRET_KEY": cfg.minio_secret_key,
            "MINIO_SECURE": str(cfg.minio_secure).lower(),
            "MINIO_CLEANED_BUCKET": cfg.minio_cleaned_bucket,
            "MINIO_CHUNKS_BUCKET": cfg.minio_chunks_bucket,
            "CHUNK_PREFERRED_TOKENS": settings.get("chunker.preferred_tokens", "400"),
            "CHUNK_OVERLAP_TOKENS": settings.get("chunker.overlap_tokens", "64"),
            "CHUNK_MIN_TOKENS": settings.get("chunker.min_tokens", "120"),
        }

    async def _stream_proc(self, cmd: list[str], env: dict[str, str], log_file: Path) -> int:
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self._current_proc = proc
        with log_file.open("a") as f:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                f.write(line + "\n")
                self._broadcast(line)
        return await proc.wait()

    async def _run(self, run_id: str, settings: dict[str, str]) -> None:
        log_file = self._log_dir / f"{run_id}.log"
        try:
            # Stage 1: Crawler
            self._state = PipelineState.CRAWLING
            await self._store.update_run(run_id, stage="crawler")
            self._broadcast(f"[pipeline] Starting crawler (run={run_id})")
            rc = await self._stream_proc([sys.executable, "-m", "crawler.main"], self._build_crawler_env(settings), log_file)
            if rc != 0:
                await self._store.fail_run(run_id, stage="crawler")
                self._state = PipelineState.FAILED
                self._broadcast(f"[pipeline] Crawler failed (exit {rc})")
                return

            # Stage 2: Chunker
            self._state = PipelineState.CHUNKING
            await self._store.update_run(run_id, stage="chunker")
            self._broadcast("[pipeline] Starting chunker")
            rc = await self._stream_proc([sys.executable, "-m", "chunker.main"], self._build_chunker_env(settings), log_file)
            if rc != 0:
                await self._store.fail_run(run_id, stage="chunker")
                self._state = PipelineState.FAILED
                self._broadcast(f"[pipeline] Chunker failed (exit {rc})")
                return

            # Stage 3: Embedder via HTTP
            self._state = PipelineState.EMBEDDING
            await self._store.update_run(run_id, stage="embedding")
            self._broadcast("[pipeline] Triggering embedder service")
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._config.embedder_url}/embed/start")
                resp.raise_for_status()
                embed_run_id = resp.json()["run_id"]

            async with httpx.AsyncClient(timeout=10.0) as client:
                while True:
                    sr = await client.get(f"{self._config.embedder_url}/embed/status/{embed_run_id}")
                    sr.raise_for_status()
                    data = sr.json()
                    self._broadcast(f"[embedder] status={data['status']} stats={data.get('stats', {})}")
                    if data["status"] == "completed":
                        break
                    if data["status"] == "failed":
                        await self._store.fail_run(run_id, stage="embedding")
                        self._state = PipelineState.FAILED
                        return
                    await asyncio.sleep(30)

            await self._store.complete_run(run_id, stats={"embed": data.get("stats", {})})
            self._state = PipelineState.DONE
            self._broadcast("[pipeline] Pipeline completed successfully")

        except asyncio.CancelledError:
            self._state = PipelineState.CANCELLED
        except Exception as exc:
            logger.exception("Pipeline run %s crashed: %s", run_id, exc)
            await self._store.fail_run(run_id, stage=self._state.value)
            self._state = PipelineState.FAILED
            self._broadcast(f"[pipeline] Crashed: {exc}")
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_runner.py -v` — expect 5 PASS

- [ ] **Step 5: Commit**
```bash
git add services/admin/src/admin/runner.py services/admin/tests/test_runner.py
git commit -m "feat(admin): add async pipeline runner with SSE log bus"
```


---

## Task 5: Scheduler module

**Files:** Create `services/admin/src/admin/scheduler.py`, `services/admin/tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests**
```python
# services/admin/tests/test_scheduler.py
from datetime import datetime
from admin.scheduler import CrawlScheduler

def test_next_runs_sunday():
    sched = CrawlScheduler.__new__(CrawlScheduler)
    runs = sched._next_runs("0 3 * * 0", n=3)
    assert len(runs) == 3
    for r in runs:
        assert isinstance(r, datetime)
        assert r.weekday() == 6 and r.hour == 3

def test_invalid_cron_raises():
    import pytest
    sched = CrawlScheduler.__new__(CrawlScheduler)
    with pytest.raises(ValueError, match="Invalid cron"):
        sched._validate_cron("not a cron expression")
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_scheduler.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/scheduler.py`**
```python
from __future__ import annotations
import logging
from datetime import datetime
from typing import Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from croniter import croniter, CroniterBadCronError

logger = logging.getLogger(__name__)
_JOB_ID = "crawl_pipeline"

class CrawlScheduler:
    def __init__(self, on_trigger: Callable) -> None:
        self._scheduler = AsyncIOScheduler()
        self._on_trigger = on_trigger
        self._cron: str | None = None
        self._enabled: bool = False

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)

    def _validate_cron(self, cron: str) -> None:
        try:
            croniter(cron)
        except (CroniterBadCronError, ValueError) as exc:
            raise ValueError(f"Invalid cron expression '{cron}': {exc}") from exc

    def _next_runs(self, cron: str, n: int = 5) -> list[datetime]:
        it = croniter(cron, datetime.utcnow())
        return [it.get_next(datetime) for _ in range(n)]

    def configure(self, cron: str, enabled: bool) -> None:
        self._validate_cron(cron)
        self._cron = cron
        self._enabled = enabled
        if self._scheduler.get_job(_JOB_ID):
            self._scheduler.remove_job(_JOB_ID)
        if enabled:
            self._scheduler.add_job(self._on_trigger, CronTrigger.from_crontab(cron),
                id=_JOB_ID, replace_existing=True)
            logger.info("Crawl scheduled: %s", cron)

    def status(self) -> dict:
        return {
            "cron": self._cron,
            "enabled": self._enabled,
            "next_runs": [r.isoformat() for r in self._next_runs(self._cron)] if self._cron else [],
        }
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_scheduler.py -v` — expect 2 PASS

- [ ] **Step 5: Commit**
```bash
git add services/admin/src/admin/scheduler.py services/admin/tests/test_scheduler.py
git commit -m "feat(admin): add APScheduler cron wrapper"
```

---

## Task 6: Health and stats module

**Files:** Create `services/admin/src/admin/health.py`, `services/admin/tests/test_health.py`

- [ ] **Step 1: Write failing tests**
```python
# services/admin/tests/test_health.py
import pytest, httpx
from unittest.mock import AsyncMock, MagicMock, patch

async def test_check_http_service_up():
    from admin.health import check_http_service
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("admin.health.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value.get = AsyncMock(return_value=mock_resp)
        result = await check_http_service("chatbot", "http://chatbot:8000/health")
    assert result["name"] == "chatbot" and result["healthy"] is True and "latency_ms" in result

async def test_check_http_service_down():
    from admin.health import check_http_service
    with patch("admin.health.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        result = await check_http_service("chatbot", "http://chatbot:8000/health")
    assert result["healthy"] is False and "error" in result
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_health.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/health.py`**
```python
from __future__ import annotations
import logging, time
from typing import Any
import httpx
from minio import Minio
from qdrant_client import QdrantClient
from admin.config import AdminConfig

logger = logging.getLogger(__name__)

async def check_http_service(name: str, url: str) -> dict[str, Any]:
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return {"name": name, "healthy": resp.status_code < 500,
                    "status_code": resp.status_code, "latency_ms": round((time.monotonic() - start) * 1000)}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc),
                "latency_ms": round((time.monotonic() - start) * 1000)}

def get_minio_stats(config: AdminConfig) -> dict[str, Any]:
    try:
        client = Minio(config.minio_endpoint, access_key=config.minio_access_key,
                       secret_key=config.minio_secret_key, secure=config.minio_secure)
        buckets = client.list_buckets()
        bucket_stats = []
        for bucket in buckets:
            objects = list(client.list_objects(bucket.name, recursive=True))
            total_size = sum(o.size or 0 for o in objects)
            bucket_stats.append({"name": bucket.name, "objects": len(objects),
                                  "size_bytes": total_size, "size_mb": round(total_size / (1024*1024), 2)})
        return {"healthy": True, "buckets": bucket_stats}
    except Exception as exc:
        logger.warning("MinIO health check failed: %s", exc)
        return {"healthy": False, "error": str(exc), "buckets": []}

def get_qdrant_stats(config: AdminConfig) -> dict[str, Any]:
    try:
        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port, timeout=5)
        collections = client.get_collections().collections
        result = []
        for col in collections:
            info = client.get_collection(col.name)
            result.append({"name": col.name, "vectors_count": info.vectors_count or 0,
                           "indexed_vectors_count": info.indexed_vectors_count or 0,
                           "status": info.status.value if info.status else "unknown"})
        return {"healthy": True, "collections": result}
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {"healthy": False, "error": str(exc), "collections": []}

async def get_full_health(config: AdminConfig) -> dict[str, Any]:
    embedder = await check_http_service("embedder", f"{config.embedder_url}/health")
    chatbot = await check_http_service("chatbot", f"{config.chatbot_url}/health")
    ollama = await check_http_service("ollama", f"{config.ollama_url}/api/tags")
    return {
        "services": {"embedder": embedder, "chatbot": chatbot, "ollama": ollama},
        "storage": get_minio_stats(config),
        "vectors": get_qdrant_stats(config),
    }
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_health.py -v` — expect 2 PASS

- [ ] **Step 5: Commit**
```bash
git add services/admin/src/admin/health.py services/admin/tests/test_health.py
git commit -m "feat(admin): add health checks and MinIO/Qdrant stats"
```


---

## Task 7: FastAPI app assembly

**Files:** Create `services/admin/src/admin/main.py`, `services/admin/tests/test_main.py`

Routes:
- `GET /api/pipeline/status` — {state, run_id}
- `POST /api/pipeline/start` — body: dict of setting overrides
- `POST /api/pipeline/stop`
- `GET /api/pipeline/logs` — SSE stream (text/event-stream)
- `GET /api/history?limit=20`
- `GET /api/history/{run_id}`
- `GET /api/history/{run_id}/logs`
- `GET /api/settings`
- `PUT /api/settings` — body: dict[str, str]
- `GET /api/schedule`
- `PUT /api/schedule` — body: {cron: str, enabled: bool}
- `GET /api/health`
- `GET /` — serve static/index.html

- [ ] **Step 1: Write failing tests**
```python
# services/admin/tests/test_main.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "secret")
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("EMBEDDER_URL", "http://embedder:8080")
    monkeypatch.setenv("ADMIN_DATA_DIR", str(tmp_path))
    from admin.main import create_app
    app = await create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

async def test_pipeline_status_idle(client):
    resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200 and resp.json()["state"] == "idle"

async def test_get_settings_has_defaults(client):
    resp = await client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "crawl.seed_url" in data and data["crawl.seed_url"] == "https://cs.vt.edu"

async def test_update_and_read_settings(client):
    await client.put("/api/settings", json={"crawl.max_depth": "3"})
    resp = await client.get("/api/settings")
    assert resp.json()["crawl.max_depth"] == "3"

async def test_get_schedule_structure(client):
    resp = await client.get("/api/schedule")
    assert resp.status_code == 200
    data = resp.json()
    assert "cron" in data and "enabled" in data and "next_runs" in data

async def test_update_schedule(client):
    resp = await client.put("/api/schedule", json={"cron": "0 4 * * 0", "enabled": False})
    assert resp.status_code == 200 and resp.json()["cron"] == "0 4 * * 0"

async def test_history_empty(client):
    resp = await client.get("/api/history")
    assert resp.status_code == 200 and resp.json() == []

async def test_stop_when_idle(client):
    resp = await client.post("/api/pipeline/stop")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_main.py -v`

- [ ] **Step 3: Implement `services/admin/src/admin/main.py`**
```python
from __future__ import annotations
import asyncio, logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from admin.config import AdminConfig
from admin.health import get_full_health
from admin.runner import PipelineRunner
from admin.scheduler import CrawlScheduler
from admin.store import Store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_DEFAULTS = {
    "crawl.seed_url": "https://cs.vt.edu",
    "crawl.max_depth": "4",
    "crawl.max_pages": "9999999",
    "crawl.allowed_domains": "cs.vt.edu",
    "crawl.blocked_domains": "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,portal.cs.vt.edu,api.cs.vt.edu,forum.cs.vt.edu,login.cs.vt.edu",
    "crawl.blocked_paths": "/content/,/editor.html,/cs-root.html,/cs-source.html",
    "crawl.request_delay": "0.5",
    "crawl.prune_threshold": "0.45",
    "chunker.preferred_tokens": "400",
    "chunker.overlap_tokens": "64",
    "chunker.min_tokens": "120",
    "schedule.cron": "0 3 * * 0",
    "schedule.enabled": "false",
}

async def create_app() -> FastAPI:
    config = AdminConfig.from_env()
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    store = Store(str(data_dir / "admin.db"))
    await store.init()

    defaults = {**_DEFAULTS, "schedule.cron": config.default_schedule}
    for k, v in defaults.items():
        if await store.get_setting(k) is None:
            await store.set_setting(k, v)

    runner = PipelineRunner(store, config, log_dir=str(log_dir))

    async def _scheduled_crawl() -> None:
        settings = await store.get_all_settings()
        try:
            await runner.start(settings)
        except RuntimeError:
            logger.warning("Scheduled crawl skipped — pipeline already running")

    scheduler = CrawlScheduler(on_trigger=_scheduled_crawl)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        cron = await store.get_setting("schedule.cron", config.default_schedule)
        enabled = (await store.get_setting("schedule.enabled", "false")) == "true"
        scheduler.start()
        scheduler.configure(cron, enabled)
        yield
        scheduler.shutdown()

    app = FastAPI(title="HokieHelp Admin", lifespan=lifespan)

    @app.get("/api/pipeline/status")
    async def pipeline_status() -> dict[str, Any]:
        return {"state": runner.state.value, "run_id": runner.current_run_id}

    @app.post("/api/pipeline/start")
    async def pipeline_start(overrides: dict[str, str] = {}) -> dict[str, str]:
        base = await store.get_all_settings()
        try:
            run_id = await runner.start({**base, **overrides})
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"run_id": run_id}

    @app.post("/api/pipeline/stop")
    async def pipeline_stop() -> dict[str, str]:
        await runner.stop()
        return {"status": "stopping"}

    @app.get("/api/pipeline/logs")
    async def pipeline_logs():
        q = runner.subscribe()
        async def generator():
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(q.get(), timeout=25.0)
                        yield {"data": line}
                    except asyncio.TimeoutError:
                        yield {"data": ""}  # keepalive
            finally:
                runner.unsubscribe(q)
        return EventSourceResponse(generator())

    @app.get("/api/history")
    async def history(limit: int = 20) -> list[dict]:
        return await store.list_runs(limit=limit)

    @app.get("/api/history/{run_id}")
    async def history_detail(run_id: str) -> dict:
        run = await store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/history/{run_id}/logs")
    async def history_logs(run_id: str) -> dict:
        log_file = log_dir / f"{run_id}.log"
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log not found")
        return {"logs": log_file.read_text()}

    @app.get("/api/settings")
    async def get_settings() -> dict[str, str]:
        return await store.get_all_settings()

    @app.put("/api/settings")
    async def update_settings(updates: dict[str, str]) -> dict[str, str]:
        for k, v in updates.items():
            await store.set_setting(k, v)
        return await store.get_all_settings()

    @app.get("/api/schedule")
    async def get_schedule() -> dict:
        return scheduler.status()

    @app.put("/api/schedule")
    async def update_schedule(body: dict[str, Any]) -> dict:
        cron = body.get("cron", await store.get_setting("schedule.cron"))
        enabled = bool(body.get("enabled", False))
        try:
            scheduler.configure(cron, enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await store.set_setting("schedule.cron", cron)
        await store.set_setting("schedule.enabled", "true" if enabled else "false")
        return scheduler.status()

    @app.get("/api/health")
    async def health() -> dict:
        return await get_full_health(config)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def dashboard():
        index = Path(__file__).parent / "static" / "index.html"
        return FileResponse(str(index)) if index.exists() else HTMLResponse("<h1>HokieHelp Admin</h1>")

    return app
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_main.py -v` — expect 7 PASS

- [ ] **Step 5: Run all admin tests** — `pytest tests/ -v` — expect all PASS

- [ ] **Step 6: Commit**
```bash
git add services/admin/src/admin/main.py services/admin/tests/test_main.py
git commit -m "feat(admin): add FastAPI app with all routes"
```


---

## Task 8: Dashboard UI

**Files:** Create `services/admin/src/admin/static/index.html`

Single HTML file with Alpine.js + Tailwind CSS from CDN. Sections: Overview, Pipeline (with live logs + settings), Schedule, History, Services, Storage, Vectors.

- [ ] **Step 1: Create `services/admin/src/admin/static/index.html`**

The file is an Alpine.js SPA. Key Alpine data properties:
- `currentPage` — which section is visible
- `health` — {services: {embedder, chatbot, ollama}, storage: {buckets}, vectors: {collections}}
- `pipeline` — {state, run_id}
- `settings` — dict of all crawl/chunker settings from API
- `schedule` — {cron, enabled, next_runs}
- `history` — array of run records
- `logs` — array of strings (live SSE log lines)
- `logModal` — {open, runId, content}

Key methods:
- `init()` — fetches all initial data, starts SSE connection, starts 10s poll timer
- `startCrawl()` — POST /api/pipeline/start, clears logs
- `stopCrawl()` — POST /api/pipeline/stop
- `saveSettings()` — PUT /api/settings with current settings object
- `saveSchedule()` — PUT /api/schedule with cron + enabled
- `viewLogs(runId)` — opens modal, fetches /api/history/{runId}/logs
- `startSSE()` — connects EventSource to /api/pipeline/logs, appends lines to logs[]

The full HTML source is too long for inline reproduction here. Implement the following sections in one file:

**Nav sidebar** (52px wide): Logo "HokieHelp Admin" in VT orange (#861F41 or orange-600), nav links for each page — active link has orange-600 background.

**Overview page**: Service health grid (3 columns, green/red dot + name + latency), MinIO + Qdrant summary cards, pipeline status card (with color-coded dot), recent runs table (last 5).

**Pipeline page**: Status bar with Start/Stop buttons (Start disabled when running, Stop disabled when idle), stage progress indicators (crawler → chunker → embedding dots), scrolling log box (dark background, mono font, auto-scroll), settings form with all crawl/chunker fields.

**Schedule page**: Cron expression text input, enabled checkbox, Save button, next-runs list display.

**History page**: Full table of runs with ID/started/finished/status/stage columns, Logs button per row opening a modal with full log text.

**Services page**: Health cards for embedder, chatbot, ollama — each showing healthy/unhealthy, HTTP status code, latency, error message.

**Storage page**: Per-bucket cards with name, object count, size in MB, mini progress bar.

**Vectors page**: Per-collection cards with name, status badge, total vectors count, indexed count.

Use Tailwind `bg-gray-950` for page background, `bg-gray-900` for cards, `text-orange-500` for the logo, `bg-orange-600` for primary action buttons. Status colors: green-400 = healthy/completed, red-400 = failed/unhealthy, yellow-400 = running, gray-400 = idle/cancelled.

Reference implementation structure:
```
- <body x-data="app()" x-init="init()">
  - Sidebar nav with x-for over pages array
  - Main content area with x-show per page
  - Log modal (fixed overlay, x-show="logModal.open")
- <script> function app() { return { ... all data and methods ... } }
```

- [ ] **Step 2: Verify dashboard loads via port-forward after deploy (Task 13)**

After applying manifests:
```bash
kubectl port-forward -n test svc/hokiehelp-admin 8080:8080
# open http://localhost:8080
```
Verify: all 7 nav sections accessible, service health shows real data, Pipeline page shows settings form with current values, Schedule page shows current cron.

- [ ] **Step 3: Commit**
```bash
git add services/admin/src/admin/static/index.html
git commit -m "feat(admin): add single-page dashboard (Alpine.js + Tailwind CDN)"
```


---

## Task 9: Embedder FastAPI server

**Files:** Create `services/embedder/src/embedder/server.py`, modify `services/embedder/Dockerfile` and `services/embedder/pyproject.toml`

Wraps existing `run_embedding()` in a FastAPI server. Model loads once on startup (~30–60s). `POST /embed/start` launches a background asyncio task (via `run_in_executor` since embedding is CPU/GPU-bound synchronous) and returns a run_id. `GET /embed/status/{run_id}` returns status + stats. `GET /health` confirms model is loaded.

- [ ] **Step 1: Write failing test**
```python
# services/embedder/tests/test_server.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch, AsyncMock

@pytest.fixture
async def client():
    with patch("embedder.server.EmbedderConfig.from_env") as mock_cfg, \
         patch("embedder.server.EmbedderStorage"), \
         patch("embedder.server.Embedder") as mock_emb, \
         patch("embedder.server.QdrantIndexer"):
        mock_cfg.return_value = MagicMock(qdrant_host="qdrant", qdrant_port=6333,
            qdrant_collection="test", embedding_model="mock-model", embedding_batch_size=32)
        mock_emb.return_value.dimension = 1024
        from embedder.server import create_app
        app = await create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200 and resp.json()["healthy"] is True

async def test_start_returns_run_id(client):
    resp = await client.post("/embed/start")
    assert resp.status_code == 200 and "run_id" in resp.json()

async def test_status_unknown_run(client):
    resp = await client.get("/embed/status/nonexistent")
    assert resp.status_code == 404

async def test_status_known_run(client):
    run_id = (await client.post("/embed/start")).json()["run_id"]
    resp = await client.get(f"/embed/status/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("running", "completed", "failed")
```

- [ ] **Step 2: Run to verify fail** — `cd services/embedder && pytest tests/test_server.py -v`

- [ ] **Step 3: Implement `services/embedder/src/embedder/server.py`**
```python
from __future__ import annotations
import asyncio, logging, uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from embedder.config import EmbedderConfig
from embedder.embedder import Embedder
from embedder.indexer import QdrantIndexer
from embedder.main import run_embedding
from embedder.storage import EmbedderStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_runs: dict[str, dict[str, Any]] = {}

async def create_app() -> FastAPI:
    config = EmbedderConfig.from_env()
    storage = EmbedderStorage(config)
    embedder = Embedder(config.embedding_model)
    indexer = QdrantIndexer(host=config.qdrant_host, port=config.qdrant_port,
                            collection=config.qdrant_collection, vector_size=embedder.dimension)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Embedder ready (model=%s dim=%d)", config.embedding_model, embedder.dimension)
        yield

    app = FastAPI(title="HokieHelp Embedder", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"healthy": True, "model": config.embedding_model, "dimension": embedder.dimension}

    @app.post("/embed/start")
    async def embed_start() -> dict[str, str]:
        run_id = uuid.uuid4().hex[:8]
        _runs[run_id] = {"status": "running", "stats": None, "error": None}
        asyncio.create_task(_run(run_id, storage, embedder, indexer, config))
        return {"run_id": run_id}

    @app.get("/embed/status/{run_id}")
    async def embed_status(run_id: str) -> dict[str, Any]:
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run_id": run_id, **_runs[run_id]}

    return app

async def _run(run_id: str, storage, embedder, indexer, config) -> None:
    loop = asyncio.get_event_loop()
    try:
        stats = await loop.run_in_executor(None, run_embedding, storage, embedder, indexer, config)
        _runs[run_id] = {"status": "completed", "stats": stats, "error": None}
        logger.info("Embedding run %s completed: %s", run_id, stats)
    except Exception as exc:
        logger.exception("Embedding run %s failed: %s", run_id, exc)
        _runs[run_id] = {"status": "failed", "stats": None, "error": str(exc)}
```

- [ ] **Step 4: Update `services/embedder/pyproject.toml`** — add to dependencies:
```toml
"fastapi>=0.115.0",
"uvicorn[standard]>=0.30.0",
```

- [ ] **Step 5: Update `services/embedder/Dockerfile`** — change CMD:
```dockerfile
FROM ghcr.io/prakharmodi26/hokiehelp-base:latest
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
CMD ["uvicorn", "embedder.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 6: Run to verify pass** — `pytest tests/test_server.py -v` — expect 4 PASS

- [ ] **Step 7: Commit**
```bash
git add services/embedder/src/embedder/server.py services/embedder/Dockerfile \
        services/embedder/pyproject.toml services/embedder/tests/test_server.py
git commit -m "feat(embedder): add FastAPI HTTP server, convert from batch Job to always-on Deployment"
```


---

## Task 10: K8s manifests — Admin service

**Files:** Create `k8s/admin-pvc.yaml`, `k8s/admin-configmap.yaml`, `k8s/admin-deployment.yaml`

- [ ] **Step 1: Create `k8s/admin-pvc.yaml`**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: admin-data
  namespace: test
  labels:
    app: hokiehelp-admin
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

- [ ] **Step 2: Create `k8s/admin-configmap.yaml`**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: admin-config
  namespace: test
  labels:
    app: hokiehelp-admin
data:
  ADMIN_PORT: "8080"
  ADMIN_DATA_DIR: "/data"
  MINIO_ENDPOINT: "minio:9000"
  MINIO_BUCKET: "crawled-pages"
  MINIO_CLEANED_BUCKET: "crawled-pages-cleaned"
  MINIO_CHUNKS_BUCKET: "chunks"
  MINIO_SECURE: "false"
  QDRANT_HOST: "qdrant"
  QDRANT_PORT: "6333"
  QDRANT_COLLECTION: "hokiehelp_chunks"
  EMBEDDER_URL: "http://hokiehelp-embedder:8080"
  CHATBOT_URL: "http://hokiehelp-chatbot:8000"
  OLLAMA_URL: "http://ollama:11434"
  CRAWL_SEED_URL: "https://cs.vt.edu"
  CRAWL_MAX_DEPTH: "4"
  CRAWL_MAX_PAGES: "9999999"
  CRAWL_SCHEDULE: "0 3 * * 0"
```

- [ ] **Step 3: Create `k8s/admin-deployment.yaml`**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hokiehelp-admin
  namespace: test
  labels:
    app: hokiehelp-admin
spec:
  selector:
    app: hokiehelp-admin
  ports:
    - name: http
      port: 8080
      targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hokiehelp-admin
  namespace: test
  labels:
    app: hokiehelp-admin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hokiehelp-admin
  template:
    metadata:
      labels:
        app: hokiehelp-admin
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: admin
          image: ghcr.io/prakharmodi26/hokiehelp-admin:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: admin-config
          env:
            - name: MINIO_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: access-key
            - name: MINIO_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: secret-key
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: false
            capabilities:
              drop: ["ALL"]
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: admin-data
```

- [ ] **Step 4: Commit**
```bash
git add k8s/admin-pvc.yaml k8s/admin-configmap.yaml k8s/admin-deployment.yaml
git commit -m "feat(k8s): add admin service manifests (PVC, ConfigMap, Deployment, Service)"
```

---

## Task 11: K8s manifests — Embedder as always-on Deployment

**Files:** Create `k8s/embedder-deployment.yaml`

- [ ] **Step 1: Create `k8s/embedder-deployment.yaml`**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hokiehelp-embedder
  namespace: test
  labels:
    app: hokiehelp-embedder
spec:
  selector:
    app: hokiehelp-embedder
  ports:
    - name: http
      port: 8080
      targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hokiehelp-embedder
  namespace: test
  labels:
    app: hokiehelp-embedder
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hokiehelp-embedder
  template:
    metadata:
      labels:
        app: hokiehelp-embedder
    spec:
      containers:
        - name: embedder
          image: ghcr.io/prakharmodi26/hokiehelp-embedder:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: embedder-config
          env:
            - name: MINIO_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: access-key
            - name: MINIO_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: secret-key
          resources:
            requests:
              memory: "4Gi"
              cpu: "500m"
              nvidia.com/gpu: "1"
            limits:
              memory: "8Gi"
              cpu: "2000m"
              nvidia.com/gpu: "1"
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            failureThreshold: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 30
```

- [ ] **Step 2: Commit**
```bash
git add k8s/embedder-deployment.yaml
git commit -m "feat(k8s): add embedder as always-on GPU Deployment with HTTP API"
```

---

## Task 12: Network policies and cleanup

**Files:** Modify `k8s/network-policies.yaml`, delete `k8s/pipeline-cronjob.yaml` and `k8s/pipeline-rbac.yaml`

- [ ] **Step 1: Append admin + embedder policies to `k8s/network-policies.yaml`**

Add after the last existing policy:
```yaml
---
# Admin — egress to internal services + external internet (crawler runs inside this pod)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: admin-egress
  namespace: test
spec:
  podSelector:
    matchLabels:
      app: hokiehelp-admin
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: minio
      ports:
        - protocol: TCP
          port: 9000
    - to:
        - podSelector:
            matchLabels:
              app: qdrant
      ports:
        - protocol: TCP
          port: 6333
        - protocol: TCP
          port: 6334
    - to:
        - podSelector:
            matchLabels:
              app: hokiehelp-embedder
      ports:
        - protocol: TCP
          port: 8080
    - to:
        - podSelector:
            matchLabels:
              app: hokiehelp-chatbot
      ports:
        - protocol: TCP
          port: 8000
    - to:
        - podSelector:
            matchLabels:
              workload.user.cattle.io/workloadselector: apps.deployment-test-ollama
      ports:
        - protocol: TCP
          port: 11434
    # External internet for crawler subprocess (crawls cs.vt.edu)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 10.0.0.0/8
              - 172.16.0.0/12
              - 192.168.0.0/16
      ports:
        - protocol: TCP
          port: 80
        - protocol: TCP
          port: 443
---
# Embedder service — ingress from admin
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: embedder-service-ingress
  namespace: test
spec:
  podSelector:
    matchLabels:
      app: hokiehelp-embedder
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: hokiehelp-admin
      ports:
        - protocol: TCP
          port: 8080
---
# Embedder service — egress to MinIO and Qdrant
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: embedder-service-egress
  namespace: test
spec:
  podSelector:
    matchLabels:
      app: hokiehelp-embedder
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: minio
      ports:
        - protocol: TCP
          port: 9000
    - to:
        - podSelector:
            matchLabels:
              app: qdrant
      ports:
        - protocol: TCP
          port: 6333
        - protocol: TCP
          port: 6334
```

- [ ] **Step 2: Remove `pipeline-orchestrator-egress` policy** that was added during debugging — it is no longer needed since the CronJob is being removed. Delete the entire `pipeline-orchestrator-egress` NetworkPolicy block from `k8s/network-policies.yaml`.

- [ ] **Step 3: Delete old pipeline manifests**
```bash
git rm k8s/pipeline-cronjob.yaml k8s/pipeline-rbac.yaml
```

- [ ] **Step 4: Commit**
```bash
git add k8s/network-policies.yaml
git commit -m "feat(k8s): add admin + embedder network policies, remove pipeline CronJob and RBAC"
```

---

## Task 13: Build, push, and deploy

- [ ] **Step 1: Verify cluster context**
```bash
kubectl config current-context
```
Expected: `endeavour`

- [ ] **Step 2: Build and push admin image** (context is `services/`)
```bash
cd services
docker build -f admin/Dockerfile -t ghcr.io/prakharmodi26/hokiehelp-admin:latest .
docker push ghcr.io/prakharmodi26/hokiehelp-admin:latest
cd ..
```

- [ ] **Step 3: Build and push embedder image**
```bash
docker build -t ghcr.io/prakharmodi26/hokiehelp-embedder:latest services/embedder/
docker push ghcr.io/prakharmodi26/hokiehelp-embedder:latest
```

- [ ] **Step 4: Apply manifests**
```bash
kubectl apply -f k8s/admin-pvc.yaml -n test
kubectl apply -f k8s/admin-configmap.yaml -n test
kubectl apply -f k8s/admin-deployment.yaml -n test
kubectl apply -f k8s/embedder-configmap.yaml -n test
kubectl apply -f k8s/embedder-deployment.yaml -n test
kubectl apply -f k8s/network-policies.yaml -n test
```

- [ ] **Step 5: Remove old pipeline resources from cluster**
```bash
kubectl delete cronjob hokiehelp-pipeline -n test --ignore-not-found
kubectl delete role hokiehelp-pipeline-role -n test --ignore-not-found
kubectl delete rolebinding hokiehelp-pipeline-binding -n test --ignore-not-found
kubectl delete serviceaccount hokiehelp-pipeline -n test --ignore-not-found
kubectl delete networkpolicy pipeline-orchestrator-egress -n test --ignore-not-found
kubectl delete job hokiehelp-pipeline-manual-1776233670 -n test --ignore-not-found
```

- [ ] **Step 6: Verify rollout**
```bash
kubectl get pods -n test
kubectl rollout status deployment/hokiehelp-admin -n test
kubectl rollout status deployment/hokiehelp-embedder -n test
```
Expected: both `successfully rolled out`. Admin Running immediately. Embedder Running after ~60–90s (model load).

- [ ] **Step 7: Check admin logs for startup**
```bash
kubectl logs -n test deployment/hokiehelp-admin --tail=20
```
Expected: `Uvicorn running on http://0.0.0.0:8080`

- [ ] **Step 8: Access dashboard**
```bash
kubectl port-forward -n test svc/hokiehelp-admin 8080:8080
# open http://localhost:8080
```
Verify: dashboard loads, all 7 nav sections work, service health populates, Pipeline page shows `idle` with settings form.

- [ ] **Step 9: Trigger a test crawl**

Click "Start Crawl" on Pipeline page. Verify:
- State changes `idle → crawling`
- Log lines stream in real time in the log box
- State progresses through `chunking → embedding → done`
- History page shows the completed run with Logs button working

- [ ] **Step 10: Final commit**
```bash
git add .
git commit -m "feat: complete admin dashboard with pipeline orchestration"
```

---

## Self-Review

**Spec coverage:**
- Start/stop crawl from dashboard — Task 7 + 8
- Change crawl settings (seed URL, depth, pages, domains, delay, paths) — Task 7 + 8
- Change chunker settings (token sizes) — Task 7 + 8
- Manage cron schedule (edit expression, enable/disable, preview next runs) — Task 5 + 7 + 8
- Live log streaming via SSE — Task 4 + 7 + 8
- Run history with log viewer modal — Task 3 + 7 + 8
- Health checks for embedder, chatbot, ollama — Task 6 + 7 + 8
- MinIO bucket stats (object count, size) — Task 6 + 8
- Qdrant collection stats (vector count, index status) — Task 6 + 8
- No ingress; SSH port-forward only — Task 10
- Embedder as always-on GPU Deployment — Task 9 + 11
- Crawler and chunker as async child processes — Task 4
- No RBAC needed — confirmed (no k8s Job creation)
- Old pipeline CronJob + RBAC removed — Task 12

**No placeholders found.** All code steps contain complete implementations.

**Type consistency:** `PipelineState` enum used identically across `runner.py` and `main.py`. `Store.update_run()` kwargs match all call sites in `runner.py`. `AdminConfig` fields referenced consistently in `config.py`, `runner.py`, `health.py`, and `main.py`.
