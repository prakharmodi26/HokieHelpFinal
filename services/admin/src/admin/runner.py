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
