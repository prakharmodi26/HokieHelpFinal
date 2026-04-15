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
