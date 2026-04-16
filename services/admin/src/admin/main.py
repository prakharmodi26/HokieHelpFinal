from __future__ import annotations
import asyncio, logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from admin.config import AdminConfig
from admin.health import get_full_health, get_minio_stats
from admin.runner import PipelineRunner
from admin.scheduler import CrawlScheduler
from admin.store import Store
from admin import http_logs, logbuffer, storage_browser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logbuffer.install(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULTS = {
    "crawl.seed_url": "https://cs.vt.edu",
    "crawl.max_depth": "4",
    "crawl.max_pages": "9999999",
    "crawl.allowed_domains": "cs.vt.edu",
    "crawl.blocked_domains": "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,portal.cs.vt.edu,api.cs.vt.edu,forum.cs.vt.edu,login.cs.vt.edu",
    "crawl.blocked_paths": "/content/,/editor.html,/cs-root.html,/cs-source.html",
    "crawl.request_delay": "0.1",
    "crawl.prune_threshold": "0.45",
    "chunker.preferred_tokens": "400",
    "chunker.overlap_tokens": "64",
    "chunker.min_tokens": "120",
}

def create_app() -> FastAPI:
    config = AdminConfig.from_env()
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    store = Store(str(data_dir / "admin.db"))
    runner = PipelineRunner(store, config, log_dir=str(log_dir))

    async def _scheduled_crawl(schedule: dict[str, Any]) -> None:
        base = await store.get_all_settings()
        overrides = schedule.get("config") or {}
        merged = {**base, **{k: str(v) for k, v in overrides.items()}}
        logger.info("Scheduled crawl firing: %s (%s)", schedule.get("name"), schedule.get("id"))
        try:
            await runner.start(merged)
        except RuntimeError:
            logger.warning("Scheduled crawl skipped — pipeline already running")

    scheduler = CrawlScheduler(on_trigger=_scheduled_crawl)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await store.init()
        for k, v in _DEFAULTS.items():
            if await store.get_setting(k) is None:
                await store.set_setting(k, v)
        scheduler.start()
        scheduler.sync(await store.list_schedules())
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

    async def _resync() -> None:
        scheduler.sync(await store.list_schedules())

    @app.get("/api/schedules")
    async def list_schedules() -> dict:
        return {"schedules": scheduler.status()}

    @app.post("/api/schedules")
    async def create_schedule(body: dict[str, Any]) -> dict:
        name = (body.get("name") or "").strip()
        cron = (body.get("cron") or "").strip()
        if not name or not cron:
            raise HTTPException(status_code=400, detail="name and cron required")
        try:
            CrawlScheduler.validate_cron(cron)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        enabled = bool(body.get("enabled", False))
        cfg = body.get("config") or {}
        if not isinstance(cfg, dict):
            raise HTTPException(status_code=400, detail="config must be object")
        sch = await store.create_schedule(name, cron, enabled, {k: str(v) for k, v in cfg.items()})
        await _resync()
        return sch

    @app.put("/api/schedules/{schedule_id}")
    async def update_schedule(schedule_id: str, body: dict[str, Any]) -> dict:
        existing = await store.get_schedule(schedule_id)
        if not existing:
            raise HTTPException(status_code=404, detail="schedule not found")
        cron = body.get("cron")
        if cron is not None:
            try:
                CrawlScheduler.validate_cron(cron)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        cfg = body.get("config")
        if cfg is not None and not isinstance(cfg, dict):
            raise HTTPException(status_code=400, detail="config must be object")
        sch = await store.update_schedule(
            schedule_id,
            name=body.get("name"),
            cron=cron,
            enabled=body.get("enabled"),
            config={k: str(v) for k, v in cfg.items()} if cfg is not None else None,
        )
        await _resync()
        return sch or {}

    @app.delete("/api/schedules/{schedule_id}")
    async def delete_schedule(schedule_id: str) -> dict:
        ok = await store.delete_schedule(schedule_id)
        if not ok:
            raise HTTPException(status_code=404, detail="schedule not found")
        await _resync()
        return {"deleted": schedule_id}

    @app.get("/api/storage/buckets")
    async def storage_buckets() -> dict:
        return {"buckets": await storage_browser.list_buckets(config)}

    @app.get("/api/storage/browse")
    async def storage_browse(
        bucket: str,
        prefix: str = "",
        sort_by: str = "name",
        order: str = "asc",
    ) -> dict:
        if sort_by not in ("name", "date"):
            sort_by = "name"
        if order not in ("asc", "desc"):
            order = "asc"
        return await storage_browser.browse(config, bucket, prefix, sort_by, order)

    @app.get("/api/storage/download")
    async def storage_download(bucket: str, key: str):
        try:
            data, content_type, _ = await storage_browser.get_object(config, bucket, key)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        filename = key.rsplit("/", 1)[-1] or "file"
        return Response(
            content=data,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/health")
    async def health() -> dict:
        return await get_full_health(config)

    @app.get("/api/storage/stats")
    async def storage_stats() -> dict:
        return await get_minio_stats(config)

    @app.get("/api/services")
    async def services_list() -> dict:
        return {"services": await http_logs.list_services(config)}

    @app.get("/api/services/{service}/logs")
    async def service_logs(service: str, lines: int = 200) -> dict:
        return await http_logs.fetch_logs(config, service, lines=lines)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def dashboard():
        index = Path(__file__).parent / "static" / "index.html"
        return FileResponse(str(index)) if index.exists() else HTMLResponse("<h1>HokieHelp Admin</h1>")

    return app
