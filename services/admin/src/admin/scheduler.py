from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from croniter import croniter, CroniterBadCronError

logger = logging.getLogger(__name__)
_JOB_PREFIX = "sched_"


class CrawlScheduler:
    def __init__(self, on_trigger: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._scheduler = AsyncIOScheduler()
        self._on_trigger = on_trigger
        self._schedules: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)

    @staticmethod
    def validate_cron(cron: str) -> None:
        try:
            croniter(cron)
        except (CroniterBadCronError, ValueError) as exc:
            raise ValueError(f"Invalid cron expression '{cron}': {exc}") from exc

    @staticmethod
    def _next_runs(cron: str, n: int = 5) -> list[datetime]:
        it = croniter(cron, datetime.now(timezone.utc))
        return [it.get_next(datetime) for _ in range(n)]

    def sync(self, schedules: list[dict[str, Any]]) -> None:
        """Replace registered jobs to match given schedule list."""
        for job in list(self._scheduler.get_jobs()):
            if job.id.startswith(_JOB_PREFIX):
                self._scheduler.remove_job(job.id)
        self._schedules = {}
        for sch in schedules:
            self._schedules[sch["id"]] = sch
            if not sch.get("enabled"):
                continue
            try:
                self.validate_cron(sch["cron"])
            except ValueError as exc:
                logger.warning("Skipping schedule %s — %s", sch.get("name"), exc)
                continue
            sch_copy = dict(sch)
            self._scheduler.add_job(
                self._on_trigger,
                CronTrigger.from_crontab(sch["cron"]),
                id=f"{_JOB_PREFIX}{sch['id']}",
                kwargs={"schedule": sch_copy},
                replace_existing=True,
            )
            logger.info("Schedule registered: %s (%s)", sch["name"], sch["cron"])

    def status(self) -> list[dict[str, Any]]:
        out = []
        for sid, sch in self._schedules.items():
            entry = {**sch}
            if sch.get("enabled"):
                try:
                    entry["next_runs"] = [r.isoformat() for r in self._next_runs(sch["cron"])]
                except Exception:
                    entry["next_runs"] = []
            else:
                entry["next_runs"] = []
            out.append(entry)
        return out
