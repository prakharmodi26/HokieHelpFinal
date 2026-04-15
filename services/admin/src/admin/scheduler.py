from __future__ import annotations

import logging
from datetime import datetime, timezone
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
        it = croniter(cron, datetime.now(timezone.utc))
        return [it.get_next(datetime) for _ in range(n)]

    def configure(self, cron: str, enabled: bool) -> None:
        self._validate_cron(cron)
        self._cron = cron
        self._enabled = enabled
        if self._scheduler.get_job(_JOB_ID):
            self._scheduler.remove_job(_JOB_ID)
        if enabled:
            self._scheduler.add_job(
                self._on_trigger,
                CronTrigger.from_crontab(cron),
                id=_JOB_ID,
                replace_existing=True,
            )
            logger.info("Crawl scheduled: %s", cron)

    def status(self) -> dict:
        return {
            "cron": self._cron,
            "enabled": self._enabled,
            "next_runs": (
                [r.isoformat() for r in self._next_runs(self._cron)]
                if self._cron
                else []
            ),
        }
