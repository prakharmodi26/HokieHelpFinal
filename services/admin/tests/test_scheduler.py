from datetime import datetime
import pytest
from admin.scheduler import CrawlScheduler


def test_next_runs_sunday():
    sched = CrawlScheduler.__new__(CrawlScheduler)
    runs = sched._next_runs("0 3 * * 0", n=3)
    assert len(runs) == 3
    for r in runs:
        assert isinstance(r, datetime)
        assert r.weekday() == 6 and r.hour == 3


def test_invalid_cron_raises():
    sched = CrawlScheduler.__new__(CrawlScheduler)
    with pytest.raises(ValueError, match="Invalid cron"):
        sched._validate_cron("not a cron expression")
