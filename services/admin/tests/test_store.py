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
