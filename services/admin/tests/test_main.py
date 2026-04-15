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
    app = create_app()
    async with app.router.lifespan_context(app):
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
