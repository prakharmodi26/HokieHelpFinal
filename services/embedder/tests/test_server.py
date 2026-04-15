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
