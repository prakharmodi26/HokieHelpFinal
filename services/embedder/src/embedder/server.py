from __future__ import annotations
import asyncio, logging, uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, Response
from embedder.config import EmbedderConfig
from embedder.embedder import Embedder
from embedder.indexer import QdrantIndexer
from embedder.main import run_embedding
from embedder.storage import EmbedderStorage
from embedder import logbuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logbuffer.install(level=logging.INFO)
logger = logging.getLogger(__name__)

_runs: dict[str, dict[str, Any]] = {}

def create_app() -> FastAPI:
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

    @app.get("/logs")
    async def logs(lines: int = 200) -> Response:
        return Response(content=logbuffer.get_logs(lines), media_type="text/plain")

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

async def _run(run_id: str, storage: EmbedderStorage, embedder: Embedder, indexer: QdrantIndexer, config: EmbedderConfig) -> None:
    loop = asyncio.get_event_loop()
    try:
        stats = await loop.run_in_executor(None, run_embedding, storage, embedder, indexer, config)
        _runs[run_id] = {"status": "completed", "stats": stats, "error": None}
        logger.info("Embedding run %s completed: %s", run_id, stats)
    except Exception as exc:
        logger.exception("Embedding run %s failed: %s", run_id, exc)
        _runs[run_id] = {"status": "failed", "stats": None, "error": str(exc)}
