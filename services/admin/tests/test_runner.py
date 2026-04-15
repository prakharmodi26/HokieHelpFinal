import pytest
from admin.runner import PipelineRunner, PipelineState
from admin.config import AdminConfig
from admin.store import Store

@pytest.fixture
async def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    await s.init()
    return s

@pytest.fixture
def cfg():
    return AdminConfig(
        port=8080, data_dir="/tmp",
        minio_endpoint="minio:9000", minio_access_key="key", minio_secret_key="secret",
        minio_secure=False, minio_bucket="crawled-pages", minio_cleaned_bucket="crawled-pages-cleaned",
        minio_chunks_bucket="chunks", qdrant_host="qdrant", qdrant_port=6333,
        qdrant_collection="test", embedder_url="http://embedder:8080",
        chatbot_url="http://chatbot:8000", ollama_url="http://ollama:11434",
        default_seed_url="https://cs.vt.edu", default_max_depth=2, default_max_pages=100,
        default_schedule="0 3 * * 0",
    )

async def test_initial_state(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    assert runner.state == PipelineState.IDLE
    assert runner.current_run_id is None

async def test_cannot_start_while_running(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    runner._state = PipelineState.CRAWLING
    with pytest.raises(RuntimeError, match="already running"):
        await runner.start({})

async def test_build_crawler_env(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    env = runner._build_crawler_env({"crawl.seed_url": "https://cs.vt.edu", "crawl.max_depth": "3", "crawl.max_pages": "500", "crawl.allowed_domains": "cs.vt.edu", "crawl.blocked_domains": "git.cs.vt.edu", "crawl.blocked_paths": "/content/", "crawl.request_delay": "0.5", "crawl.prune_threshold": "0.45"})
    assert env["MINIO_ENDPOINT"] == "minio:9000"
    assert env["CRAWL_SEED_URL"] == "https://cs.vt.edu"
    assert env["CRAWL_MAX_DEPTH"] == "3"

async def test_build_chunker_env(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    env = runner._build_chunker_env({"chunker.preferred_tokens": "400", "chunker.overlap_tokens": "64", "chunker.min_tokens": "120"})
    assert env["MINIO_ENDPOINT"] == "minio:9000"
    assert env["CHUNK_PREFERRED_TOKENS"] == "400"

async def test_subscribe_unsubscribe(store, cfg, tmp_path):
    runner = PipelineRunner(store, cfg, log_dir=str(tmp_path))
    q = runner.subscribe()
    assert q in runner._log_queues
    runner.unsubscribe(q)
    assert q not in runner._log_queues
