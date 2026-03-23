# Embedding & Indexing Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `services/embedder` microservice that reads chunk JSON from MinIO, validates chunks, generates embeddings with BAAI/bge-large-en-v1.5 via SentenceTransformers, and upserts them into Qdrant with stale-chunk cleanup.

**Architecture:** The service reads `chunks/<doc_id>.json` files from the MinIO `chunks` bucket (written by the chunker), validates each chunk, enriches the text with title/headings context, generates 1024-dim embeddings using SentenceTransformers locally, upserts into a Qdrant collection with cosine similarity, and deletes any stale chunks from Qdrant that no longer exist for a given document. Runs as a weekly K8s CronJob (Sunday 5 AM, one hour after chunker). Qdrant runs as a persistent Deployment in the same namespace.

**Tech Stack:** Python 3.11, `sentence-transformers>=2.2.0`, `qdrant-client>=1.7.0`, `minio>=7.2.0`, `pytest>=8.0`, Docker, Kubernetes, GitHub Actions

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/embedder/pyproject.toml` | CREATE | Package metadata, deps (minio, qdrant-client, sentence-transformers), CLI entrypoint |
| `services/embedder/Dockerfile` | CREATE | `python:3.11-slim`, CPU-only torch, pre-download model |
| `services/embedder/src/embedder/__init__.py` | CREATE | Empty package marker |
| `services/embedder/src/embedder/config.py` | CREATE | `EmbedderConfig` frozen dataclass, `from_env()` |
| `services/embedder/src/embedder/storage.py` | CREATE | Read chunk JSON files from MinIO |
| `services/embedder/src/embedder/validator.py` | CREATE | Validate chunk records, skip invalid |
| `services/embedder/src/embedder/embedder.py` | CREATE | SentenceTransformers model, context builder, batch embed |
| `services/embedder/src/embedder/indexer.py` | CREATE | Qdrant client, collection setup, upsert, stale deletion |
| `services/embedder/src/embedder/main.py` | CREATE | `run_embedding()` + `cli()` entrypoint |
| `services/embedder/tests/__init__.py` | CREATE | Empty |
| `services/embedder/tests/conftest.py` | CREATE | Shared fixtures |
| `services/embedder/tests/test_config.py` | CREATE | Config loading + validation |
| `services/embedder/tests/test_storage.py` | CREATE | Storage read with mocked MinIO |
| `services/embedder/tests/test_validator.py` | CREATE | Validation pass/fail/partial |
| `services/embedder/tests/test_embedder.py` | CREATE | Context building + mocked model |
| `services/embedder/tests/test_indexer.py` | CREATE | Qdrant operations with mocked client |
| `services/embedder/tests/test_main.py` | CREATE | End-to-end with all mocks |
| `k8s/qdrant-pvc.yaml` | CREATE | 5Gi PVC on ceph-rbd for Qdrant storage |
| `k8s/qdrant-deployment.yaml` | CREATE | Qdrant Deployment + Service (REST 6333, gRPC 6334) |
| `k8s/embedder-configmap.yaml` | CREATE | Non-secret env vars for embedder |
| `k8s/embedder-cronjob.yaml` | CREATE | Weekly CronJob, same minio-credentials Secret |
| `.github/workflows/embedder-ci.yaml` | CREATE | Test on PR, build+push to GHCR on main |

---

## Qdrant Collection Schema

```
Collection: hokiehelp_chunks
Vector size: 1024 (bge-large-en-v1.5 output dim)
Distance: Cosine
Payload indexes: document_id (keyword)

Point structure:
  id:      UUID5(chunk_id)        ← deterministic, idempotent
  vector:  float[1024]
  payload:
    chunk_id:        str           ← original string ID
    document_id:     str
    url:             str
    title:           str
    page_type:       str
    headings_path:   list[str]
    chunk_index:     int
    content_hash:    str
    crawl_timestamp: str
    token_count:     int
```

---

## Embedding Context Template

For each chunk, the embedding input is built as:

```
Title: Faculty | CS | VT
Section: Tenured Faculty
Path: People > Faculty > Tenured Faculty

## Tenured Faculty

Professor Smith researches distributed systems...
```

This gives the embedding model semantic context beyond the raw chunk text.

---

## Task 1: Service scaffold

**Files:**
- Create: `services/embedder/pyproject.toml`
- Create: `services/embedder/src/embedder/__init__.py`
- Create: `services/embedder/tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p services/embedder/src/embedder
mkdir -p services/embedder/tests
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "hokiehelp-embedder"
version = "0.1.0"
description = "Embedding and Qdrant indexing service for HokieHelp RAG pipeline"
requires-python = ">=3.11"
dependencies = [
    "minio>=7.2.0",
    "qdrant-client>=1.7.0",
    "sentence-transformers>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[project.scripts]
hokiehelp-embed = "embedder.main:cli"
```

- [ ] **Step 3: Create empty `__init__.py` files**

Both `services/embedder/src/embedder/__init__.py` and `services/embedder/tests/__init__.py` — empty files.

- [ ] **Step 4: Verify package installs**

```bash
cd services/embedder && pip install -e ".[dev]"
```

Expected: `Successfully installed hokiehelp-embedder-0.1.0` (plus torch, sentence-transformers, etc.)

Note: This will download PyTorch (~2GB) on first install. Subsequent installs are cached.

- [ ] **Step 5: Commit**

```bash
git add services/embedder/pyproject.toml services/embedder/src/embedder/__init__.py services/embedder/tests/__init__.py
git commit -m "feat: scaffold embedder service package"
```

---

## Task 2: Config module

**Files:**
- Create: `services/embedder/src/embedder/config.py`
- Create: `services/embedder/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

File: `services/embedder/tests/test_config.py`

```python
"""Tests for EmbedderConfig."""
import pytest
from embedder.config import EmbedderConfig


def _base_env():
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "QDRANT_HOST": "localhost",
    }


def test_from_env_loads_required(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    assert cfg.minio_endpoint == "localhost:9000"
    assert cfg.qdrant_host == "localhost"


def test_from_env_defaults(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    assert cfg.minio_chunks_bucket == "chunks"
    assert cfg.minio_secure is False
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp_chunks"
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.embedding_batch_size == 32


def test_from_env_overrides(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("QDRANT_PORT", "7777")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    cfg = EmbedderConfig.from_env()
    assert cfg.qdrant_port == 7777
    assert cfg.embedding_batch_size == 64
    assert cfg.qdrant_collection == "test_chunks"


def test_from_env_missing_required_raises(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        EmbedderConfig.from_env()


def test_config_is_frozen(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = EmbedderConfig.from_env()
    with pytest.raises((AttributeError, TypeError)):
        cfg.minio_endpoint = "other"  # type: ignore
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `config.py`**

```python
"""Configuration loaded from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderConfig:
    """Immutable embedder configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_chunks_bucket: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model: str
    embedding_batch_size: int

    @classmethod
    def from_env(cls) -> EmbedderConfig:
        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required environment variable {name} is not set")
            return val

        return cls(
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            qdrant_host=_require("QDRANT_HOST"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
        )
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/embedder && python -m pytest tests/test_config.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/config.py services/embedder/tests/test_config.py
git commit -m "feat: add embedder config module"
```

---

## Task 3: Storage layer (read chunks from MinIO)

**Files:**
- Create: `services/embedder/tests/conftest.py`
- Create: `services/embedder/src/embedder/storage.py`
- Create: `services/embedder/tests/test_storage.py`

- [ ] **Step 1: Write `conftest.py`**

```python
"""Shared test fixtures for the embedder service."""
import pytest
from embedder.config import EmbedderConfig


@pytest.fixture
def embedder_config():
    return EmbedderConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        minio_chunks_bucket="chunks",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_chunks",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_batch_size=32,
    )
```

- [ ] **Step 2: Write failing tests**

File: `services/embedder/tests/test_storage.py`

```python
"""Tests for EmbedderStorage using mocked MinIO client."""
import json
from unittest.mock import MagicMock, patch
import pytest

from embedder.storage import EmbedderStorage


@pytest.fixture
def mock_minio():
    client = MagicMock()
    client.bucket_exists.return_value = True
    return client


@pytest.fixture
def storage(embedder_config, mock_minio):
    with patch("embedder.storage.Minio", return_value=mock_minio):
        s = EmbedderStorage(embedder_config)
    s._client = mock_minio
    return s


SAMPLE_CHUNKS = [
    {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Hello world",
        "url": "https://example.com",
        "title": "Test",
        "page_type": "general",
        "headings_path": [],
        "content_hash": "deadbeef01234567",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 2,
    }
]


def test_list_chunk_keys(storage, mock_minio):
    obj1 = MagicMock()
    obj1.object_name = "chunks/abc.json"
    obj2 = MagicMock()
    obj2.object_name = "chunks/def.json"
    obj3 = MagicMock()
    obj3.object_name = "other/file.txt"
    mock_minio.list_objects.return_value = [obj1, obj2, obj3]
    keys = storage.list_chunk_keys()
    assert keys == ["chunks/abc.json", "chunks/def.json"]


def test_download_chunks(storage, mock_minio):
    data = json.dumps(SAMPLE_CHUNKS).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = data
    mock_minio.get_object.return_value = resp
    chunks = storage.download_chunks("chunks/abc.json")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "abc_0000"
    resp.close.assert_called_once()
    resp.release_conn.assert_called_once()


def test_download_chunks_empty_file(storage, mock_minio):
    resp = MagicMock()
    resp.read.return_value = b"[]"
    mock_minio.get_object.return_value = resp
    chunks = storage.download_chunks("chunks/empty.json")
    assert chunks == []
```

- [ ] **Step 3: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_storage.py -v
```

- [ ] **Step 4: Implement `storage.py`**

```python
"""MinIO storage layer — reads chunk JSON files from the chunks bucket."""
from __future__ import annotations

import json
import logging
from typing import List

from minio import Minio

from embedder.config import EmbedderConfig

logger = logging.getLogger(__name__)


class EmbedderStorage:
    """Reads chunk JSON files from MinIO."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._bucket = config.minio_chunks_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )

    def list_chunk_keys(self) -> List[str]:
        """Return all chunks/*.json object keys."""
        return [
            obj.object_name
            for obj in self._client.list_objects(self._bucket, prefix="chunks/", recursive=True)
            if obj.object_name.endswith(".json")
        ]

    def download_chunks(self, key: str) -> List[dict]:
        """Download and parse a chunk JSON file. Returns list of chunk dicts."""
        response = self._client.get_object(self._bucket, key)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
```

- [ ] **Step 5: Run to confirm PASS**

```bash
cd services/embedder && python -m pytest tests/test_storage.py -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add services/embedder/tests/conftest.py services/embedder/src/embedder/storage.py services/embedder/tests/test_storage.py
git commit -m "feat: add embedder storage layer"
```

---

## Task 4: Chunk validator

**Files:**
- Create: `services/embedder/src/embedder/validator.py`
- Create: `services/embedder/tests/test_validator.py`

- [ ] **Step 1: Write failing tests**

File: `services/embedder/tests/test_validator.py`

```python
"""Tests for chunk validation."""
import pytest
from embedder.validator import validate_chunks, ValidationResult


def _valid_chunk(**overrides):
    base = {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Some content here",
        "url": "https://example.com",
        "title": "Test Page",
        "page_type": "general",
        "headings_path": [],
        "content_hash": "deadbeef01234567",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 4,
    }
    base.update(overrides)
    return base


def test_valid_chunk_passes():
    result = validate_chunks([_valid_chunk()])
    assert len(result.valid) == 1
    assert len(result.invalid) == 0


def test_missing_chunk_id_rejected():
    result = validate_chunks([_valid_chunk(chunk_id="")])
    assert len(result.valid) == 0
    assert len(result.invalid) == 1
    assert "chunk_id" in result.invalid[0][1]


def test_missing_document_id_rejected():
    result = validate_chunks([_valid_chunk(document_id="")])
    assert len(result.valid) == 0
    assert "document_id" in result.invalid[0][1]


def test_empty_text_rejected():
    result = validate_chunks([_valid_chunk(text="")])
    assert len(result.valid) == 0
    assert "text" in result.invalid[0][1]


def test_whitespace_only_text_rejected():
    result = validate_chunks([_valid_chunk(text="   \n  ")])
    assert len(result.valid) == 0
    assert "text" in result.invalid[0][1]


def test_missing_url_rejected():
    result = validate_chunks([_valid_chunk(url="")])
    assert len(result.valid) == 0
    assert "url" in result.invalid[0][1]


def test_missing_title_gets_fallback():
    result = validate_chunks([_valid_chunk(title="")])
    assert len(result.valid) == 1
    assert result.valid[0]["title"] == "Untitled"


def test_none_title_gets_fallback():
    result = validate_chunks([_valid_chunk(title=None)])
    assert len(result.valid) == 1
    assert result.valid[0]["title"] == "Untitled"


def test_mixed_valid_and_invalid():
    chunks = [
        _valid_chunk(chunk_id="a_0000"),
        _valid_chunk(chunk_id=""),  # invalid
        _valid_chunk(chunk_id="c_0000", text=""),  # invalid
    ]
    result = validate_chunks(chunks)
    assert len(result.valid) == 1
    assert len(result.invalid) == 2


def test_empty_list_returns_empty():
    result = validate_chunks([])
    assert result.valid == []
    assert result.invalid == []
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_validator.py -v
```

- [ ] **Step 3: Implement `validator.py`**

```python
"""Validate chunk records before embedding."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a batch of chunks."""
    valid: List[dict] = field(default_factory=list)
    invalid: List[Tuple[dict, str]] = field(default_factory=list)


def validate_chunks(chunks: List[dict]) -> ValidationResult:
    """Validate chunk records. Invalid chunks are skipped with a reason."""
    result = ValidationResult()

    for chunk in chunks:
        reason = _check(chunk)
        if reason:
            result.invalid.append((chunk, reason))
            logger.debug("Invalid chunk %s: %s", chunk.get("chunk_id", "?"), reason)
        else:
            result.valid.append(chunk)

    return result


def _check(chunk: dict) -> str | None:
    """Return a failure reason string, or None if valid."""
    if not chunk.get("chunk_id"):
        return "missing chunk_id"
    if not chunk.get("document_id"):
        return "missing document_id"
    if not (chunk.get("text") or "").strip():
        return "empty text"
    if not chunk.get("url"):
        return "missing url"

    # Title fallback — not a rejection, just a fix-up
    if not chunk.get("title"):
        chunk["title"] = "Untitled"

    return None
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/embedder && python -m pytest tests/test_validator.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/validator.py services/embedder/tests/test_validator.py
git commit -m "feat: add chunk validator"
```

---

## Task 5: Embedder (SentenceTransformers)

**Files:**
- Create: `services/embedder/src/embedder/embedder.py`
- Create: `services/embedder/tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

File: `services/embedder/tests/test_embedder.py`

```python
"""Tests for the embedding module — model is always mocked."""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from embedder.embedder import Embedder, build_context


# --- build_context (pure function, no mock needed) ---

def test_build_context_full():
    chunk = {
        "title": "Faculty | CS | VT",
        "headings_path": ["People", "Faculty", "Tenured Faculty"],
        "text": "Professor Smith researches systems.",
    }
    ctx = build_context(chunk)
    assert "Title: Faculty | CS | VT" in ctx
    assert "Section: Tenured Faculty" in ctx
    assert "Path: People > Faculty > Tenured Faculty" in ctx
    assert "Professor Smith researches systems." in ctx


def test_build_context_no_headings():
    chunk = {
        "title": "Home",
        "headings_path": [],
        "text": "Welcome to CS.",
    }
    ctx = build_context(chunk)
    assert "Title: Home" in ctx
    assert "Section:" not in ctx
    assert "Path:" not in ctx
    assert "Welcome to CS." in ctx


def test_build_context_single_heading():
    chunk = {
        "title": "About",
        "headings_path": ["About Us"],
        "text": "We are the CS department.",
    }
    ctx = build_context(chunk)
    assert "Section: About Us" in ctx
    assert "Path: About Us" in ctx


# --- Embedder class (model mocked) ---

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 1024
    # Return numpy arrays mimicking real embeddings
    model.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 1024).astype(np.float32)
    return model


@pytest.fixture
def embedder(mock_model):
    with patch("embedder.embedder.SentenceTransformer", return_value=mock_model):
        return Embedder("mock-model")


def test_embedder_dimension(embedder):
    assert embedder.dimension == 1024


def test_embed_batch_returns_list_of_lists(embedder):
    result = embedder.embed_batch(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 1024
    assert isinstance(result[0], list)


def test_embed_batch_single(embedder):
    result = embedder.embed_batch(["hello"])
    assert len(result) == 1


def test_embed_batch_empty(embedder):
    result = embedder.embed_batch([])
    assert result == []
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_embedder.py -v
```

- [ ] **Step 3: Implement `embedder.py`**

```python
"""Embedding model wrapper using SentenceTransformers."""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def build_context(chunk: dict) -> str:
    """Build enriched embedding input from chunk metadata + text.

    Format:
        Title: <title>
        Section: <last heading>
        Path: <headings joined>

        <chunk text>
    """
    parts = [f"Title: {chunk['title']}"]

    headings = chunk.get("headings_path") or []
    if headings:
        parts.append(f"Section: {headings[-1]}")
        parts.append(f"Path: {' > '.join(headings)}")

    parts.append("")
    parts.append(chunk["text"])
    return "\n".join(parts)


class Embedder:
    """Loads a SentenceTransformers model and generates embeddings."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Model loaded — dimension=%d", self.dimension)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of float lists."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/embedder && python -m pytest tests/test_embedder.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/embedder.py services/embedder/tests/test_embedder.py
git commit -m "feat: add embedding model wrapper"
```

---

## Task 6: Qdrant indexer

**Files:**
- Create: `services/embedder/src/embedder/indexer.py`
- Create: `services/embedder/tests/test_indexer.py`

- [ ] **Step 1: Write failing tests**

File: `services/embedder/tests/test_indexer.py`

```python
"""Tests for Qdrant indexer using mocked client."""
import uuid
from unittest.mock import MagicMock, patch, call
import pytest

from embedder.indexer import QdrantIndexer, chunk_id_to_uuid


# --- chunk_id_to_uuid ---

def test_chunk_id_to_uuid_deterministic():
    u1 = chunk_id_to_uuid("abc_0000")
    u2 = chunk_id_to_uuid("abc_0000")
    assert u1 == u2


def test_chunk_id_to_uuid_different_ids_differ():
    assert chunk_id_to_uuid("abc_0000") != chunk_id_to_uuid("abc_0001")


def test_chunk_id_to_uuid_is_valid_uuid():
    result = chunk_id_to_uuid("abc_0000")
    uuid.UUID(result)  # raises if invalid


# --- QdrantIndexer ---

@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    # Collection does not exist yet
    collection_info = MagicMock()
    collection_info.collections = []
    client.get_collections.return_value = collection_info
    return client


@pytest.fixture
def indexer(mock_qdrant):
    with patch("embedder.indexer.QdrantClient", return_value=mock_qdrant):
        idx = QdrantIndexer(
            host="localhost",
            port=6333,
            collection="test_chunks",
            vector_size=1024,
        )
    idx._client = mock_qdrant
    return idx


def test_ensure_collection_creates_if_missing(mock_qdrant, indexer):
    mock_qdrant.create_collection.assert_called_once()
    call_args = mock_qdrant.create_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_chunks"


def test_ensure_collection_creates_payload_index(mock_qdrant, indexer):
    mock_qdrant.create_payload_index.assert_called_once()
    call_args = mock_qdrant.create_payload_index.call_args
    assert call_args.kwargs["field_name"] == "document_id"


def test_upsert_chunks(indexer, mock_qdrant):
    chunks = [
        {
            "chunk_id": "abc_0000",
            "document_id": "abc",
            "url": "https://example.com",
            "title": "Test",
            "page_type": "general",
            "headings_path": [],
            "chunk_index": 0,
            "content_hash": "deadbeef",
            "crawl_timestamp": "2026-03-16T00:00:00",
            "token_count": 10,
        }
    ]
    embeddings = [[0.1] * 1024]
    indexer.upsert_chunks(chunks, embeddings)
    mock_qdrant.upsert.assert_called_once()
    call_args = mock_qdrant.upsert.call_args
    assert call_args.kwargs["collection_name"] == "test_chunks"
    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == chunk_id_to_uuid("abc_0000")
    assert points[0].payload["document_id"] == "abc"
    assert points[0].payload["chunk_id"] == "abc_0000"


def test_delete_stale_chunks_removes_old(indexer, mock_qdrant):
    # Simulate Qdrant returning 3 points for this doc, but only 2 are current
    old_point = MagicMock()
    old_point.id = chunk_id_to_uuid("abc_0002")
    current_point_1 = MagicMock()
    current_point_1.id = chunk_id_to_uuid("abc_0000")
    current_point_2 = MagicMock()
    current_point_2.id = chunk_id_to_uuid("abc_0001")

    mock_qdrant.scroll.return_value = ([old_point, current_point_1, current_point_2], None)

    deleted = indexer.delete_stale_chunks("abc", {"abc_0000", "abc_0001"})
    assert deleted == 1
    mock_qdrant.delete.assert_called_once()


def test_delete_stale_chunks_nothing_to_delete(indexer, mock_qdrant):
    point = MagicMock()
    point.id = chunk_id_to_uuid("abc_0000")
    mock_qdrant.scroll.return_value = ([point], None)

    deleted = indexer.delete_stale_chunks("abc", {"abc_0000"})
    assert deleted == 0
    mock_qdrant.delete.assert_not_called()
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_indexer.py -v
```

- [ ] **Step 3: Implement `indexer.py`**

```python
"""Qdrant vector database indexer."""
from __future__ import annotations

import logging
import uuid
from typing import List, Set

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id string to a deterministic UUID for Qdrant point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantIndexer:
    """Manages a Qdrant collection: create, upsert, delete stale."""

    def __init__(self, host: str, port: int, collection: str, vector_size: int) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection with cosine similarity if it does not exist."""
        existing = self._client.get_collections().collections
        if any(c.name == self._collection for c in existing):
            logger.info("Collection %s already exists", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection %s (dim=%d, cosine)", self._collection, self._vector_size)

    def upsert_chunks(self, chunks: List[dict], embeddings: List[List[float]]) -> None:
        """Upsert chunk embeddings as Qdrant points."""
        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(models.PointStruct(
                id=chunk_id_to_uuid(chunk["chunk_id"]),
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "page_type": chunk.get("page_type", "general"),
                    "headings_path": chunk.get("headings_path", []),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content_hash": chunk.get("content_hash", ""),
                    "crawl_timestamp": chunk.get("crawl_timestamp", ""),
                    "token_count": chunk.get("token_count", 0),
                },
            ))
        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.debug("Upserted %d points to %s", len(points), self._collection)

    def delete_stale_chunks(self, document_id: str, current_chunk_ids: Set[str]) -> int:
        """Delete Qdrant points for document_id not in current_chunk_ids."""
        current_uuids = {chunk_id_to_uuid(cid) for cid in current_chunk_ids}

        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )]
            ),
            limit=10000,
        )

        stale_ids = [p.id for p in results if p.id not in current_uuids]
        if stale_ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=stale_ids),
            )
            logger.info("Deleted %d stale points for doc %s", len(stale_ids), document_id)
        return len(stale_ids)
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/embedder && python -m pytest tests/test_indexer.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/indexer.py services/embedder/tests/test_indexer.py
git commit -m "feat: add Qdrant indexer with stale chunk cleanup"
```

---

## Task 7: Main entrypoint

**Files:**
- Create: `services/embedder/src/embedder/main.py`
- Create: `services/embedder/tests/test_main.py`

- [ ] **Step 1: Write failing tests**

File: `services/embedder/tests/test_main.py`

```python
"""End-to-end tests for embedding orchestration with all deps mocked."""
from unittest.mock import MagicMock, patch
import pytest

from embedder.main import run_embedding


VALID_CHUNKS = [
    {
        "chunk_id": "abc_0000",
        "document_id": "abc",
        "chunk_index": 0,
        "text": "Faculty info",
        "url": "https://example.com",
        "title": "Faculty",
        "page_type": "faculty",
        "headings_path": ["People", "Faculty"],
        "content_hash": "deadbeef",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 3,
    },
    {
        "chunk_id": "abc_0001",
        "document_id": "abc",
        "chunk_index": 1,
        "text": "More faculty info",
        "url": "https://example.com",
        "title": "Faculty",
        "page_type": "faculty",
        "headings_path": ["People", "Faculty"],
        "content_hash": "cafebabe",
        "crawl_timestamp": "2026-03-16T00:00:00",
        "token_count": 4,
    },
]


def _make_storage(chunk_files: dict) -> MagicMock:
    storage = MagicMock()
    storage.list_chunk_keys.return_value = list(chunk_files.keys())
    storage.download_chunks.side_effect = lambda key: chunk_files[key]
    return storage


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.dimension = 1024
    embedder.embed_batch.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    return embedder


def _make_indexer() -> MagicMock:
    indexer = MagicMock()
    indexer.delete_stale_chunks.return_value = 0
    return indexer


def test_run_embedding_processes_doc(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["docs_processed"] == 1
    assert stats["chunks_embedded"] == 2
    assert stats["chunks_skipped"] == 0
    assert stats["failed"] == 0
    assert embedder.embed_batch.called
    assert indexer.upsert_chunks.called


def test_run_embedding_skips_invalid(embedder_config):
    bad_chunks = [{"chunk_id": "", "text": "", "document_id": "", "url": ""}]
    storage = _make_storage({"chunks/bad.json": bad_chunks})
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["chunks_skipped"] == 1
    assert stats["chunks_embedded"] == 0
    assert not embedder.embed_batch.called


def test_run_embedding_handles_failure(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()
    indexer.upsert_chunks.side_effect = RuntimeError("Qdrant down")

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["failed"] == 1
    assert stats["docs_processed"] == 0


def test_run_embedding_deletes_stale(embedder_config):
    storage = _make_storage({"chunks/abc.json": VALID_CHUNKS})
    embedder = _make_embedder()
    indexer = _make_indexer()
    indexer.delete_stale_chunks.return_value = 3

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["stale_deleted"] == 3
    indexer.delete_stale_chunks.assert_called_once_with("abc", {"abc_0000", "abc_0001"})


def test_run_embedding_multiple_docs(embedder_config):
    storage = _make_storage({
        "chunks/abc.json": VALID_CHUNKS,
        "chunks/def.json": [
            {**VALID_CHUNKS[0], "chunk_id": "def_0000", "document_id": "def"},
        ],
    })
    embedder = _make_embedder()
    indexer = _make_indexer()

    stats = run_embedding(storage, embedder, indexer, embedder_config)
    assert stats["docs_processed"] == 2
    assert stats["chunks_embedded"] == 3
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/embedder && python -m pytest tests/test_main.py -v
```

- [ ] **Step 3: Implement `main.py`**

```python
"""Embedder entrypoint — read chunks, embed, index into Qdrant."""
from __future__ import annotations

import logging
import sys
from typing import Any

from embedder.config import EmbedderConfig
from embedder.embedder import Embedder, build_context
from embedder.indexer import QdrantIndexer
from embedder.storage import EmbedderStorage
from embedder.validator import validate_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_embedding(
    storage: Any,
    embedder: Any,
    indexer: Any,
    config: EmbedderConfig,
) -> dict:
    """Process all chunk files: validate → embed → upsert → clean stale.

    Returns stats dict.
    """
    keys = storage.list_chunk_keys()
    logger.info("Found %d chunk files to process", len(keys))

    stats = {
        "docs_processed": 0,
        "chunks_embedded": 0,
        "chunks_skipped": 0,
        "stale_deleted": 0,
        "failed": 0,
    }

    for key in keys:
        try:
            raw_chunks = storage.download_chunks(key)
            result = validate_chunks(raw_chunks)

            for _, reason in result.invalid:
                logger.warning("Skipped invalid chunk in %s: %s", key, reason)
            stats["chunks_skipped"] += len(result.invalid)

            if not result.valid:
                logger.debug("No valid chunks in %s, skipping", key)
                continue

            # Build contexts and embed
            contexts = [build_context(c) for c in result.valid]

            # Batch embedding according to config
            all_embeddings = []
            batch_size = config.embedding_batch_size
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i : i + batch_size]
                all_embeddings.extend(embedder.embed_batch(batch))

            # Upsert into Qdrant
            indexer.upsert_chunks(result.valid, all_embeddings)

            # Delete stale chunks for this document
            doc_id = result.valid[0]["document_id"]
            current_ids = {c["chunk_id"] for c in result.valid}
            deleted = indexer.delete_stale_chunks(doc_id, current_ids)

            stats["docs_processed"] += 1
            stats["chunks_embedded"] += len(result.valid)
            stats["stale_deleted"] += deleted
            logger.info(
                "Indexed %s → doc=%s, %d embedded, %d stale deleted",
                key, doc_id, len(result.valid), deleted,
            )

        except Exception as exc:
            logger.error("Failed to process %s: %s", key, exc)
            stats["failed"] += 1

    logger.info(
        "Embedding complete: %d docs, %d chunks embedded, %d skipped, %d stale deleted, %d failed",
        stats["docs_processed"],
        stats["chunks_embedded"],
        stats["chunks_skipped"],
        stats["stale_deleted"],
        stats["failed"],
    )
    return stats


def cli() -> None:
    """CLI entrypoint for the embedder service."""
    logger.info("Starting HokieHelp embedder")

    try:
        config = EmbedderConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = EmbedderStorage(config)
    embedder = Embedder(config.embedding_model)
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection=config.qdrant_collection,
        vector_size=embedder.dimension,
    )

    stats = run_embedding(storage, embedder, indexer, config)

    if stats["failed"] > 0:
        logger.warning("%d documents failed", stats["failed"])
        sys.exit(1)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run ALL tests to confirm PASS**

```bash
cd services/embedder && python -m pytest -v
```

Expected: all tests pass (5 config + 3 storage + 11 validator + 8 embedder + 8 indexer + 5 main = ~40)

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/main.py services/embedder/tests/test_main.py
git commit -m "feat: add embedder main entrypoint"
```

---

## Task 8: Dockerfile

**Files:**
- Create: `services/embedder/Dockerfile`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Install with CPU-only PyTorch to minimize image size (~200MB vs ~2GB)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Pre-download the embedding model so the CronJob doesn't re-download 1.3GB each run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

CMD ["python", "-m", "embedder.main"]
```

Note: This produces a ~2.5GB image (model weights + torch CPU). The model is baked in so the weekly CronJob starts instantly without downloading.

- [ ] **Step 2: Commit**

```bash
git add services/embedder/Dockerfile
git commit -m "feat: add embedder Dockerfile with pre-downloaded model"
```

---

## Task 9: Kubernetes manifests

**Files:**
- Create: `k8s/qdrant-pvc.yaml`
- Create: `k8s/qdrant-deployment.yaml`
- Create: `k8s/embedder-configmap.yaml`
- Create: `k8s/embedder-cronjob.yaml`

- [ ] **Step 1: Write `k8s/qdrant-pvc.yaml`**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-data
  namespace: test
  labels:
    app: qdrant
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ceph-rbd
  resources:
    requests:
      storage: 5Gi
```

- [ ] **Step 2: Write `k8s/qdrant-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: test
  labels:
    app: qdrant
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
              name: rest
            - containerPort: 6334
              name: grpc
          volumeMounts:
            - name: data
              mountPath: /qdrant/storage
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /readyz
              port: 6333
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /livez
              port: 6333
            initialDelaySeconds: 10
            periodSeconds: 30
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: qdrant-data
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: test
  labels:
    app: qdrant
spec:
  selector:
    app: qdrant
  ports:
    - port: 6333
      targetPort: 6333
      name: rest
    - port: 6334
      targetPort: 6334
      name: grpc
```

- [ ] **Step 3: Write `k8s/embedder-configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: embedder-config
  namespace: test
  labels:
    app: hokiehelp-embedder
data:
  MINIO_ENDPOINT: "minio:9000"
  MINIO_CHUNKS_BUCKET: "chunks"
  MINIO_SECURE: "false"
  QDRANT_HOST: "qdrant"
  QDRANT_PORT: "6333"
  QDRANT_COLLECTION: "hokiehelp_chunks"
  EMBEDDING_MODEL: "BAAI/bge-large-en-v1.5"
  EMBEDDING_BATCH_SIZE: "32"
```

- [ ] **Step 4: Write `k8s/embedder-cronjob.yaml`**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hokiehelp-embedder
  namespace: test
  labels:
    app: hokiehelp-embedder
spec:
  schedule: "0 5 * * 0"  # Weekly on Sunday at 5am (1h after chunker at 4am)
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 7200  # 2 hour timeout (model loading + embedding)
      template:
        metadata:
          labels:
            app: hokiehelp-embedder
        spec:
          restartPolicy: Never
          containers:
            - name: embedder
              image: ghcr.io/prakharmodi26/hokiehelp-embedder:latest
              imagePullPolicy: Always
              envFrom:
                - configMapRef:
                    name: embedder-config
              env:
                - name: MINIO_ACCESS_KEY
                  valueFrom:
                    secretKeyRef:
                      name: minio-credentials
                      key: access-key
                - name: MINIO_SECRET_KEY
                  valueFrom:
                    secretKeyRef:
                      name: minio-credentials
                      key: secret-key
              resources:
                requests:
                  memory: "2Gi"
                  cpu: "500m"
                limits:
                  memory: "4Gi"
                  cpu: "2000m"
```

- [ ] **Step 5: Commit**

```bash
git add k8s/qdrant-pvc.yaml k8s/qdrant-deployment.yaml k8s/embedder-configmap.yaml k8s/embedder-cronjob.yaml
git commit -m "feat: add K8s manifests for Qdrant and embedder service"
```

---

## Task 10: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/embedder-ci.yaml`

- [ ] **Step 1: Write `.github/workflows/embedder-ci.yaml`**

```yaml
name: Embedder CI

on:
  push:
    branches: [main]
    paths:
      - "services/embedder/**"
      - ".github/workflows/embedder-ci.yaml"
  pull_request:
    branches: [main]
    paths:
      - "services/embedder/**"

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: prakharmodi26/hokiehelp-embedder

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/embedder
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[dev]"
      - run: pytest -v

  build-and-push:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest

      - uses: docker/build-push-action@v5
        with:
          context: services/embedder
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/embedder-ci.yaml
git commit -m "feat: add GitHub Actions CI/CD for embedder service"
```

---

## Task 11: Deploy to cluster

- [ ] **Step 1: Show current kubectl context**

```bash
kubectl config current-context
```

Expected: `endeavour`

- [ ] **Step 2: Deploy Qdrant**

```bash
kubectl apply -f k8s/qdrant-pvc.yaml
kubectl apply -f k8s/qdrant-deployment.yaml
```

- [ ] **Step 3: Verify Qdrant is running**

```bash
kubectl get pods -n test -l app=qdrant
kubectl get svc -n test -l app=qdrant
kubectl rollout status deployment/qdrant -n test
```

- [ ] **Step 4: Apply embedder ConfigMap and CronJob**

```bash
kubectl apply -f k8s/embedder-configmap.yaml
kubectl apply -f k8s/embedder-cronjob.yaml
```

- [ ] **Step 5: Verify resources**

```bash
kubectl get pods -n test
kubectl get svc -n test
kubectl get cronjobs -n test
```

- [ ] **Step 6: Push code to trigger CI build**

```bash
git push origin main
```

Wait for CI to build and push the Docker image.

- [ ] **Step 7: Trigger manual embedder job**

```bash
kubectl create job --from=cronjob/hokiehelp-embedder embedder-test -n test
kubectl logs -n test -l app=hokiehelp-embedder -f
```

Expected logs:
```
INFO embedder.main: Starting HokieHelp embedder
INFO embedder.embedder: Loading embedding model: BAAI/bge-large-en-v1.5
INFO embedder.embedder: Model loaded — dimension=1024
INFO embedder.indexer: Created collection hokiehelp_chunks (dim=1024, cosine)
INFO embedder.main: Found N chunk files to process
INFO embedder.main: Indexed chunks/... → doc=..., N embedded, 0 stale deleted
...
INFO embedder.main: Embedding complete: N docs, N chunks embedded, 0 skipped, 0 stale deleted, 0 failed
```

- [ ] **Step 8: Verify Qdrant has data**

```bash
kubectl port-forward -n test svc/qdrant 6333:6333 &
sleep 2
curl -s http://localhost:6333/collections/hokiehelp_chunks | python3 -m json.tool
```

Expected: collection info showing points_count > 0, vector dimension 1024, distance Cosine.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| SentenceTransformers (not Ollama) | User requirement; model runs in-process, no external service dependency |
| bge-large-en-v1.5 (1024 dim) | Strong English embedding model; well-suited for RAG semantic search |
| CPU-only torch in Docker | Image ~2.5GB vs ~5GB with CUDA; sufficient for 463 docs weekly |
| Model baked into Docker image | Avoids 1.3GB download on every CronJob run |
| UUID5 for Qdrant point IDs | Deterministic from chunk_id — upserts are idempotent |
| `document_id` payload index | Enables efficient stale chunk queries (scroll + filter) |
| Enriched context (title + headings + text) | Richer embeddings than raw text alone |
| CronJob Sunday 5 AM | 1h after chunker (4 AM), 2h after crawler (3 AM) |
| 2Gi–4Gi memory for embedder pod | bge-large-en-v1.5 needs ~1.3GB RAM + overhead |
| Validation before embedding | Prevents bad data in vector DB; logs reasons for debugging |
