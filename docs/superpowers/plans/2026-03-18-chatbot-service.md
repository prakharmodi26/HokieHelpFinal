# HokieHelp Chatbot Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG chatbot service that retrieves relevant chunks from Qdrant, constructs a prompt with context, and generates answers using VT ARC's LLM API (gpt-oss-120b model via OpenAI-compatible endpoint).

**Architecture:** The chatbot service is a lightweight FastAPI application. On each query: (1) embed the user question with BAAI/bge-large-en-v1.5 using BGE query prefix, (2) search Qdrant for top-k relevant chunks, (3) build a system prompt with retrieved context, (4) call VT ARC LLM API (OpenAI-compatible) for the answer, (5) return the response with sources. Deployed as a K8s Deployment with a ClusterIP Service.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, sentence-transformers, qdrant-client, openai (Python SDK), httpx

---

## Files to Create/Modify

| File | Action | Responsibility |
|------|--------|----------------|
| `services/chatbot/pyproject.toml` | CREATE | Package metadata + dependencies |
| `services/chatbot/src/chatbot/__init__.py` | CREATE | Package init |
| `services/chatbot/src/chatbot/config.py` | CREATE | Environment-driven config (Qdrant, LLM, model settings) |
| `services/chatbot/src/chatbot/retriever.py` | CREATE | Embed query + search Qdrant, return ranked chunks |
| `services/chatbot/src/chatbot/llm.py` | CREATE | Build prompt with context, call VT ARC LLM API |
| `services/chatbot/src/chatbot/app.py` | CREATE | FastAPI app with `/ask` endpoint |
| `services/chatbot/src/chatbot/main.py` | CREATE | Uvicorn entry point |
| `services/chatbot/tests/__init__.py` | CREATE | Test package |
| `services/chatbot/tests/test_config.py` | CREATE | Config loading tests |
| `services/chatbot/tests/test_retriever.py` | CREATE | Retriever tests (mocked Qdrant + model) |
| `services/chatbot/tests/test_llm.py` | CREATE | LLM client tests (mocked HTTP) |
| `services/chatbot/tests/test_app.py` | CREATE | API endpoint integration tests |
| `services/chatbot/Dockerfile` | CREATE | Container image |
| `k8s/chatbot-configmap.yaml` | CREATE | Non-secret env vars |
| `k8s/chatbot-deployment.yaml` | CREATE | Deployment + Service |
| `.github/workflows/chatbot-ci.yaml` | CREATE | Test + build + push to GHCR |

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `services/chatbot/pyproject.toml`
- Create: `services/chatbot/src/chatbot/__init__.py`
- Create: `services/chatbot/src/chatbot/config.py`
- Create: `services/chatbot/tests/__init__.py`
- Create: `services/chatbot/tests/test_config.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "hokiehelp-chatbot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sentence-transformers>=2.2.0",
    "qdrant-client>=1.7.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create package init**

`services/chatbot/src/chatbot/__init__.py` — empty file.

- [ ] **Step 3: Write failing config test**

```python
# services/chatbot/tests/test_config.py
import os
from unittest.mock import patch

from chatbot.config import ChatbotConfig


def test_from_env_defaults():
    env = {
        "LLM_API_KEY": "sk-test-key",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.llm_api_key == "sk-test-key"
    assert cfg.llm_base_url == "https://llm-api.arc.vt.edu/api/v1"
    assert cfg.llm_model == "gpt-oss-120b"
    assert cfg.qdrant_host == "qdrant"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp_chunks"
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.top_k == 5


def test_from_env_overrides():
    env = {
        "LLM_API_KEY": "sk-override",
        "LLM_MODEL": "custom-model",
        "TOP_K": "10",
        "QDRANT_HOST": "custom-host",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.llm_model == "custom-model"
    assert cfg.top_k == 10
    assert cfg.qdrant_host == "custom-host"


def test_from_env_missing_api_key():
    import pytest
    env = {}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(KeyError):
            ChatbotConfig.from_env()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd services/chatbot && pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chatbot.config'`

- [ ] **Step 5: Implement config.py**

```python
# services/chatbot/src/chatbot/config.py
"""Chatbot service configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatbotConfig:
    """Immutable configuration for the chatbot service."""

    # LLM settings
    llm_api_key: str
    llm_base_url: str
    llm_model: str

    # Qdrant settings
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str

    # Embedding settings
    embedding_model: str

    # Retrieval settings
    top_k: int

    @classmethod
    def from_env(cls) -> ChatbotConfig:
        return cls(
            llm_api_key=os.environ["LLM_API_KEY"],
            llm_base_url=os.environ.get("LLM_BASE_URL", "https://llm-api.arc.vt.edu/api/v1"),
            llm_model=os.environ.get("LLM_MODEL", "gpt-oss-120b"),
            qdrant_host=os.environ.get("QDRANT_HOST", "qdrant"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            top_k=int(os.environ.get("TOP_K", "5")),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 PASSED

- [ ] **Step 7: Commit**

```bash
git add services/chatbot/
git commit -m "feat(chatbot): scaffold service with config module"
```

---

## Task 2: Retriever Module

**Files:**
- Create: `services/chatbot/src/chatbot/retriever.py`
- Create: `services/chatbot/tests/test_retriever.py`

- [ ] **Step 1: Write failing retriever tests**

```python
# services/chatbot/tests/test_retriever.py
"""Tests for the retriever module — model and Qdrant are mocked."""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from chatbot.retriever import Retriever, BGE_QUERY_PREFIX


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 1024
    model.encode.side_effect = lambda text, **kw: np.random.randn(1024).astype(np.float32)
    return model


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    return client


@pytest.fixture
def retriever(mock_model, mock_qdrant):
    with patch("chatbot.retriever.SentenceTransformer", return_value=mock_model), \
         patch("chatbot.retriever.QdrantClient", return_value=mock_qdrant), \
         patch("chatbot.retriever.torch.cuda.is_available", return_value=False):
        return Retriever(
            embedding_model="mock-model",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection="test_collection",
            top_k=3,
        )


def test_bge_query_prefix():
    assert BGE_QUERY_PREFIX.startswith("Represent this sentence")


def test_search_returns_results(retriever, mock_qdrant):
    # Mock Qdrant response
    mock_point = MagicMock()
    mock_point.score = 0.85
    mock_point.payload = {
        "chunk_id": "abc_0000",
        "text": "Sally Hamouda is a professor.",
        "url": "https://website.cs.vt.edu/faculty/sally.html",
        "title": "Sally Hamouda | CS | VT",
        "headings_path": ["Faculty"],
    }

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_response

    results = retriever.search("Sally Hamouda")
    assert len(results) == 1
    assert results[0]["score"] == 0.85
    assert results[0]["text"] == "Sally Hamouda is a professor."
    assert results[0]["url"] == "https://website.cs.vt.edu/faculty/sally.html"


def test_search_prepends_bge_prefix(retriever, mock_model, mock_qdrant):
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points.return_value = mock_response

    retriever.search("test query")
    call_args = mock_model.encode.call_args
    assert call_args[0][0] == BGE_QUERY_PREFIX + "test query"


def test_search_empty_results(retriever, mock_qdrant):
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points.return_value = mock_response

    results = retriever.search("nonexistent topic")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chatbot.retriever'`

- [ ] **Step 3: Implement retriever.py**

```python
# services/chatbot/src/chatbot/retriever.py
"""Embed user queries and search Qdrant for relevant chunks."""
from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    """Embeds queries with BGE prefix and searches Qdrant."""

    def __init__(
        self,
        embedding_model: str,
        qdrant_host: str,
        qdrant_port: int,
        collection: str,
        top_k: int,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = "float16"
        self._model = SentenceTransformer(
            embedding_model,
            device=device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        self._client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._collection = collection
        self._top_k = top_k
        logger.info(
            "Retriever ready — model=%s  device=%s  collection=%s  top_k=%d",
            embedding_model, device, collection, top_k,
        )

    def search(self, query: str) -> List[dict]:
        """Embed query and return top-k chunks from Qdrant."""
        text = BGE_QUERY_PREFIX + query
        vector = self._model.encode(text, show_progress_bar=False).tolist()

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=self._top_k,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            results.append({
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id"),
                "text": hit.payload.get("text", ""),
                "url": hit.payload.get("url", ""),
                "title": hit.payload.get("title", ""),
                "headings_path": hit.payload.get("headings_path", []),
                "page_type": hit.payload.get("page_type", ""),
            })
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/chatbot/src/chatbot/retriever.py services/chatbot/tests/test_retriever.py
git commit -m "feat(chatbot): add retriever module for Qdrant search"
```

---

## Task 3: LLM Client Module

**Files:**
- Create: `services/chatbot/src/chatbot/llm.py`
- Create: `services/chatbot/tests/test_llm.py`

- [ ] **Step 1: Write failing LLM client tests**

```python
# services/chatbot/tests/test_llm.py
"""Tests for the LLM client — API calls are mocked."""
from unittest.mock import MagicMock, patch
import pytest

from chatbot.llm import LLMClient, build_rag_prompt


def test_build_rag_prompt_with_chunks():
    chunks = [
        {"text": "Sally Hamouda is a CS professor.", "url": "https://example.com/sally", "title": "Sally Hamouda"},
        {"text": "She researches CS education.", "url": "https://example.com/sally2", "title": "Sally Hamouda"},
    ]
    prompt = build_rag_prompt("Who is Sally Hamouda?", chunks)
    assert "Sally Hamouda is a CS professor." in prompt
    assert "She researches CS education." in prompt
    assert "Who is Sally Hamouda?" in prompt
    assert "https://example.com/sally" in prompt


def test_build_rag_prompt_no_chunks():
    prompt = build_rag_prompt("random question", [])
    assert "no relevant information" in prompt.lower() or "random question" in prompt


def test_llm_client_ask():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Sally Hamouda is a professor of CS."
    mock_client.chat.completions.create.return_value = mock_response

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        answer = client.ask("Who is Sally?", [{"text": "Some context", "url": "https://example.com", "title": "Test"}])

    assert answer == "Sally Hamouda is a professor of CS."
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-oss-120b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chatbot.llm'`

- [ ] **Step 3: Implement llm.py**

```python
# services/chatbot/src/chatbot/llm.py
"""LLM client for VT ARC's OpenAI-compatible API."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are HokieHelp, a helpful assistant for Virginia Tech's Computer Science department.
Answer questions using ONLY the provided context from the CS department website.
If the context doesn't contain enough information to answer, say so honestly.
Always cite your sources by mentioning the relevant page title and URL.
Be concise and direct."""


def build_rag_prompt(question: str, chunks: List[dict]) -> str:
    """Build the user message with retrieved context."""
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No relevant information was found in the CS department website. "
            "Please let the user know you couldn't find an answer."
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk.get('title', 'Untitled')} — {chunk.get('url', '')}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Context from the VT CS department website:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )


class LLMClient:
    """Calls VT ARC's OpenAI-compatible LLM API."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("LLM client ready — model=%s  base_url=%s", model, base_url)

    def ask(self, question: str, chunks: List[dict]) -> str:
        """Send a RAG query to the LLM and return the answer."""
        user_message = build_rag_prompt(question, chunks)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        logger.info("LLM responded — question=%r  answer_len=%d", question, len(answer))
        return answer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/chatbot/src/chatbot/llm.py services/chatbot/tests/test_llm.py
git commit -m "feat(chatbot): add LLM client for VT ARC API"
```

---

## Task 4: FastAPI Application

**Files:**
- Create: `services/chatbot/src/chatbot/app.py`
- Create: `services/chatbot/src/chatbot/main.py`
- Create: `services/chatbot/tests/test_app.py`

- [ ] **Step 1: Write failing app tests**

```python
# services/chatbot/tests/test_app.py
"""Tests for the FastAPI app — retriever and LLM are mocked."""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.search.return_value = [
        {
            "score": 0.85,
            "chunk_id": "abc_0000",
            "text": "Sally Hamouda is a professor of CS.",
            "url": "https://website.cs.vt.edu/faculty/sally.html",
            "title": "Sally Hamouda | CS | VT",
            "headings_path": ["Faculty"],
            "page_type": "faculty",
        }
    ]
    return r


@pytest.fixture
def mock_llm():
    l = MagicMock()
    l.ask.return_value = "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    return l


@pytest.fixture
def client(mock_retriever, mock_llm):
    with patch("chatbot.app.retriever", mock_retriever), \
         patch("chatbot.app.llm_client", mock_llm):
        from chatbot.app import app
        yield TestClient(app)


def test_ask_endpoint(client, mock_retriever, mock_llm):
    response = client.post("/ask", json={"question": "Who is Sally Hamouda?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["url"] == "https://website.cs.vt.edu/faculty/sally.html"


def test_ask_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422 or response.status_code == 400


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chatbot.app'`

- [ ] **Step 3: Implement app.py**

```python
# services/chatbot/src/chatbot/app.py
"""FastAPI application for HokieHelp chatbot."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from chatbot.config import ChatbotConfig
from chatbot.retriever import Retriever
from chatbot.llm import LLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="HokieHelp Chatbot", version="0.1.0")

# Initialized at startup
retriever: Retriever | None = None
llm_client: LLMClient | None = None


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class Source(BaseModel):
    title: str
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
def startup() -> None:
    global retriever, llm_client
    if retriever is not None:
        return  # Already initialized (e.g., in tests)
    cfg = ChatbotConfig.from_env()
    retriever = Retriever(
        embedding_model=cfg.embedding_model,
        qdrant_host=cfg.qdrant_host,
        qdrant_port=cfg.qdrant_port,
        collection=cfg.qdrant_collection,
        top_k=cfg.top_k,
    )
    llm_client = LLMClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    logger.info("Chatbot startup complete")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    chunks = retriever.search(req.question)
    answer = llm_client.ask(req.question, chunks)

    sources = [
        Source(title=c.get("title", ""), url=c.get("url", ""), score=c["score"])
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
```

- [ ] **Step 4: Create main.py entry point**

```python
# services/chatbot/src/chatbot/main.py
"""Entry point for the chatbot service."""
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    uvicorn.run("chatbot.app:app", host="0.0.0.0", port=8000)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add services/chatbot/src/chatbot/app.py services/chatbot/src/chatbot/main.py services/chatbot/tests/test_app.py
git commit -m "feat(chatbot): add FastAPI app with /ask endpoint"
```

---

## Task 5: Dockerfile

**Files:**
- Create: `services/chatbot/Dockerfile`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "chatbot.main"]
```

- [ ] **Step 2: Verify Dockerfile builds locally**

Run: `docker build -t hokiehelp-chatbot:test services/chatbot/`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add services/chatbot/Dockerfile
git commit -m "feat(chatbot): add Dockerfile"
```

---

## Task 6: Kubernetes Manifests

**Files:**
- Create: `k8s/chatbot-configmap.yaml`
- Create: `k8s/chatbot-deployment.yaml`

The LLM API key must be stored in a K8s Secret. The user will create this manually.

- [ ] **Step 1: Create ConfigMap**

```yaml
# k8s/chatbot-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chatbot-config
  namespace: test
  labels:
    app: hokiehelp-chatbot
data:
  QDRANT_HOST: "qdrant"
  QDRANT_PORT: "6333"
  QDRANT_COLLECTION: "hokiehelp_chunks"
  EMBEDDING_MODEL: "BAAI/bge-large-en-v1.5"
  TOP_K: "5"
  LLM_BASE_URL: "https://llm-api.arc.vt.edu/api/v1"
  LLM_MODEL: "gpt-oss-120b"
```

- [ ] **Step 2: Create Deployment + Service**

```yaml
# k8s/chatbot-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hokiehelp-chatbot
  namespace: test
  labels:
    app: hokiehelp-chatbot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hokiehelp-chatbot
  template:
    metadata:
      labels:
        app: hokiehelp-chatbot
    spec:
      containers:
        - name: chatbot
          image: ghcr.io/prakharmodi26/hokiehelp-chatbot:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: chatbot-config
          env:
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-api-secret
                  key: api-key
          resources:
            requests:
              memory: "4Gi"
              cpu: "500m"
              nvidia.com/gpu: "1"
            limits:
              memory: "8Gi"
              cpu: "2000m"
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: chatbot
  namespace: test
  labels:
    app: hokiehelp-chatbot
spec:
  selector:
    app: hokiehelp-chatbot
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
  type: ClusterIP
```

- [ ] **Step 3: Commit**

```bash
git add k8s/chatbot-configmap.yaml k8s/chatbot-deployment.yaml
git commit -m "feat(chatbot): add K8s deployment manifests"
```

---

## Task 7: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/chatbot-ci.yaml`

- [ ] **Step 1: Create CI workflow**

```yaml
# .github/workflows/chatbot-ci.yaml
name: Chatbot CI

on:
  push:
    branches: [main]
    paths: ["services/chatbot/**"]
  pull_request:
    paths: ["services/chatbot/**"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install --extra-index-url https://download.pytorch.org/whl/cpu \
            -e "services/chatbot/.[dev]"
      - name: Run tests
        run: pytest services/chatbot/tests -v

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
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: services/chatbot
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/hokiehelp-chatbot:latest
            ghcr.io/${{ github.repository_owner }}/hokiehelp-chatbot:${{ github.sha }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/chatbot-ci.yaml
git commit -m "ci(chatbot): add test + build + push workflow"
```

---

## Task 8: Build, Push, Deploy

- [ ] **Step 1: Build Docker image locally**

```bash
docker build -t ghcr.io/prakharmodi26/hokiehelp-chatbot:latest services/chatbot/
```

- [ ] **Step 2: Push to GHCR**

```bash
docker push ghcr.io/prakharmodi26/hokiehelp-chatbot:latest
```

- [ ] **Step 3: Create LLM API secret**

The user needs to create this secret with their VT ARC API key:

```bash
kubectl create secret generic llm-api-secret \
  --from-literal=api-key='sk-YOUR-ACTUAL-API-KEY' \
  -n test
```

- [ ] **Step 4: Apply K8s manifests**

```bash
kubectl apply -f k8s/chatbot-configmap.yaml
kubectl apply -f k8s/chatbot-deployment.yaml
```

- [ ] **Step 5: Verify deployment**

```bash
kubectl get pods -n test -l app=hokiehelp-chatbot
kubectl rollout status deployment/hokiehelp-chatbot -n test
kubectl logs -n test -l app=hokiehelp-chatbot --tail=20
```

- [ ] **Step 6: Test the chatbot**

```bash
kubectl port-forward -n test svc/chatbot 8000:8000 &
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Sally Hamouda?"}'
```

Expected: JSON response with answer and sources.

---

## Verification Checklist

```bash
# All tests pass
cd services/chatbot && pytest tests/ -v

# K8s deployment healthy
kubectl get pods -n test -l app=hokiehelp-chatbot
kubectl get svc -n test chatbot

# API responds
kubectl port-forward -n test svc/chatbot 8000:8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Tell me about Sally Hamouda"}'
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What are the prerequisites for CS 3214?"}'
```
