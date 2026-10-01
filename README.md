# HokieHelp

**HokieHelp** is a Retrieval-Augmented Generation (RAG) chatbot for Virginia Tech. It answers questions using the indexed university sources, with inline citations and links to the pages it cites. Answer coverage and freshness depend on the actual corpus.

> **Unfinished answer-quality checkpoint:** start with the [session handoff](docs/HANDOFF-answer-quality.md), then the [design and evaluation](docs/answer-quality.md). An isolated Kubernetes canary and eight repaired catalog cards have been tested; the existing website deployments remain unchanged. Current source includes an unfinished retrieval refinement with one failing test and is not the tested canary image. This checkout is based on older `main`, not the newer unmerged live stack.

> Live at [hokiehelp.endeavour.cs.vt.edu](https://hokiehelp.endeavour.cs.vt.edu)

---

## What It Does

Ask HokieHelp questions about Virginia Tech, for example:

- *"Who are the faculty working on machine learning?"*
- *"What are the PhD admission requirements?"*
- *"What courses cover operating systems?"*

It retrieves evidence from its knowledge base, then asks an LLM for a grounded, cited answer. Greetings, clarification requests, and insufficient-evidence responses do not attach sources.

---

## Architecture Overview

HokieHelp is a multi-service system deployed on Kubernetes. Data flows through an automated ingestion pipeline and is queried at runtime by the chatbot.

```
cs.vt.edu ──► Crawler ──► Chunker ──► Embedder ──► Qdrant (vector DB)
                                                          │
User ──► Frontend ──► Chatbot ──► Query Planner ──► Retriever ──► Bounded Evidence ──► ARC LLM ──► Cited Answer
```

---

## Services

### Crawler
Performs a BFS deep crawl of `cs.vt.edu` using [Crawl4AI](https://github.com/unclecode/crawl4ai). Converts pages to clean Markdown and uploads them to MinIO (object storage). Automatically filters out stale pages (e.g., old seminar announcements), error pages, and non-content paths.

### Chunker
Splits cleaned Markdown documents into semantically meaningful chunks at heading boundaries. Uses token-aware splitting (target: 400 tokens, 64-token overlap) to produce dense, context-preserving segments. Each chunk is tagged with metadata: URL, title, page type, heading path, and content hash.

### Embedder
The legacy ingestion code in this checkout encodes chunks with `BAAI/bge-large-en-v1.5`. **The updated chatbot instead uses the existing live BGE-M3 embeddings/reranking service and named dense/sparse collection.** The model is unchanged; eight repaired program cards were reingested into an isolated clone using the compatible deployed ingestor. Do not run the legacy embedder against the live collection; see the [compatibility notes](docs/answer-quality.md).

### Chatbot
FastAPI service that handles user queries. Supports:
- **Single-turn** (`/ask`): One-shot Q&A
- **Multi-turn** (`/chat`): Conversation with bounded history and structured query planning
- **Streaming** (`/chat/stream`): Real-time token streaming via SSE

All three endpoints use bounded query planning. Retrieval searches original and planned wording using the existing dense/sparse index, merges candidates, and reranks them against the resolved question. There is no neighboring-chunk expansion. At most 12 selected chunks enter a character-budgeted context. VT-ARC generates answers with inline citations; the source list contains only cited pages. Model and retrieval budgets are configurable. Live evaluation has exposed unresolved program-matching failures; this branch is not yet accepted as an answer-quality fix.

### Frontend
Nginx-served static web app. Streams chatbot responses in real time, renders Markdown formatting, and displays source citations.

### Admin
Internal dashboard for managing the ingestion pipeline. Supports:
- Manual or scheduled crawls (cron-based)
- Live pipeline logs (SSE stream)
- Settings management
- Storage browser (MinIO)
- Health monitoring across all services



---

## Tech Stack

| Layer | Technology |
|---|---|
| Web scraping | Crawl4AI (crawl4ai) |
| Object storage | MinIO (S3-compatible) |
| Vector database | Qdrant |
| Chatbot embedding/reranking | Existing BGE-M3 / BGE reranker service (legacy ingestion remains separate) |
| Generative AI | VT-ARC OpenAI-compatible API (configurable planner and answer models) |
| Backend framework | FastAPI + Uvicorn |
| Scheduling | APScheduler |
| Streaming | Server-Sent Events (sse-starlette) |
| Orchestration | Kubernetes (Python SDK) |
| Frontend | Nginx + Vanilla JS |

---

## Deployment

The existing live services run on Kubernetes (namespace: `test`). Legacy manifests are in `k8s/`; they have **not** been updated or validated for this answer-quality branch. Do not apply them as an integrated rollout of these changes.

See the [handoff](docs/HANDOFF-answer-quality.md) before any rollout. Inspect without changing resources:

```bash
kubectl --context endeavour -n test get deployments,pods
```

The pipeline is triggered manually via the Admin dashboard or runs on a cron schedule (default: 3am every Sunday).

---

## Project Structure

```
services/
├── crawler/       # Web scraper
├── chunker/       # Text splitter
├── embedder/      # Embedding + indexing server
├── chatbot/       # RAG query service
├── admin/         # Dashboard + pipeline orchestrator
├── frontend/      # Chat UI
└── pipeline/      # K8s job sequencer
k8s/               # Kubernetes manifests
```
