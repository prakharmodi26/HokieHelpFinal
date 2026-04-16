# HokieHelp — Technical Overview

HokieHelp is an AI-powered question-answering assistant for Virginia Tech's Department of Computer Science. It answers questions about faculty, courses, programs, policies, and department resources by crawling the official VT CS website, indexing the content, and using a retrieval-augmented generation (RAG) pipeline at query time.

---

## System Architecture

The system is composed of six Python microservices, three stateful infrastructure components, and a Kubernetes deployment layer. All services are containerized with Docker and run in the `test` namespace on a Kubernetes cluster.

```
User
 │
 ▼
Frontend (port 8080)
 │  HTTP POST /chat/stream  (SSE)
 ▼
Chatbot Service (FastAPI, port 8000)
 │             │
 │             ▼
 │         Qdrant (port 6333)          ← vector + keyword index
 │
 ▼
Ollama LLM (port 11434)                ← qwen2.5:14b inference


Data Pipeline (triggered by Admin):

Admin Service
 │
 ├── subprocess: Crawler  →  MinIO (raw + cleaned buckets)
 ├── subprocess: Chunker  →  MinIO (chunks bucket)
 └── HTTP POST: Embedder  →  Qdrant (vector index)
```

---

## Full Tech Stack

| Layer | Technology | Details |
|---|---|---|
| Language | Python 3.12 | All backend services |
| Web framework | FastAPI + Uvicorn | Chatbot, Embedder, Admin |
| Web crawling | Crawl4AI ≥0.6.0 | BFSDeepCrawlStrategy, Playwright headless |
| Embedding model | BAAI/bge-large-en-v1.5 | SentenceTransformers; 1024-dim; GPU fp16 / CPU fp32 |
| Vector database | Qdrant | Cosine similarity + full-text payload index |
| LLM inference | Ollama (qwen2.5:14b) | Streaming chat; temperature 0.3 |
| Object storage | MinIO | S3-compatible; 4 buckets |
| Admin scheduler | APScheduler | Cron-based, AsyncIO |
| Admin state DB | SQLite via aiosqlite | Runs, settings, schedules |
| Admin UI | Alpine.js + Tailwind CSS | Served from FastAPI |
| Document parsing | PyMuPDF, python-docx | PDF and Word document extraction |
| Session management | HTTP-only cookie | UUID session, 1-week TTL |
| SSE streaming | sse-starlette | Token-by-token response streaming |
| Containers | Docker (multi-stage) | Shared base image |
| Orchestration | Kubernetes | namespace: test |
| CI/CD | GitHub Actions | Per-service workflows |

---

## Services

### 1. Crawler

**Role:** Discovers and downloads all pages from the VT CS website, converts them to clean Markdown, and stores them in MinIO.

**Source:** `services/crawler/src/crawler/`

**Dependencies:** `crawl4ai>=0.6.0`, `minio`, `pymupdf`, `python-docx`, `httpx`

#### How it works

**1. BFS Web Crawl**
Uses Crawl4AI's `BFSDeepCrawlStrategy` seeded from `https://cs.vt.edu`. Runs up to `max_depth=4` levels deep with `semaphore_count=5` (5 concurrent browser tabs). Playwright runs in headless + text-mode + light-mode to minimize overhead. Each page's HTML is converted to Markdown by Crawl4AI's `DefaultMarkdownGenerator`, which strips scripts, styles, nav, and iframes.

**2. Domain and Path Filtering**
`DomainFilter` restricts BFS traversal to `cs.vt.edu` and all its subdomains at any nesting depth (e.g., `students.website.cs.vt.edu` matches `cs.vt.edu`). Blocked domains (GitLab, mail, portal, forum, login, etc.) are excluded before fetching. Blocked paths (`/content/`, `/editor.html`, `/cs-root.html`, `/cs-source.html`) are excluded by path-prefix check after BFS returns results.

**3. Deduplication**
Each page's Markdown is SHA-256 hashed. Content hashes of all previously stored pages are loaded from MinIO at startup into a `seen_content_hashes` set. Pages with matching hashes are skipped without re-storing. This makes re-crawls efficient — unchanged pages produce no I/O.

**4. Metadata Collection**
For every successfully stored page, a metadata sidecar is written alongside the Markdown:
- `doc_id` — SHA-256 of the URL, first 16 hex chars
- `url`, `title`, `crawl_depth`, `crawl_timestamp`, `content_hash`, `markdown_size_bytes`
- `status_code`, `response_headers`, `last-modified`, `etag`
- `internal_links`, `external_links`

**5. Document Extraction (PDFs and Word)**
During the web crawl, all links to PDF and Word documents are collected into a `document_urls` set. After the web crawl completes, these documents are downloaded via `httpx`, converted to Markdown (PyMuPDF for PDFs, python-docx for Word), and stored in MinIO using the same format. Linked documents are trusted regardless of their host domain — the trust anchor is the cs.vt.edu page that linked them.

**6. Post-Crawl Cleaning**
After crawling, the crawler performs a second pass: reads every raw page from `crawled-pages`, applies `clean_markdown()` (strips navigation boilerplate, repeated headers/footers, CMS artifacts), detects CMS error pages using regex (`# Resource at '...' not found`) and skips them, then writes cleaned output to `crawled-pages-cleaned`. Metadata sidecars are mirrored with updated `markdown_size_bytes`.

A static `_department-info.md` document (campus addresses, phone numbers, social links) is also injected into the cleaned bucket so the RAG system has contact and location data even if those pages weren't crawled.

**7. Visit Log**
A human-readable visit log is uploaded to the `logs` MinIO bucket after each run. It records every URL visited with status (SAVED, FAILED, DUPLICATE, BLOCKED, EMPTY), depth, timestamp, and a summary by status and domain.

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `CRAWL_SEED_URL` | `https://cs.vt.edu` | Start URL |
| `CRAWL_MAX_DEPTH` | `4` | BFS depth limit |
| `CRAWL_MAX_PAGES` | `9999999` | Max pages to visit |
| `CRAWL_ALLOWED_DOMAINS` | `cs.vt.edu` | Comma-separated allowed domains |
| `CRAWL_BLOCKED_DOMAINS` | git, mail, portal, etc. | Comma-separated blocked subdomains |
| `CRAWL_BLOCKED_PATHS` | `/content/`, `/editor.html`, etc. | Comma-separated blocked path prefixes |
| `CRAWL_REQUEST_DELAY` | `0.1` | Seconds between requests |
| `CRAWL_PRUNE_THRESHOLD` | `0.45` | Low-content page prune threshold |
| `MINIO_ENDPOINT` | required | MinIO host:port |

---

### 2. Chunker

**Role:** Reads cleaned Markdown pages from MinIO, splits each page into semantically meaningful chunks at heading boundaries, and writes chunk JSON files back to MinIO.

**Source:** `services/chunker/src/chunker/`

**Dependencies:** `minio`

#### How it works

**1. Frontmatter Parsing**
Each Markdown document begins with a YAML frontmatter block (`---`). The parser extracts `doc_id`, `url`, `title`, `content_hash`, and `crawl_timestamp`. If frontmatter is absent (e.g., injected documents like `_department-info.md`), fallbacks are derived: `doc_id = SHA256(url)[:16]`, `content_hash = SHA256(body)[:16]`.

**2. Content Filters**
Before chunking, two filters are applied:

- **CMS error pages:** Detected by regex match for `# Resource at '...' not found` in the first 500 chars. These are pages the CMS served as "not found" — skipped entirely since they contain no useful content.

- **Stale time-sensitive pages:** Seminar and news pages at paths `/research/Seminars/` and `/News/Seminars/` with event dates older than 6 months are skipped. Event dates are extracted by matching the pattern `"Friday, November 7, 2025"` near the top of the page. If the date cannot be extracted, the page is kept (fail-safe). This prevents outdated seminar announcements from polluting the knowledge base.

**3. Heading-Based Section Splitting**
The Markdown body is split at H1/H2/H3 heading boundaries. Each section contains its heading line and all text until the next heading. A heading stack tracks the full breadcrumb path. For example, a section under "Graduate Programs > Admission > Deadlines" has `headings_path = ["Graduate Programs", "Admission", "Deadlines"]`. Text before the first heading is collected as a section with an empty `headings_path`.

**4. Token-Aware Merging and Splitting**
Token count is approximated as `len(text) // 4` (4 chars ≈ 1 token). Three cases:

- **Section < 120 tokens (minimum):** Accumulated into a pending batch. If pending total reaches minimum, the batch is flushed as one chunk. Prevents tiny sections (a heading with one sentence) from becoming isolated, low-signal chunks.

- **Section 120–400 tokens (preferred):** Emitted as a single chunk. If there are pending small sections, they are combined with this section if the combined total fits within `preferred`, otherwise the pending batch is flushed first.

- **Section > 400 tokens (preferred):** Split into overlapping windows at paragraph (`\n\n`) boundaries. Each window grows by appending paragraphs until the preferred size is exceeded, then it is flushed. The next window rewinds by `overlap_tokens * 4` chars from the tail of the previous window, preserving cross-boundary context.

**5. Chunk Record Structure**
Each chunk is a JSON object:
```json
{
  "chunk_id": "{doc_id}_{index:04d}",
  "document_id": "a1b2c3d4e5f6g7h8",
  "chunk_index": 0,
  "text": "...",
  "url": "https://cs.vt.edu/...",
  "title": "Page Title",
  "page_type": "faculty|course|research|news|about|general",
  "headings_path": ["Graduate Programs", "Admission Requirements"],
  "content_hash": "sha256_16chars",
  "crawl_timestamp": "2026-04-15T12:00:00Z",
  "token_count": 347
}
```

`page_type` is inferred from URL path keywords (e.g., `/people/` or `/faculty/` → `"faculty"`, `/courses/` → `"course"`, `/research/` or `/labs/` → `"research"`, etc.).

**6. Output**
One JSON file per source page written to the `chunks` MinIO bucket.

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `CHUNK_PREFERRED_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between split windows |
| `CHUNK_MIN_TOKENS` | `120` | Minimum before merging small sections |

---

### 3. Embedder

**Role:** Reads chunk JSON files from MinIO, generates dense vector embeddings, and upserts them into Qdrant. Exposes an HTTP API for async triggering by the Admin service.

**Source:** `services/embedder/src/embedder/`

**Dependencies:** `sentence-transformers`, `qdrant-client`, `minio`, `fastapi`, `uvicorn`

**Embedding model:** `BAAI/bge-large-en-v1.5` — 1024-dimensional dense embeddings trained for asymmetric retrieval. Uses fp16 on CUDA GPUs (~2× faster, half the memory); falls back to fp32 on CPU.

#### How it works

**1. Context Enrichment**
Before embedding, each chunk is enriched with its metadata:
```
Title: <page title>
Section: <last heading in path>
Path: <full heading breadcrumb joined with " > ">

<chunk text>
```
This gives the embedding model structural context beyond raw text, improving retrieval precision when questions reference specific sections or topics.

**2. Batch Embedding**
Chunks are embedded in configurable batches (`embedding_batch_size`, default 32). The full embedding run processes all chunk files from MinIO sequentially.

**3. Qdrant Upsert**
Each chunk becomes a Qdrant point:
- **ID:** Deterministic UUID via `uuid5(NAMESPACE_DNS, chunk_id)` — same chunk always produces the same UUID, enabling idempotent upserts across pipeline re-runs
- **Vector:** 1024-dim float32 (or float16 on GPU) embedding
- **Payload:** `chunk_id`, `document_id`, `url`, `title`, `page_type`, `headings_path`, `chunk_index`, `content_hash`, `crawl_timestamp`, `token_count`, `text`

**4. Stale Chunk Deletion**
After upserting the current chunks for a document, the indexer queries Qdrant for all points with that `document_id` and deletes any whose UUIDs are not in the current chunk set. This handles pages that shrunk across re-crawls — old chunks from removed sections are purged automatically.

**5. Qdrant Collection Setup**
On first startup, creates the `hokiehelp_chunks` collection with:
- Cosine distance metric
- `document_id` keyword payload index (fast filtering for stale deletion)
- Full-text `text` payload index (word tokenizer, min/max token len 2–20, lowercase) — enables keyword search in the chatbot

**6. HTTP API (async triggering)**
`POST /embed/start` returns a `run_id` immediately and starts embedding in an asyncio thread pool. `GET /embed/status/{run_id}` polls status (`running` / `completed` / `failed`) and returns final stats on completion. `GET /logs` returns recent service log lines from an in-memory ring buffer.

---

### 4. Chatbot

**Role:** Serves user questions via REST API. Implements hybrid retrieval (vector + keyword search with RRF fusion), query rewriting for multi-turn conversations, and streaming LLM response generation.

**Source:** `services/chatbot/src/chatbot/`

**Dependencies:** `fastapi`, `sentence-transformers`, `qdrant-client`, `ollama`, `uvicorn`

**LLM:** `qwen2.5:14b` via Ollama (configurable; separate rewriter model supported for different sizing)

#### Endpoints

| Endpoint | Description |
|---|---|
| `POST /ask` | Single-turn RAG query → `{answer, sources}` |
| `POST /chat` | Multi-turn RAG with `{question, history}` → `{answer, sources}` |
| `POST /chat/stream` | Multi-turn RAG with SSE streaming (tokens then sources then done) |
| `GET /health` | Health check |
| `GET /logs` | Last N lines from in-memory ring buffer |

#### Retrieval Pipeline (per request)

**Step 1 — Prompt Guard**
A lightweight content filter runs before any retrieval. Off-topic or abusive prompts are rejected with HTTP 400 before consuming compute.

**Step 2 — Rate Limit Check**
Each user session is tracked via an `hokiehelp_session` HTTP-only cookie (UUID, 1-week TTL). A sliding window counter limits sessions to `rate_limit_requests` (default 100) per `rate_limit_window_seconds` (default 3600). Exceeded sessions receive HTTP 429 with `Retry-After` and `X-RateLimit-Remaining` headers.

**Step 3 — Query Rewriting (multi-turn only)**
For requests with conversation history, the question is rewritten by an LLM call to resolve pronouns and references. The rewriter uses `temperature=0.0` (deterministic) and only receives the last 4 history messages (2 exchanges) to avoid injecting stale context from old turns. If history is empty, this step is skipped. Falls back to the original question on any LLM error.

Examples:
- "What are his research interests?" + history mentioning "Denis Gracanin" → "Denis Gracanin research interests"
- "What are the admission requirements?" (no pronoun) → unchanged

**Step 4 — Vector Search**
The (possibly rewritten) query is prefixed with:
```
Represent this sentence for searching relevant passages:
```
This is the BGE asymmetric retrieval prefix — documents are indexed without a prefix, queries use this prefix, which is how BAAI/bge-large-en-v1.5 is designed for Q&A retrieval. The prefixed query is encoded to 1024 dims and passed to Qdrant `query_points` for cosine similarity top-K search.

**Step 5 — Keyword Search (hybrid mode)**
The query is split into tokens. Common English stop words (~70 words: a, the, is, are, of, in, what, who, etc.) are removed. Each remaining token is matched against the full-text `text` index in Qdrant using `should` (OR) conditions — any single matching token can surface results. Results are scored manually by counting how many query tokens appear in each chunk's text, then sorted by match count descending.

This handles typos and proper noun lookups. Example: "Kirk Camron" — vector search may miss "Cameron" due to misspelling, but keyword search on "Kirk" still surfaces the correct faculty profile.

**Step 6 — RRF Fusion**
Vector and keyword results are merged using Reciprocal Rank Fusion:
```
RRF_score(chunk) = Σ  1 / (k + rank_in_list)
```
where `k=60` (standard parameter) and the sum is over each ranked list the chunk appears in. Chunks appearing in both lists score higher than chunks appearing in only one. Results sorted by RRF score, trimmed to `top_k` (default 5), filtered by `min_rrf_score`.

Each result is tagged `from_vector=True/False` — keyword-only chunks (those that never appeared in vector results) are kept in RRF scoring but excluded from LLM context and source display. They boosted ranking but aren't semantically validated.

**Step 7 — Adjacent Chunk Expansion**
For every vector result, the immediately adjacent chunks (sequence index ±1) from the same page are fetched from Qdrant. Chunk IDs follow `{page_hash}_{seq:04d}` format — adjacent chunks often contain complementary information. For example, if a faculty bio is in chunk `_0001`, contact details may be in chunk `_0000`. Fetching neighbors restores this context without making individual chunks larger (which would dilute embedding precision). Adjacent chunks are tagged `from_vector=True` and appended to the retrieval set.

**Step 8 — LLM Generation**
Vector results and adjacent chunks are formatted as numbered sources in the system prompt:
```
[Source 1] <title> — <url>
<chunk text>

---

[Source 2] ...
```
The full conversation history (up to 20 messages) is included as alternating user/assistant messages. The current question is the final user message. LLM is called via the Ollama Python client with `temperature=0.3`.

For `/chat/stream`, the LLM response is streamed token-by-token as SSE events:
```
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " CS"}
...
data: {"type": "sources", "sources": [{...}, ...]}
data: {"type": "done"}
```

**Step 9 — Source Deduplication**
Sources are deduplicated by URL, keeping the highest-scoring chunk per URL, then sorted by score descending.

#### System Prompt Design

The LLM system prompt enforces strict RAG discipline:
- Answer ONLY from retrieved context — never from training knowledge
- Person queries are high-risk: if a person's name is not in the retrieved context, use the fallback message immediately
- Contact details (email, phone, office) only if they appear word-for-word in retrieved context
- Never answer questions outside VT CS scope
- Specific fallback message: "I don't have enough context to answer that fully. HokieHelp is constantly building — we hope to have your answer soon!"
- Specific out-of-scope message for non-VT-CS questions
- Formatting rules per question type (person queries use a structured card format; lists use bullet points; processes use numbered steps)

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:14b` | Main answer model |
| `REWRITER_MODEL` | same as LLM_MODEL | Query rewrite model |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Query embedding model |
| `QDRANT_COLLECTION` | `hokiehelp_chunks` | Qdrant collection name |
| `TOP_K` | `5` | Max chunks returned |
| `MIN_SCORE` | `0.53` | Min cosine score (vector-only path) |
| `HYBRID_ENABLED` | `true` | Enable keyword + RRF |
| `KEYWORD_SEARCH_LIMIT` | `10` | Max keyword search results |
| `RRF_K` | `60` | RRF k parameter |
| `MIN_RRF_SCORE` | `0.0` | Min RRF score threshold |
| `MAX_HISTORY_MESSAGES` | `20` | History window for LLM |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `3600` | Rate limit window (1 hour) |

---

### 5. Admin

**Role:** Web-based control plane for the entire pipeline. Orchestrates crawl → chunk → embed as a sequential pipeline run, stores run history, supports cron-scheduled automatic runs, and provides a storage browser and service log viewer.

**Source:** `services/admin/src/admin/`

**Dependencies:** `fastapi`, `apscheduler`, `aiosqlite`, `httpx`, `sse-starlette`, `minio`, `qdrant-client`, `croniter`

**UI:** Single-page Alpine.js + Tailwind CSS application served directly from FastAPI at `/`. Dark theme (gray-950 background). Features: pipeline control panel, live log stream, run history table, cron schedule manager, settings editor, MinIO storage browser with file preview (Markdown rendered), and per-service log viewer.

#### Pipeline Orchestration

The `PipelineRunner` executes three stages sequentially. State machine:

```
IDLE → CRAWLING → CHUNKING → EMBEDDING → DONE
                                       ↘ FAILED (on any stage failure)
                                       ↘ CANCELLED (on stop)
```

**Stage 1 — Crawler:** Spawns `python -m crawler.main` as a subprocess with environment variables built from current admin settings. Streams stdout/stderr line-by-line via asyncio to all SSE-subscribed clients (live log tailing) and to a per-run log file on disk. If exit code ≠ 0, marks run as FAILED and stops.

**Stage 2 — Chunker:** After crawler exits 0, spawns `python -m chunker.main` as a subprocess. Same streaming behavior. If exit code ≠ 0, marks run as FAILED.

**Stage 3 — Embedder:** After chunker exits 0, sends `POST /embed/start` to the Embedder HTTP API. Polls `GET /embed/status/{run_id}` every 30 seconds until status is `completed` or `failed`. Broadcasts embedder status lines to SSE clients.

Every pipeline run is persisted in SQLite: run ID (UUID), start/end timestamps, current stage, settings snapshot, completion stats. Run history is queryable via `GET /api/history`.

#### Scheduler

Uses APScheduler's `AsyncIOScheduler` with `CronTrigger`. Schedules are stored in SQLite. On any CRUD operation, `scheduler.sync(schedules)` replaces all APScheduler jobs to match the current schedule list. Cron expressions are validated with `croniter` before saving. On each trigger, the scheduler fires `runner.start()` with the stored settings — if the pipeline is already running, the scheduled run is silently skipped with a warning log.

Next 5 run times are computed and returned with schedule status for display in the UI.

#### API Surface

| Endpoint | Description |
|---|---|
| `POST /api/pipeline/start` | Start pipeline run |
| `POST /api/pipeline/stop` | Cancel running pipeline |
| `GET /api/pipeline/status` | Current state + run_id |
| `GET /api/pipeline/logs` | SSE stream of live pipeline log lines |
| `GET /api/history` | List recent runs (paginated) |
| `GET /api/history/{run_id}` | Single run details |
| `GET /api/history/{run_id}/logs` | Full log text for a past run |
| `GET /api/settings` | Current settings dict |
| `PUT /api/settings` | Update settings |
| `GET /api/schedules` | List cron schedules with next-run times |
| `POST /api/schedules` | Create schedule |
| `PUT /api/schedules/{id}` | Update schedule |
| `DELETE /api/schedules/{id}` | Delete schedule |
| `GET /api/storage/buckets` | List MinIO buckets |
| `GET /api/storage/browse` | Browse bucket objects (sort by name/date) |
| `GET /api/storage/download` | Download a MinIO object |
| `GET /api/health` | Full health check (all services + MinIO stats) |
| `GET /api/storage/stats` | MinIO storage stats |
| `GET /api/services` | List services with /logs endpoints |
| `GET /api/services/{service}/logs` | Fetch recent logs from a service |

**Persistence:** SQLite database at `/data/admin.db` (PVC-backed in Kubernetes). Settings, run history, and schedules stored here.

---

### 6. Frontend

**Role:** User-facing chat interface. Communicates with the Chatbot service via SSE streaming.

**Deployment:** Kubernetes Deployment (`hokiehelp-frontend`, port 8080). NetworkPolicy restricts egress to the chatbot only (port 8000). Public access via Kubernetes Ingress.

**Interaction model:**
- Sends `POST /chat/stream` with `{question, history}`
- Receives SSE events: `{type: "token"}` streamed in real time, then `{type: "sources"}`, then `{type: "done"}`
- Renders Markdown responses (headings styled as purple H2/H3)
- Maintains conversation history client-side across turns

---

## Infrastructure Components

### MinIO (Object Storage)

S3-compatible object store. Four buckets:

| Bucket | Contents |
|---|---|
| `crawled-pages` | Raw Markdown + metadata sidecars from crawler |
| `crawled-pages-cleaned` | Cleaned Markdown + metadata after post-crawl pass |
| `chunks` | JSON chunk files (one per page) from chunker |
| `logs` | Visit logs from each crawl run |

NetworkPolicy allows port 9000 access only from pipeline pods, admin, and embedder.

### Qdrant (Vector Database)

Collection: `hokiehelp_chunks`

- **Distance metric:** Cosine similarity
- **Vector dimensions:** 1024 (BAAI/bge-large-en-v1.5)
- **Point ID scheme:** Deterministic UUID via `uuid5(NAMESPACE_DNS, chunk_id)` — idempotent upserts
- **Payload indexes:**
  - `document_id` (keyword) — used for stale chunk cleanup queries
  - `text` (full-text, word tokenizer, min/max token len 2–20, lowercase) — used for keyword search
- **Payload stored:** `chunk_id`, `document_id`, `url`, `title`, `page_type`, `headings_path`, `chunk_index`, `content_hash`, `crawl_timestamp`, `token_count`, `text`

NetworkPolicy allows port 6333/6334 access from chatbot, admin, embedder, and pipeline pods only.

### Ollama (LLM Server)

Serves `qwen2.5:14b` (configurable) for both answer generation and query rewriting. Chatbot connects via the Ollama Python client to port 11434. NetworkPolicy allows access from chatbot and admin pods only.

---

## Data Identifiers

```
Document ID  = SHA256(url)[:16]               (16-char hex, deterministic per URL)
Chunk ID     = "{doc_id}_{chunk_index:04d}"   (e.g., "a1b2c3d4e5f6g7h8_0000")
Qdrant ID    = UUID5(NAMESPACE_DNS, chunk_id) (deterministic UUID, idempotent upserts)
```

---

## Complete Data Flow: Pipeline Run

```
Admin triggers run (manual or cron)
       │
       ▼
1. CRAWLER STAGE
   • BFS crawl of cs.vt.edu via Playwright (max_depth=4, semaphore=5)
   • DomainFilter restricts BFS to cs.vt.edu and subdomains
   • Path-based blocking for CMS/editor paths
   • SHA-256 content-hash deduplication (skip unchanged pages)
   • Download linked PDFs (PyMuPDF) and Word docs (python-docx)
   • Build metadata sidecar per page
   • Upload raw Markdown + metadata → MinIO:crawled-pages
   • Post-crawl cleaning pass:
       - clean_markdown() strips nav, boilerplate
       - is_error_page() filters CMS 404 pages
       - cleaned Markdown → MinIO:crawled-pages-cleaned
   • Inject _department-info.md static doc
   • Upload visit log → MinIO:logs
       │
       ▼
2. CHUNKER STAGE
   • Read all .md files from MinIO:crawled-pages-cleaned
   • Parse YAML frontmatter → doc_id, url, title
   • Filter: skip CMS error pages (regex)
   • Filter: skip stale seminar/news pages (>180 days old)
   • Split body at H1/H2/H3 heading boundaries → sections
   • Track heading breadcrumb path per section
   • Token-aware merging: merge small (<120 tok), split large (>400 tok) with 64-tok overlap
   • Infer page_type from URL path
   • Write chunk JSON files → MinIO:chunks
       │
       ▼
3. EMBEDDER STAGE
   • Read all chunk JSON files from MinIO:chunks
   • Validate chunk schema (required fields, text non-empty)
   • Enrich each chunk: prepend Title + Section + Path headers
   • Embed enriched text in batches (BAAI/bge-large-en-v1.5, 1024-dim, fp16/fp32)
   • Upsert vectors + payloads into Qdrant (deterministic UUID IDs)
   • Delete stale chunks (re-crawled doc had fewer sections)
   • Report stats: docs_processed, chunks_embedded, stale_deleted, failed
```

---

## Complete Data Flow: Query Request

```
User sends question (with optional conversation history)
       │
       ▼
1. PROMPT GUARD
   • Lightweight content filter
   • Reject off-topic/abusive input with HTTP 400
       │
       ▼
2. RATE LIMIT CHECK
   • Read/create hokiehelp_session cookie (UUID, 1-week TTL)
   • Sliding window counter: 100 requests per hour per session
   • Reject with HTTP 429 if exceeded
       │
       ▼
3. QUERY REWRITING (multi-turn only, skipped if no history)
   • LLM call with last 4 history messages (temperature=0.0)
   • Decision rule: detect pronouns/references → resolve with entity from history
   • No follow-up signals → return question unchanged
   • On LLM error → fall back to original question
       │
       ▼
4. VECTOR SEARCH
   • Prefix query: "Represent this sentence for searching relevant passages: " + query
   • Encode to 1024-dim vector (BAAI/bge-large-en-v1.5)
   • Qdrant cosine similarity search → top_k=5 results with scores and payloads
       │
       ▼
5. KEYWORD SEARCH (hybrid mode, parallel with vector)
   • Tokenize query, strip trailing punctuation
   • Remove stop words (~70 common English words)
   • Qdrant full-text OR match on remaining tokens
   • Score each result by count of matched tokens in text
   • Sort by match count descending
       │
       ▼
6. RRF FUSION
   • For each chunk: RRF_score += 1 / (60 + rank) per ranked list it appears in
   • Sort by RRF score descending
   • Trim to top_k, drop below min_rrf_score
   • Tag from_vector=True if chunk appeared in vector results
       │
       ▼
7. ADJACENT CHUNK EXPANSION
   • For each vector result, parse chunk_id: "{page_hash}_{seq:04d}"
   • Fetch sequence ±1 chunks from same page via Qdrant scroll
   • Append adjacent chunks to context (tagged from_vector=True)
       │
       ▼
8. FILTER FOR LLM CONTEXT
   • Keep only from_vector=True chunks for LLM context and source display
   • Keyword-only chunks contributed to RRF ranking but are excluded here
       │
       ▼
9. LLM GENERATION
   • System prompt: role definition + RAG rules + numbered chunk sources
   • History: last 20 messages as alternating user/assistant
   • Current question as final user message
   • Ollama qwen2.5:14b, temperature=0.3
   • /chat: blocking call, return full answer
   • /chat/stream: SSE token stream → sources event → done event
       │
       ▼
10. SOURCE DEDUPLICATION
    • Group chunks by URL, keep highest-scoring chunk per URL
    • Sort by score descending
       │
       ▼
Response: {answer: string, sources: [{title, url, score}, ...]}
```

---

## Kubernetes Deployment

All workloads run in namespace `test`.

| Workload | Kind | Ports | PVC |
|---|---|---|---|
| `hokiehelp-frontend` | Deployment | 8080 | — |
| `hokiehelp-chatbot` | Deployment | 8000 | — |
| `hokiehelp-admin` | Deployment | 8080 | admin-data (SQLite) |
| `hokiehelp-embedder` | Deployment | 8080 | — |
| `minio` | Deployment | 9000 | minio-data |
| `qdrant` | Deployment | 6333, 6334 | qdrant-data |
| `ollama` | Deployment | 11434 | — |

### Network Policies

Default deny all ingress in the namespace. Granular allow rules:

| Service | Allowed Ingress From | Allowed Egress To |
|---|---|---|
| Frontend | Anywhere (8080) | Chatbot (8000) |
| Chatbot | Frontend, Admin | Qdrant (6333), Ollama (11434), external HTTPS (443) |
| Admin | Port-forward (8080) | MinIO (9000), Qdrant (6333), Embedder (8080), Chatbot (8000), Ollama (11434), external HTTP/HTTPS |
| Embedder | Admin (8080) | MinIO (9000), Qdrant (6333), external HTTPS (model download) |
| Qdrant | Chatbot, Admin, Embedder, Pipeline pods | — |
| MinIO | Admin, Embedder, Pipeline pods | — |
| Ollama | Chatbot, Admin | — |
| Pipeline pods | — | MinIO (9000), Qdrant (6333), external HTTP/HTTPS (crawling) |

### RBAC

Admin service has a dedicated `ServiceAccount` with RBAC rules to list/get pods, services, and deployments within `test` namespace — used by the health check and service log viewer endpoints.

---

## Key Design Decisions

**Why hybrid retrieval (vector + keyword)?**
Pure vector search fails on proper nouns, misspelled names, and specific identifiers (course numbers, faculty names typed incorrectly). Keyword search catches these but has no semantic understanding of intent. RRF fusion combines both: semantic relevance from vectors, exact-match recall from keywords.

**Why adjacent chunk expansion?**
Chunking at heading boundaries sometimes separates tightly related content. A faculty bio may be in chunk `_0001` but contact details in chunk `_0000`. Fetching ±1 chunks from the same page restores this context without making chunks larger (which would dilute embedding precision by mixing too many topics into one vector).

**Why BGE with asymmetric prefix?**
BAAI/bge-large-en-v1.5 uses asymmetric retrieval: documents are indexed without a prefix, but queries use `"Represent this sentence for searching relevant passages: "`. This is the correct setup for Q&A over a document corpus — the model is trained to map queries and passages to the same embedding space using this asymmetric approach.

**Why context enrichment before embedding?**
Prepending `Title: ...`, `Section: ...`, `Path: ...` before embedding gives the model structural context. A chunk about "Admission Deadlines" under "Graduate Programs > MS Thesis" embeds in a different region than the same text appearing in an undergraduate context, improving precision.

**Why stale seminar filtering in the chunker?**
Seminar announcement pages are time-sensitive and expire. Without filtering, old seminar pages pollute the knowledge base with past dates and speakers. The LLM might confidently return stale event information, which damages trust. The filter removes pages whose event date is older than 6 months, keeping only current and upcoming events.

**Why a separate query rewriter?**
Pronoun resolution requires only the last 1–2 conversation turns. Sending the full history to a rewriter causes it to inject irrelevant old context into standalone questions, corrupting retrieval. A dedicated rewriter with a 4-message window and temperature=0.0 produces minimal, deterministic rewrites.

**Why keyword stop-word filtering?**
Without stop-word filtering, tokens like "the", "is", "what" match nearly every chunk in the collection. These high-frequency tokens corrupt RRF ranking by causing generic hub pages (which mention every common word) to rank first, pushing specific content down.

---

## Key Numbers

| Metric | Value |
|---|---|
| Embedding model | BAAI/bge-large-en-v1.5 |
| Embedding dimensions | 1,024 |
| Similarity metric | Cosine |
| Chunk target size | 400 tokens |
| Chunk overlap | 64 tokens |
| Chunk minimum size | 120 tokens |
| Top-K retrieval | 5 chunks |
| Min cosine score (vector-only) | 0.53 |
| RRF k parameter | 60 |
| Keyword search candidate limit | 10 |
| Adjacent chunks expanded per result | ±1 (up to 2 per vector result) |
| Max conversation history (LLM) | 20 messages |
| Rewriter history window | 4 messages |
| LLM temperature (answer) | 0.3 |
| LLM temperature (rewriter) | 0.0 |
| Crawl depth | 4 levels |
| Crawler concurrency | 5 browser tabs |
| Rate limit | 100 req/hr/session |
| Stale page threshold | 180 days |
