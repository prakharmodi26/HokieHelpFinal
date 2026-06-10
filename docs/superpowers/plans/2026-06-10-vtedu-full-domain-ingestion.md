# vt.edu Full-Domain Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the crawler from cs.vt.edu-only to a discovery-first, resumable ingestion system that covers the entire vt.edu web presence (all subdomains) plus one hop of external links, with generic template-agnostic content extraction and richer metadata flowing through to retrieval.

**Architecture:** Replace the single-seed BFS with a three-stage crawler — **discover** (enumerate subdomains via crt.sh + per-host URL seeding via Crawl4AI `AsyncUrlSeeder`) → **frontier** (durable resumable SQLite queue with hop history) → **render+extract** (`arun_many` + `MemoryAdaptiveDispatcher` + `PruningContentFilter.fit_markdown`, with a JS/SPA retry tier). VT pages route to the existing cleaned bucket (chunker unchanged); one-hop external HTML routes to a separate review bucket; PDF/DOCX are ingested wherever hosted. Richer metadata (source_host, subdomain, hop_history, page_type) propagates frontmatter → chunker → embedder → Qdrant payload so retrieval can filter by unit/type.

**Tech Stack:** Python 3.11, Crawl4AI 0.8 (`AsyncUrlSeeder`, `arun_many`, `MemoryAdaptiveDispatcher`, `RateLimiter`, `PruningContentFilter`), MinIO, SQLite (frontier), pytest + pytest-asyncio, Kubernetes (Jobs), Qdrant.

**Rollout (per product decisions):**
1. Bounded validation pass on the MacBook (cap depth + pages-per-host, ~10 high-value subdomains) to prove discover→render→clean→verify end-to-end.
2. Full-domain test run in a dedicated k8s **test pod** writing to scratch buckets.
3. Verify completeness + content quality. **Promotion to the main buckets/collection is performed by the repo owner, not the executor** — the executor stops after Task 7.2 verification and hands off.

**On crawl depth and full-domain coverage (read before setting `CRAWL_MAX_DEPTH`):**
In this design, breadth/coverage does **not** come from deep BFS — it comes from **discovery** (Phase 2): crt.sh subdomain enumeration + per-host `AsyncUrlSeeder` (sitemap.xml + Common Crawl URL index). Those enumerate URLs directly and enqueue them at depth 0, independent of `max_depth`. `max_depth` only bounds **link-following beyond the seeded set** — the backstop that catches pages missing from sitemaps/CC (many VT sites have no sitemap, e.g. cs.vt.edu). So:
- **Default `CRAWL_MAX_DEPTH = 10`** — high enough to act as a full backstop on sitemap-less hosts, while still finite to avoid infinite crawler traps (calendars, faceted search, session URLs).
- The real coverage levers are: complete discovery (crt.sh ∪ CC ∪ seeding), `CRAWL_MAX_URLS_PER_HOST = -1` (unlimited) in production, and trap guards (`filter_nonsense_urls=True` in the seeder, `CRAWL_BLOCKED_PATHS`, and query-string normalization in `metadata._normalise_url`).
- Bounded validation runs override this to `CRAWL_MAX_DEPTH=2` for speed; the full k8s run uses `10`.
- If a verification run shows missed pages on a specific sitemap-less host, raise depth for that run rather than globally — depth is the safety net, not the primary mechanism.

**Scope decisions locked:**
- Hosts: all `*.vt.edu` (full ingest) **+ one-hop external** pages, stored **separately** in `crawled-pages-external` for later review, never auto-fed to RAG.
- Every stored page records its **full hop history** (referrer chain from a vt.edu seed).
- External **documents** (PDF/DOCX) are ingested into the main corpus regardless of host.
- Cleaning is **generic** (`fit_markdown` via `PruningContentFilter`), storing both raw + fit. No new hand-tuned regex.

---

## Research foundation (verified June 2026, cite when in doubt)

- crt.sh JSON API for subdomains: `https://crt.sh/?q=%25.vt.edu&output=json` → live count **3,219 distinct subdomains**. Field `name_value` holds newline-separated SANs incl. wildcards.
- Common Crawl whole-tree query: `url=vt.edu&matchType=domain` (the `*.vt.edu` wildcard is a trap — only matches apex).
- Crawl4AI `AsyncUrlSeeder` / `SeedingConfig(source="sitemap+cc", extract_head=True, live_check=True, filter_nonsense_urls=True, hits_per_sec=5)`; returns `[{url, status, head_data, relevance_score}]`; handles sitemap index files. Fallback to direct `robots.txt`/`/sitemap.xml` (issues #1747/#1353).
- SPA/lazy pages: `CrawlerRunConfig(wait_for="css:...", scan_full_page=True, scroll_delay=0.5, wait_for_images=True, delay_before_return_html=1.0, page_timeout=90000)`. `text_mode`/`light_mode` do **not** disable JS (safe speed wins); `java_script_enabled=False` disables JS.
- Generic cleaning: `DefaultMarkdownGenerator(content_filter=PruningContentFilter(threshold=0.5, threshold_type="dynamic"))`; read `result.markdown.fit_markdown` (cleaned) and `result.markdown.raw_markdown` (full). Store both.
- Scale/politeness: `CrawlerRunConfig(check_robots_txt=True)` (default False — enable); `arun_many(..., dispatcher=MemoryAdaptiveDispatcher(memory_threshold_percent=70, max_session_permit=10, rate_limiter=RateLimiter(base_delay=(1,3), max_delay=60, rate_limit_codes=[429,503])), stream=True)`. Parallelize across hosts, bound per-host; checkpoint each streamed result (resumability is our responsibility).

---

## Current system context (before this plan — for a fresh executor with zero conversation history)

The ingestion → RAG pipeline is a set of Python services under `services/`, each a `uv`-managed package (Bun is not used here; this is the one Python corner of the stack). A `pipeline` orchestrator runs three Kubernetes Jobs **sequentially**: crawler → chunker → embedder.

- **crawler** (`services/crawler`): Crawl4AI `BFSDeepCrawlStrategy` from a single seed (`https://cs.vt.edu`), headless Chromium, `DomainFilter` restricting to `cs.vt.edu` + subdomains. Writes raw markdown + a `.meta.json` sidecar to MinIO bucket `crawled-pages` (`url_to_object_key` → `<host>/<path>.md`). A post-crawl pass runs `cleaner.py` (hand-tuned cs.vt.edu boilerplate regex — **this plan replaces it with generic `fit_markdown`**) and writes cleaned markdown to `crawled-pages-cleaned`. PDFs/DOCX linked from pages are downloaded + text-extracted (`documents.py`). Dedup is by SHA-256 content hash. Entry point: `crawler.main:cli`.
- **chunker** (`services/chunker`): reads `.md` from `crawled-pages-cleaned`, parses YAML frontmatter (`parser.parse_frontmatter` → `FrontmatterData`), splits into heading-based sections, builds `ChunkRecord`s, writes `chunks/<doc_id>.json` to the `chunks` bucket.
- **embedder** (`services/embedder`): GPU Job; reads chunk JSON, embeds with `BAAI/bge-large-en-v1.5`, upserts into Qdrant collection `hokiehelp_chunks` (`indexer.QdrantIndexer`), payload = chunk metadata + text; deletes stale points per document.
- **retriever** (`k8s/retriever-configmap.yaml`): a tester that embeds queries and logs top-k Qdrant hits with full payload.

Data contract that must be preserved: **the chunker reads `crawled-pages-cleaned`.** So VT pages' cleaned (`fit_markdown`) text and ingested documents must land in `crawled-pages-cleaned` with frontmatter the chunker can parse. One-hop external HTML must NOT land there (it goes to the new `crawled-pages-external` review bucket).

Conventions: Python 3.11; `uv run pytest` per service (run from the service dir, e.g. `cd services/crawler && uv run pytest -v`); tests live in `services/<svc>/tests/`, source in `services/<svc>/src/<svc>/`; MinIO creds come from the `minio-credentials` k8s secret; namespace is `test`; manifests in `k8s/`. Project rule (`.claude/CLAUDE.md`): use the `ctx7` CLI to verify library APIs; show `kubectl config current-context` before applying; never deploy to prod unless explicitly told.

Key files this plan touches (all paths relative to repo root): `services/crawler/src/crawler/{config,crawl,main,cleaner,documents,storage,markdown_doc,metadata}.py`, `services/chunker/src/chunker/{parser,models,splitter}.py`, `services/embedder/src/embedder/indexer.py`, and `k8s/{crawler-configmap,pipeline-configmap}.yaml`.

---

## File structure

New and modified files, by responsibility. All paths are under the repo root `/Users/prakharmodi/Projects/HokieHelpFinal/HokieHelp`.

**New — crawler core (`services/crawler/src/crawler/`):**
- `scope.py` — URL classification (vt / external-within-hops / document) + hop-history computation. Pure logic.
- `frontier.py` — durable resumable SQLite frontier (enqueue/lease/mark + hop history + content_hash dedup).
- `discovery/__init__.py`
- `discovery/subdomains.py` — crt.sh (+ optional CC) subdomain enumeration.
- `discovery/liveness.py` — DNS + HTTP liveness filter for hosts.
- `discovery/seeder.py` — Crawl4AI `AsyncUrlSeeder` wrapper + robots/sitemap fallback.
- `extract.py` — generic markdown generator factory (`PruningContentFilter`) + raw/fit selection + empty-detection.
- `render.py` — `arun_many` engine with dispatcher/rate-limiter, two-tier (fast + JS) rendering, writes results to storage via frontier.
- `verify.py` — completeness + content-quality verification over MinIO buckets vs frontier.

**Modified — crawler:**
- `config.py` — widen domain defaults to `vt.edu`; add discovery/scope/external/frontier/render settings.
- `metadata.py` — enrich `PageMetadata` (source_host, subdomain, scope, is_external, hop_history, discovery_source, content_type, raw/fit sizes, lang, word_count, render_tier, robots_allowed, fetched_at).
- `markdown_doc.py` — add source_host, subdomain, page_type, is_external, last_modified to frontmatter.
- `storage.py` — add `minio_external_bucket`; generalize bucket helpers (already mostly generic).
- `main.py` — new orchestration: discover → frontier → render → (per-bucket) → verify summary. Keep the documents pass; drop the cs.vt.edu post-crawl regex clean for general hosts (fit_markdown replaces it).
- `pyproject.toml` — add `aiosqlite`/stdlib-sqlite note + `tldextract` for robust host parsing.

**New scripts:**
- `services/crawler/scripts/discover_vtedu.py` — standalone subdomain+seed discovery CLI → frontier DB.
- `services/crawler/scripts/verify_ingestion.py` — standalone verification CLI → markdown report.

**Modified — downstream metadata propagation:**
- `services/chunker/src/chunker/parser.py` — `FrontmatterData` += source_host, subdomain, page_type, last_modified; `infer_page_type` reused; read subdomain from host.
- `services/chunker/src/chunker/models.py` — `ChunkRecord` += source_host, subdomain.
- `services/chunker/src/chunker/splitter.py` — pass new fields into `ChunkRecord`.
- `services/embedder/src/embedder/indexer.py` — payload += source_host, subdomain; add KEYWORD payload indexes for source_host, subdomain, page_type.

**Modified — k8s:**
- `k8s/crawler-configmap.yaml` — `vt.edu` scope, discovery + render envs.
- `k8s/pipeline-configmap.yaml` — match new defaults.
- `k8s/crawler-pvc.yaml` (new) — PVC for the frontier SQLite DB so a crawl Job is resumable across restarts.
- `k8s/crawler-test-job.yaml` (new) — full-domain test-pod Job writing to scratch buckets.

---

## Phase 0 — Scope router & richer metadata schema (pure logic, no network)

Foundational deterministic logic everything else builds on. Fully unit-testable offline.

### Task 0.1: Host/scope utilities

**Files:**
- Create: `services/crawler/src/crawler/scope.py`
- Test: `services/crawler/tests/test_scope.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_scope.py
from crawler.scope import (
    Scope,
    registered_domain,
    subdomain_of,
    is_vt_host,
    is_document_url,
    classify_url,
)


def test_registered_domain_and_subdomain():
    assert registered_domain("https://dining.vt.edu/menus") == "vt.edu"
    assert subdomain_of("https://dining.vt.edu/menus") == "dining"
    assert subdomain_of("https://website.cs.vt.edu/About.html") == "website.cs"
    assert subdomain_of("https://vt.edu/") == ""


def test_is_vt_host():
    assert is_vt_host("https://registrar.vt.edu/x") is True
    assert is_vt_host("https://vt.edu/") is True
    assert is_vt_host("https://news.google.com/vt") is False


def test_is_document_url():
    assert is_document_url("https://x.vt.edu/a/checksheet.pdf") is True
    assert is_document_url("https://files.example.com/form.docx") is True
    assert is_document_url("https://x.vt.edu/page.html") is False


def test_classify_vt_page():
    scope, hops = classify_url(
        "https://dining.vt.edu/menus",
        parent_hops=["https://vt.edu/"],
        external_hops=1,
    )
    assert scope is Scope.VT_PAGE
    assert hops == ["https://vt.edu/", "https://dining.vt.edu/menus"]


def test_classify_document_anywhere():
    scope, _ = classify_url(
        "https://cdn.thirdparty.com/vt/form.pdf",
        parent_hops=["https://hr.vt.edu/forms"],
        external_hops=1,
    )
    assert scope is Scope.DOCUMENT


def test_classify_external_within_one_hop():
    # parent is a VT page (0 external hops so far) -> external child allowed, tagged EXTERNAL
    scope, hops = classify_url(
        "https://www.linkedin.com/company/vt-cs",
        parent_hops=["https://vt.edu/", "https://cs.vt.edu/people"],
        external_hops=1,
    )
    assert scope is Scope.EXTERNAL_PAGE
    assert hops[-1] == "https://www.linkedin.com/company/vt-cs"


def test_classify_external_beyond_hop_is_rejected():
    # parent chain already contains one external host -> a further external is OUT_OF_SCOPE
    scope, _ = classify_url(
        "https://medium.com/@someone",
        parent_hops=["https://cs.vt.edu/x", "https://www.linkedin.com/y"],
        external_hops=1,
    )
    assert scope is Scope.OUT_OF_SCOPE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_scope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.scope'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/scope.py
"""URL scope classification and hop-history computation.

Three storage scopes:
  VT_PAGE       — host is vt.edu or a subdomain; full ingest to the main corpus.
  EXTERNAL_PAGE — non-VT HTML reached within `external_hops` of a VT page;
                  stored separately for later review, never auto-fed to RAG.
  DOCUMENT      — a PDF/DOC/DOCX URL on ANY host (VT content often lives on CDNs);
                  ingested to the main corpus.
  OUT_OF_SCOPE  — anything else (e.g. a second external hop); never fetched.
"""
from __future__ import annotations

import enum
from urllib.parse import urlparse

import tldextract

# Use a no-network extractor (no live public-suffix fetch in containers/tests).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

DOCUMENT_EXTENSIONS = (".pdf", ".doc", ".docx")
_VT_REGISTERED = "vt.edu"


class Scope(enum.Enum):
    VT_PAGE = "vt_page"
    EXTERNAL_PAGE = "external_page"
    DOCUMENT = "document"
    OUT_OF_SCOPE = "out_of_scope"


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def registered_domain(url: str) -> str:
    ext = _EXTRACT(_host(url))
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}"


def subdomain_of(url: str) -> str:
    """Return the subdomain label(s) below the registered domain ('' for apex)."""
    return _EXTRACT(_host(url)).subdomain


def is_vt_host(url: str) -> bool:
    return registered_domain(url) == _VT_REGISTERED


def is_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOCUMENT_EXTENSIONS)


def _external_hops_in_chain(hops: list[str]) -> int:
    """Count how many hops in the chain are on non-VT, non-document hosts."""
    return sum(
        1 for u in hops if not is_vt_host(u) and not is_document_url(u)
    )


def classify_url(
    url: str,
    parent_hops: list[str],
    external_hops: int,
) -> tuple[Scope, list[str]]:
    """Classify a URL given its parent's hop chain; return (scope, new_hop_chain).

    new_hop_chain = parent_hops + [url], i.e. the full referrer path to `url`.
    """
    hops = list(parent_hops) + [url]
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        return Scope.OUT_OF_SCOPE, hops

    if is_document_url(url):
        return Scope.DOCUMENT, hops
    if is_vt_host(url):
        return Scope.VT_PAGE, hops

    # External HTML: allowed only if it does not exceed the external-hop budget.
    # parent_hops already includes any external hosts traversed to get here.
    already_external = _external_hops_in_chain(parent_hops)
    if already_external < external_hops:
        return Scope.EXTERNAL_PAGE, hops
    return Scope.OUT_OF_SCOPE, hops
```

- [ ] **Step 4: Add `tldextract` dependency**

Edit `services/crawler/pyproject.toml` — add to `dependencies`:

```toml
    "tldextract>=5.1.0",
```

Then: `cd services/crawler && uv sync`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_scope.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/scope.py services/crawler/tests/test_scope.py services/crawler/pyproject.toml services/crawler/uv.lock
git commit -m "feat(crawler): add URL scope classifier with hop-history tracking"
```

### Task 0.2: Enrich PageMetadata

**Files:**
- Modify: `services/crawler/src/crawler/metadata.py`
- Test: `services/crawler/tests/test_metadata.py` (add cases; create if absent)

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_metadata.py  (append; create file if missing)
from datetime import datetime, timezone

from crawler.metadata import PageMetadata


def _base_kwargs():
    return dict(
        doc_id="abc123",
        url="https://dining.vt.edu/menus",
        title="Menus",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        content_hash="deadbeef",
        markdown_size_bytes=100,
        status_code=200,
        response_headers={"content-type": "text/html"},
        internal_links=[],
        external_links=[],
        last_modified=None,
        etag=None,
    )


def test_metadata_roundtrip_with_new_fields():
    meta = PageMetadata(
        **_base_kwargs(),
        source_host="dining.vt.edu",
        subdomain="dining",
        scope="vt_page",
        is_external=False,
        hop_history=["https://vt.edu/", "https://dining.vt.edu/menus"],
        discovery_source="sitemap",
        content_type="text/html",
        raw_markdown_size_bytes=500,
        fit_markdown_size_bytes=100,
        lang="en",
        word_count=42,
        render_tier="fast",
        robots_allowed=True,
        fetched_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    restored = PageMetadata.from_json(meta.to_json())
    assert restored.source_host == "dining.vt.edu"
    assert restored.subdomain == "dining"
    assert restored.hop_history[-1] == "https://dining.vt.edu/menus"
    assert restored.render_tier == "fast"
    assert restored.is_external is False


def test_metadata_new_fields_default_when_absent():
    # Old sidecars (pre-upgrade) must still deserialise.
    meta = PageMetadata(**_base_kwargs())
    restored = PageMetadata.from_json(meta.to_json())
    assert restored.source_host is None
    assert restored.hop_history == []
    assert restored.is_external is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_metadata.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'source_host'`

- [ ] **Step 3: Write minimal implementation**

In `services/crawler/src/crawler/metadata.py`, extend the dataclass. Add these fields after `etag` (all defaulted so old sidecars still load):

```python
    # --- enrichment for full-domain ingestion (all optional/defaulted) ---
    source_host: Optional[str] = None
    subdomain: Optional[str] = None
    scope: str = "vt_page"
    is_external: bool = False
    hop_history: list[str] = field(default_factory=list)
    discovery_source: Optional[str] = None
    content_type: Optional[str] = None
    raw_markdown_size_bytes: Optional[int] = None
    fit_markdown_size_bytes: Optional[int] = None
    lang: Optional[str] = None
    word_count: Optional[int] = None
    render_tier: str = "fast"
    robots_allowed: bool = True
    fetched_at: Optional[datetime] = None
```

Add `from dataclasses import dataclass, field` to the imports (currently only `dataclass`). Then update `to_json` to include the new keys:

```python
            "source_host": self.source_host,
            "subdomain": self.subdomain,
            "scope": self.scope,
            "is_external": self.is_external,
            "hop_history": self.hop_history,
            "discovery_source": self.discovery_source,
            "content_type": self.content_type,
            "raw_markdown_size_bytes": self.raw_markdown_size_bytes,
            "fit_markdown_size_bytes": self.fit_markdown_size_bytes,
            "lang": self.lang,
            "word_count": self.word_count,
            "render_tier": self.render_tier,
            "robots_allowed": self.robots_allowed,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
```

And update `from_json` to read them with defaults:

```python
            source_host=data.get("source_host"),
            subdomain=data.get("subdomain"),
            scope=data.get("scope", "vt_page"),
            is_external=data.get("is_external", False),
            hop_history=data.get("hop_history", []),
            discovery_source=data.get("discovery_source"),
            content_type=data.get("content_type"),
            raw_markdown_size_bytes=data.get("raw_markdown_size_bytes"),
            fit_markdown_size_bytes=data.get("fit_markdown_size_bytes"),
            lang=data.get("lang"),
            word_count=data.get("word_count"),
            render_tier=data.get("render_tier", "fast"),
            robots_allowed=data.get("robots_allowed", True),
            fetched_at=(
                datetime.fromisoformat(data["fetched_at"])
                if data.get("fetched_at") else None
            ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_metadata.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/metadata.py services/crawler/tests/test_metadata.py
git commit -m "feat(crawler): enrich PageMetadata with host/scope/hop-history fields"
```

### Task 0.3: Richer frontmatter

**Files:**
- Modify: `services/crawler/src/crawler/markdown_doc.py`
- Test: `services/crawler/tests/test_markdown_doc.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_markdown_doc.py  (append)
from datetime import datetime, timezone
from crawler.markdown_doc import build_markdown_document


def test_frontmatter_includes_host_and_page_type():
    doc = build_markdown_document(
        url="https://dining.vt.edu/menus",
        title="Menus",
        markdown_content="# Menus\nToday: pizza",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        doc_id="abc123",
        content_hash="deadbeef",
        source_host="dining.vt.edu",
        subdomain="dining",
        page_type="general",
        is_external=False,
        last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
    )
    assert "source_host: 'dining.vt.edu'" in doc
    assert "subdomain: 'dining'" in doc
    assert "page_type: 'general'" in doc
    assert "is_external: false" in doc
    assert doc.endswith("# Menus\nToday: pizza")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_markdown_doc.py -v`
Expected: FAIL — `TypeError: build_markdown_document() got an unexpected keyword argument 'source_host'`

- [ ] **Step 3: Write minimal implementation**

Replace `build_markdown_document` in `services/crawler/src/crawler/markdown_doc.py` with a version accepting the new optional fields and emitting them in the frontmatter (keep existing fields/positions for back-compat):

```python
def build_markdown_document(
    *,
    url: str,
    title: str | None,
    markdown_content: str,
    crawl_depth: int,
    crawl_timestamp: datetime,
    doc_id: str,
    content_hash: str,
    source_host: str | None = None,
    subdomain: str | None = None,
    page_type: str = "general",
    is_external: bool = False,
    last_modified: str | None = None,
) -> str:
    """Return a Markdown string with YAML frontmatter metadata."""
    effective_title = title if title else url
    ts = crawl_timestamp.isoformat()

    lines = [
        "---",
        f"doc_id: '{doc_id}'",
        f"url: '{url}'",
        f"title: '{_yaml_escape(effective_title)}'",
        f"crawl_depth: {crawl_depth}",
        f"crawl_timestamp: '{ts}'",
        f"content_hash: '{content_hash}'",
        f"source_host: '{source_host or ''}'",
        f"subdomain: '{subdomain or ''}'",
        f"page_type: '{page_type}'",
        f"is_external: {'true' if is_external else 'false'}",
        f"last_modified: '{last_modified or ''}'",
        "---",
    ]
    return "\n".join(lines) + f"\n\n{markdown_content}"


def _yaml_escape(value: str) -> str:
    """Escape single quotes for single-quoted YAML scalars."""
    return value.replace("'", "''")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_markdown_doc.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole crawler suite to catch callers**

Run: `cd services/crawler && uv run pytest -v`
Expected: existing callers of `build_markdown_document` in `crawl.py`/`documents.py` still pass because new params are optional. If any test references removed behaviour, fix the call site to pass the new optional kwargs. Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/markdown_doc.py services/crawler/tests/test_markdown_doc.py
git commit -m "feat(crawler): add host/page_type/external fields to frontmatter"
```

---

## Phase 1 — Durable resumable frontier (SQLite, pure logic)

The frontier is the checkpoint that makes a million-URL crawl resumable. Unit-testable with a temp DB file; no network.

### Task 1.1: Frontier store

**Files:**
- Create: `services/crawler/src/crawler/frontier.py`
- Test: `services/crawler/tests/test_frontier.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_frontier.py
import hashlib

from crawler.frontier import Frontier, FrontierItem


def _digest(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def test_enqueue_dedup_and_counts(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    added1 = db.enqueue("https://vt.edu/", scope="vt_page", depth=0,
                        hop_history=["https://vt.edu/"], discovery_source="seed")
    added2 = db.enqueue("https://vt.edu/", scope="vt_page", depth=0,
                        hop_history=["https://vt.edu/"], discovery_source="seed")
    assert added1 is True
    assert added2 is False  # same URL digest -> not re-added
    assert db.count_by_status()["pending"] == 1


def test_lease_and_mark_done(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=1,
               hop_history=["https://vt.edu/", "https://a.vt.edu/x"],
               discovery_source="bfs")
    batch = db.lease_pending(limit=10)
    assert len(batch) == 1
    item = batch[0]
    assert isinstance(item, FrontierItem)
    assert item.url == "https://a.vt.edu/x"
    assert item.hop_history[-1] == "https://a.vt.edu/x"
    # leased items are no longer pending
    assert db.count_by_status().get("pending", 0) == 0
    assert db.count_by_status()["leased"] == 1

    db.mark_done("https://a.vt.edu/x", content_hash="abc", http_status=200)
    counts = db.count_by_status()
    assert counts["done"] == 1
    assert counts.get("leased", 0) == 0


def test_mark_failed_and_requeue(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    db.lease_pending(limit=10)
    db.mark_failed("https://a.vt.edu/x", error="timeout")
    assert db.count_by_status()["failed"] == 1
    # requeue resets failed -> pending for a retry pass
    n = db.requeue_failed()
    assert n == 1
    assert db.count_by_status()["pending"] == 1


def test_content_hash_seen_dedup(tmp_path):
    db = Frontier(str(tmp_path / "f.db"))
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    db.lease_pending(limit=10)
    db.mark_done("https://a.vt.edu/x", content_hash="HASH1", http_status=200)
    assert db.content_hash_seen("HASH1") is True
    assert db.content_hash_seen("NOPE") is False


def test_resumability_reopen(tmp_path):
    path = str(tmp_path / "f.db")
    db = Frontier(path)
    db.enqueue("https://a.vt.edu/x", scope="vt_page", depth=0,
               hop_history=["https://a.vt.edu/x"], discovery_source="seed")
    del db
    db2 = Frontier(path)  # reopen same file
    assert db2.count_by_status()["pending"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_frontier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.frontier'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/frontier.py
"""Durable, resumable crawl frontier backed by SQLite.

One row per URL (keyed by a 16-hex digest). Status lifecycle:
  pending -> leased -> done | failed
Stores the full hop_history (referrer chain) and the content_hash of stored
pages so a re-run skips unchanged content. The DB file lives on a PVC in k8s so
a restarted crawl Job resumes instead of starting over.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass


def url_digest(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@dataclass
class FrontierItem:
    url: str
    scope: str
    depth: int
    hop_history: list[str]
    discovery_source: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    digest           TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    host             TEXT,
    scope            TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    depth            INTEGER NOT NULL DEFAULT 0,
    hop_history      TEXT NOT NULL DEFAULT '[]',
    discovery_source TEXT,
    content_hash     TEXT,
    http_status      INTEGER,
    error            TEXT,
    updated_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON frontier(status);
CREATE INDEX IF NOT EXISTS idx_content_hash ON frontier(content_hash);
"""


class Frontier:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def enqueue(
        self,
        url: str,
        *,
        scope: str,
        depth: int,
        hop_history: list[str],
        discovery_source: str,
    ) -> bool:
        """Insert a new URL. Returns False if the digest already exists."""
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        try:
            self._conn.execute(
                "INSERT INTO frontier(digest, url, host, scope, depth, hop_history, discovery_source) "
                "VALUES(?,?,?,?,?,?,?)",
                (url_digest(url), url, host, scope, depth,
                 json.dumps(hop_history), discovery_source),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def lease_pending(self, limit: int) -> list[FrontierItem]:
        """Atomically move up to `limit` pending rows to 'leased' and return them."""
        cur = self._conn.execute(
            "SELECT digest, url, scope, depth, hop_history, discovery_source "
            "FROM frontier WHERE status='pending' ORDER BY depth ASC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        digests = [r[0] for r in rows]
        if digests:
            self._conn.executemany(
                "UPDATE frontier SET status='leased', updated_at=datetime('now') WHERE digest=?",
                [(d,) for d in digests],
            )
            self._conn.commit()
        return [
            FrontierItem(
                url=r[1], scope=r[2], depth=r[3],
                hop_history=json.loads(r[4]), discovery_source=r[5],
            )
            for r in rows
        ]

    def mark_done(self, url: str, *, content_hash: str, http_status: int) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='done', content_hash=?, http_status=?, "
            "updated_at=datetime('now') WHERE digest=?",
            (content_hash, http_status, url_digest(url)),
        )
        self._conn.commit()

    def mark_failed(self, url: str, *, error: str, http_status: int | None = None) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='failed', error=?, http_status=?, "
            "updated_at=datetime('now') WHERE digest=?",
            (error[:500], http_status, url_digest(url)),
        )
        self._conn.commit()

    def mark_skipped(self, url: str, *, reason: str) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='skipped', error=?, updated_at=datetime('now') WHERE digest=?",
            (reason[:500], url_digest(url)),
        )
        self._conn.commit()

    def requeue_failed(self) -> int:
        cur = self._conn.execute(
            "UPDATE frontier SET status='pending', error=NULL WHERE status='failed'"
        )
        self._conn.commit()
        return cur.rowcount

    def content_hash_seen(self, content_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM frontier WHERE content_hash=? LIMIT 1", (content_hash,)
        )
        return cur.fetchone() is not None

    def count_by_status(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM frontier GROUP BY status"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def iter_status(self, status: str):
        """Yield (url, host, scope, content_hash, http_status) for a status."""
        cur = self._conn.execute(
            "SELECT url, host, scope, content_hash, http_status FROM frontier WHERE status=?",
            (status,),
        )
        yield from cur.fetchall()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_frontier.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/frontier.py services/crawler/tests/test_frontier.py
git commit -m "feat(crawler): add resumable SQLite frontier with hop history"
```

---

## Phase 2 — Discovery (subdomains + per-host URL seeding)

Cheap URL discovery before paying full-render cost. Parsers are unit-tested offline with fixtures; one live smoke test is gated behind a marker.

### Task 2.1: Subdomain enumeration (crt.sh parsing)

**Files:**
- Create: `services/crawler/src/crawler/discovery/__init__.py` (empty)
- Create: `services/crawler/src/crawler/discovery/subdomains.py`
- Test: `services/crawler/tests/test_subdomains.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_subdomains.py
from crawler.discovery.subdomains import parse_crtsh_json, normalize_hosts


def test_parse_crtsh_json_splits_sans_and_strips_wildcards():
    raw = [
        {"common_name": "dining.vt.edu",
         "name_value": "dining.vt.edu\n*.dining.vt.edu"},
        {"common_name": "registrar.vt.edu",
         "name_value": "registrar.vt.edu"},
        {"common_name": "notvt.example.com",
         "name_value": "notvt.example.com"},
    ]
    hosts = parse_crtsh_json(raw)
    assert "dining.vt.edu" in hosts
    assert "registrar.vt.edu" in hosts
    # wildcard stripped to its base
    assert "dining.vt.edu" in hosts
    # non vt.edu dropped
    assert "notvt.example.com" not in hosts


def test_normalize_hosts_dedups_lowercases():
    out = normalize_hosts(["Dining.VT.edu", "dining.vt.edu", "*.cs.vt.edu", "vt.edu"])
    assert out == sorted({"dining.vt.edu", "cs.vt.edu", "vt.edu"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_subdomains.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.discovery.subdomains'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/discovery/subdomains.py
"""Enumerate vt.edu subdomains.

Primary source: Certificate Transparency via crt.sh JSON API. The crt.sh
`name_value` field can hold multiple newline-separated SANs including wildcards
(`*.foo.vt.edu`). We split, strip wildcards, lowercase, keep only *.vt.edu, dedup.

Live fetch is isolated in `fetch_subdomains()`; parsing is pure for testing.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

CRTSH_URL = "https://crt.sh/?q=%25.vt.edu&output=json"
_VT_SUFFIX = "vt.edu"


def normalize_hosts(hosts: list[str]) -> list[str]:
    out: set[str] = set()
    for h in hosts:
        h = h.strip().lower().lstrip("*.")
        if not h:
            continue
        if h == _VT_SUFFIX or h.endswith("." + _VT_SUFFIX):
            out.add(h)
    return sorted(out)


def parse_crtsh_json(records: list[dict]) -> list[str]:
    raw: list[str] = []
    for rec in records:
        name_value = rec.get("name_value", "") or ""
        raw.extend(name_value.split("\n"))
        cn = rec.get("common_name")
        if cn:
            raw.append(cn)
    return normalize_hosts(raw)


def fetch_subdomains(timeout: float = 60.0) -> list[str]:
    """Live query crt.sh and return normalized *.vt.edu hosts."""
    logger.info("Querying crt.sh for vt.edu subdomains")
    resp = httpx.get(CRTSH_URL, timeout=timeout,
                     headers={"User-Agent": "HokieHelp-Crawler/2.0"})
    resp.raise_for_status()
    hosts = parse_crtsh_json(resp.json())
    logger.info("crt.sh returned %d distinct vt.edu subdomains", len(hosts))
    return hosts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_subdomains.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/discovery/__init__.py services/crawler/src/crawler/discovery/subdomains.py services/crawler/tests/test_subdomains.py
git commit -m "feat(crawler): crt.sh subdomain enumeration for vt.edu"
```

### Task 2.2: Host liveness filter

**Files:**
- Create: `services/crawler/src/crawler/discovery/liveness.py`
- Test: `services/crawler/tests/test_liveness.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_liveness.py
import asyncio
from unittest.mock import AsyncMock

from crawler.discovery import liveness


def test_filter_live_hosts_keeps_only_reachable(monkeypatch):
    async def fake_probe(client, host):
        return host != "dead.vt.edu"

    monkeypatch.setattr(liveness, "_probe_host", AsyncMock(side_effect=fake_probe))
    hosts = ["alive.vt.edu", "dead.vt.edu", "vt.edu"]
    live = asyncio.run(liveness.filter_live_hosts(hosts, concurrency=2))
    assert "alive.vt.edu" in live
    assert "vt.edu" in live
    assert "dead.vt.edu" not in live
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_liveness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.discovery.liveness'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/discovery/liveness.py
"""Filter a host list down to those that actually resolve and respond over HTTP.

CT logs include dead/staging/internal hosts; probe each with a short HEAD/GET
(HTTPS first, HTTP fallback) before committing it to the crawl frontier.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


async def _probe_host(client: httpx.AsyncClient, host: str) -> bool:
    for scheme in ("https", "http"):
        url = f"{scheme}://{host}/"
        try:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code < 500:
                return True
        except Exception:
            continue
    return False


async def filter_live_hosts(hosts: list[str], concurrency: int = 20,
                            timeout: float = 10.0) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    live: list[str] = []

    async with httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": "HokieHelp-Crawler/2.0"}
    ) as client:
        async def worker(host: str) -> None:
            async with sem:
                if await _probe_host(client, host):
                    live.append(host)

        await asyncio.gather(*(worker(h) for h in hosts))

    logger.info("Liveness: %d/%d hosts reachable", len(live), len(hosts))
    return sorted(live)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_liveness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/discovery/liveness.py services/crawler/tests/test_liveness.py
git commit -m "feat(crawler): HTTP liveness filter for discovered hosts"
```

### Task 2.3: Per-host URL seeding (Crawl4AI AsyncUrlSeeder wrapper)

**Files:**
- Create: `services/crawler/src/crawler/discovery/seeder.py`
- Test: `services/crawler/tests/test_seeder.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_seeder.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from crawler.discovery.seeder import seed_hosts, _extract_urls


def test_extract_urls_filters_valid_and_keeps_head_lang():
    raw = [
        {"url": "https://dining.vt.edu/a", "status": "valid",
         "head_data": {"lang": "en", "title": "A"}},
        {"url": "https://dining.vt.edu/b", "status": "not_valid"},
        {"url": "https://dining.vt.edu/c", "status": "unknown"},
    ]
    urls = _extract_urls(raw, require_valid=True)
    assert urls == {"https://dining.vt.edu/a"}
    urls_any = _extract_urls(raw, require_valid=False)
    assert "https://dining.vt.edu/c" in urls_any


def test_seed_hosts_aggregates_many_urls():
    fake_seeder = MagicMock()
    fake_seeder.__aenter__ = AsyncMock(return_value=fake_seeder)
    fake_seeder.__aexit__ = AsyncMock(return_value=False)
    fake_seeder.many_urls = AsyncMock(return_value={
        "dining.vt.edu": [{"url": "https://dining.vt.edu/a", "status": "valid"}],
        "registrar.vt.edu": [{"url": "https://registrar.vt.edu/x", "status": "valid"}],
    })
    with patch("crawler.discovery.seeder.AsyncUrlSeeder", return_value=fake_seeder):
        result = asyncio.run(seed_hosts(["dining.vt.edu", "registrar.vt.edu"]))
    assert "https://dining.vt.edu/a" in result
    assert "https://registrar.vt.edu/x" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_seeder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.discovery.seeder'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/discovery/seeder.py
"""Discover URLs per host using Crawl4AI's AsyncUrlSeeder (sitemap + Common Crawl).

Cheap bulk URL discovery before crawling. Uses source="sitemap+cc" which reads
sitemap.xml (and robots.txt Sitemap: directives) plus the Common Crawl URL index.
filter_nonsense_urls drops media/admin/API junk; live_check verifies reachability.
"""
from __future__ import annotations

import logging

from crawl4ai import AsyncUrlSeeder, SeedingConfig

logger = logging.getLogger(__name__)


def _extract_urls(records: list[dict], require_valid: bool) -> set[str]:
    out: set[str] = set()
    for rec in records:
        url = rec.get("url")
        if not url:
            continue
        if require_valid and rec.get("status") not in ("valid", "unknown"):
            continue
        out.add(url)
    return out


async def seed_hosts(
    hosts: list[str],
    *,
    source: str = "sitemap+cc",
    extract_head: bool = True,
    live_check: bool = True,
    hits_per_sec: int = 5,
    concurrency: int = 10,
    max_urls_per_host: int = -1,
    require_valid: bool = True,
) -> set[str]:
    """Return a deduped set of URLs discovered across all hosts."""
    cfg = SeedingConfig(
        source=source,
        extract_head=extract_head,
        live_check=live_check,
        filter_nonsense_urls=True,
        concurrency=concurrency,
        hits_per_sec=hits_per_sec,
        max_urls=max_urls_per_host,
    )
    all_urls: set[str] = set()
    async with AsyncUrlSeeder() as seeder:
        per_host = await seeder.many_urls(hosts, cfg)
    for host, records in per_host.items():
        urls = _extract_urls(records, require_valid=require_valid)
        logger.info("Seeded %d URLs from %s", len(urls), host)
        all_urls |= urls
    logger.info("Seeding complete: %d total URLs across %d hosts",
                len(all_urls), len(hosts))
    return all_urls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_seeder.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/discovery/seeder.py services/crawler/tests/test_seeder.py
git commit -m "feat(crawler): per-host URL seeding via Crawl4AI AsyncUrlSeeder"
```

### Task 2.4: Discovery orchestrator script (subdomains → liveness → seed → frontier)

**Files:**
- Create: `services/crawler/src/crawler/discovery/run.py`
- Create: `services/crawler/scripts/discover_vtedu.py`
- Test: `services/crawler/tests/test_discovery_run.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_discovery_run.py
import asyncio
from unittest.mock import AsyncMock, patch

from crawler.discovery import run as drun
from crawler.frontier import Frontier


def test_populate_frontier_enqueues_seed_and_discovered(tmp_path, monkeypatch):
    monkeypatch.setattr(drun, "fetch_subdomains", lambda: ["dining.vt.edu", "vt.edu"])
    monkeypatch.setattr(
        drun, "filter_live_hosts",
        AsyncMock(return_value=["dining.vt.edu", "vt.edu"]),
    )
    monkeypatch.setattr(
        drun, "seed_hosts",
        AsyncMock(return_value={
            "https://dining.vt.edu/menus",
            "https://vt.edu/",
        }),
    )
    db_path = str(tmp_path / "f.db")
    asyncio.run(drun.populate_frontier(db_path, max_hosts=None, max_urls_per_host=-1))

    fr = Frontier(db_path)
    counts = fr.count_by_status()
    assert counts["pending"] == 2  # both discovered URLs enqueued, deduped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_discovery_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.discovery.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/discovery/run.py
"""Discovery orchestrator: subdomains -> liveness -> per-host seeding -> frontier."""
from __future__ import annotations

import logging

from crawler.discovery.liveness import filter_live_hosts
from crawler.discovery.seeder import seed_hosts
from crawler.discovery.subdomains import fetch_subdomains
from crawler.frontier import Frontier
from crawler.scope import classify_url

logger = logging.getLogger(__name__)


async def populate_frontier(
    db_path: str,
    *,
    max_hosts: int | None,
    max_urls_per_host: int,
    external_hops: int = 1,
) -> dict:
    """Discover all vt.edu URLs and enqueue them into the frontier.

    Returns a stats dict. Seed URLs are tagged discovery_source per origin.
    """
    hosts = fetch_subdomains()
    if max_hosts is not None:
        hosts = hosts[:max_hosts]
    live = await filter_live_hosts(hosts)
    urls = await seed_hosts(live, max_urls_per_host=max_urls_per_host)

    fr = Frontier(db_path)
    added = 0
    for url in sorted(urls):
        scope, hops = classify_url(url, parent_hops=[], external_hops=external_hops)
        if fr.enqueue(url, scope=scope.value, depth=0,
                      hop_history=hops, discovery_source="seed"):
            added += 1
    stats = {"hosts_discovered": len(hosts), "hosts_live": len(live),
             "urls_seeded": len(urls), "urls_enqueued": added}
    logger.info("Discovery complete: %s", stats)
    fr.close()
    return stats
```

```python
# services/crawler/scripts/discover_vtedu.py
#!/usr/bin/env python3
"""Standalone discovery CLI: build/refresh the crawl frontier for all of vt.edu.

    uv run python scripts/discover_vtedu.py --db frontier.db [--max-hosts N] [--max-urls-per-host N]
"""
import argparse
import asyncio
import logging

from crawler.discovery.run import populate_frontier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Discover all vt.edu URLs into a frontier DB")
    p.add_argument("--db", default="frontier.db")
    p.add_argument("--max-hosts", type=int, default=None,
                   help="Cap number of subdomains (bounded validation runs)")
    p.add_argument("--max-urls-per-host", type=int, default=-1,
                   help="-1 = unlimited")
    args = p.parse_args()
    stats = asyncio.run(populate_frontier(
        args.db, max_hosts=args.max_hosts, max_urls_per_host=args.max_urls_per_host,
    ))
    print(stats)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_discovery_run.py -v`
Expected: PASS

- [ ] **Step 5: Live smoke test (manual, optional but recommended now)**

Run: `cd services/crawler && uv run python scripts/discover_vtedu.py --db /tmp/vt_frontier.db --max-hosts 10 --max-urls-per-host 200`
Expected: prints a stats dict with `hosts_discovered` in the thousands, `hosts_live` ≤ 10, `urls_enqueued` > 0. Inspect:
`sqlite3 /tmp/vt_frontier.db "SELECT status, COUNT(*) FROM frontier GROUP BY status;"`

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/discovery/run.py services/crawler/scripts/discover_vtedu.py services/crawler/tests/test_discovery_run.py
git commit -m "feat(crawler): discovery orchestrator + CLI populating the frontier"
```

---

## Phase 3 — Generic extraction + two-tier render engine

Replaces the cs.vt.edu-specific cleaner with `PruningContentFilter.fit_markdown` and adds a JS/SPA retry tier. Pure-logic parts unit-tested; the crawl loop tested with a mocked crawler.

### Task 3.1: Generic extraction helpers

**Files:**
- Create: `services/crawler/src/crawler/extract.py`
- Test: `services/crawler/tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_extract.py
from crawler.extract import (
    make_markdown_generator,
    select_content,
    looks_empty,
    word_count,
)


def test_make_markdown_generator_uses_pruning_filter():
    gen = make_markdown_generator(threshold=0.5)
    # the generator carries a content filter; just assert it constructs
    assert gen is not None


def test_select_content_prefers_fit_over_raw():
    class FakeMd:
        fit_markdown = "# Real content\nlots of words here"
        raw_markdown = "# Real content\nlots of words here\n[nav][nav][nav]"
    text, used = select_content(FakeMd())
    assert used == "fit"
    assert text.startswith("# Real content")


def test_select_content_falls_back_to_raw_when_fit_empty():
    class FakeMd:
        fit_markdown = "   "
        raw_markdown = "# Something\nbody"
    text, used = select_content(FakeMd())
    assert used == "raw"
    assert "Something" in text


def test_looks_empty_and_word_count():
    assert looks_empty("") is True
    assert looks_empty("   \n  ") is True
    assert looks_empty("# Title\nApp loading...") is True  # below min words
    assert looks_empty("# Title\n" + "word " * 40) is False
    assert word_count("one two three") == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/extract.py
"""Generic, template-agnostic content extraction.

Uses Crawl4AI's PruningContentFilter to strip nav/footer/boilerplate by scoring
text density / link density / tag importance — no per-site regex. We store both
raw_markdown and fit_markdown; fit is the cleaned text fed to the chunker.
`looks_empty` flags pages that came back without real content (typical of SPAs
that need JS rendering) so the render engine can retry them in the JS tier.
"""
from __future__ import annotations

import re

from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

MIN_CONTENT_WORDS = 25


def make_markdown_generator(threshold: float = 0.5) -> DefaultMarkdownGenerator:
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=threshold, threshold_type="dynamic"
        ),
    )


def select_content(markdown_obj) -> tuple[str, str]:
    """Return (text, used) where used is 'fit' or 'raw'. Prefer fit_markdown."""
    fit = (getattr(markdown_obj, "fit_markdown", None) or "").strip()
    if fit:
        return fit, "fit"
    raw = (getattr(markdown_obj, "raw_markdown", None) or "").strip()
    return raw, "raw"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def looks_empty(text: str) -> bool:
    """True if the page has no usable content (empty or below the word floor)."""
    if not text or not text.strip():
        return True
    return word_count(text) < MIN_CONTENT_WORDS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_extract.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/extract.py services/crawler/tests/test_extract.py
git commit -m "feat(crawler): generic fit_markdown extraction + empty detection"
```

### Task 3.2: Render configs (fast tier + JS tier) and bucket routing

**Files:**
- Create: `services/crawler/src/crawler/render.py`
- Test: `services/crawler/tests/test_render.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_render.py
from crawler.render import fast_run_config, js_run_config, bucket_for_scope
from crawler.scope import Scope


def test_run_configs_set_robots_and_generator():
    fast = fast_run_config(threshold=0.5)
    js = js_run_config(threshold=0.5)
    assert fast.check_robots_txt is True
    assert js.check_robots_txt is True
    assert js.scan_full_page is True
    assert js.page_timeout >= fast.page_timeout


def test_bucket_for_scope_routing():
    assert bucket_for_scope(Scope.VT_PAGE, "crawled-pages", "crawled-pages-external") == "crawled-pages"
    assert bucket_for_scope(Scope.DOCUMENT, "crawled-pages", "crawled-pages-external") == "crawled-pages"
    assert bucket_for_scope(Scope.EXTERNAL_PAGE, "crawled-pages", "crawled-pages-external") == "crawled-pages-external"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/render.py
"""Two-tier render configs + scope→bucket routing for the crawl engine.

Fast tier: quick render with generic fit_markdown extraction, robots respected.
JS tier:   for pages that came back empty (SPAs, lazy-loaded dining menus) —
           scans the full page, waits for late JS, longer timeout.

Routing:   VT pages and documents -> main bucket (chunker input);
           external HTML -> separate review bucket (never auto-chunked).
"""
from __future__ import annotations

from crawl4ai import CrawlerRunConfig

from crawler.extract import make_markdown_generator
from crawler.scope import Scope


def fast_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        page_timeout=30000,
        delay_before_return_html=0.3,
        stream=True,
        verbose=False,
    )


def js_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        scan_full_page=True,
        scroll_delay=0.5,
        wait_for_images=True,
        delay_before_return_html=1.0,
        page_timeout=90000,
        stream=True,
        verbose=False,
    )


def bucket_for_scope(scope: Scope, main_bucket: str, external_bucket: str) -> str:
    if scope is Scope.EXTERNAL_PAGE:
        return external_bucket
    return main_bucket
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_render.py -v`
Expected: PASS

> Note: if a `CrawlerRunConfig` kwarg here is rejected by the installed Crawl4AI 0.8 build (the research flagged a couple of prose-only defaults), pin exact names against `uv run python -c "from crawl4ai import CrawlerRunConfig; help(CrawlerRunConfig)"` and adjust. The test above will catch attribute mismatches.

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/render.py services/crawler/tests/test_render.py
git commit -m "feat(crawler): two-tier render configs + scope-to-bucket routing"
```

### Task 3.3: The crawl loop (frontier-driven, streamed, checkpointed)

**Files:**
- Create: `services/crawler/src/crawler/engine.py`
- Test: `services/crawler/tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_engine.py
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from crawler.engine import process_result
from crawler.frontier import Frontier, FrontierItem
from crawler.scope import Scope


def _fake_result(url, fit="# Title\n" + "word " * 40, success=True):
    md = SimpleNamespace(fit_markdown=fit, raw_markdown=fit + "\n[nav]")
    return SimpleNamespace(
        url=url, success=success, markdown=md,
        metadata={"title": "T"}, links={"internal": [], "external": []},
        status_code=200, response_headers={"content-type": "text/html"},
        error_message=None,
    )


def test_process_result_stores_vt_page_and_enqueues_children(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    storage = MagicMock()
    item = FrontierItem(
        url="https://cs.vt.edu/people", scope="vt_page", depth=0,
        hop_history=["https://cs.vt.edu/people"], discovery_source="seed",
    )
    result = _fake_result("https://cs.vt.edu/people")
    result.links = {
        "internal": [{"href": "https://cs.vt.edu/people/jane.html"}],
        "external": [{"href": "https://linkedin.com/in/jane"}],
    }
    new_docs = process_result(
        result, item, fr, storage,
        main_bucket="crawled-pages", external_bucket="crawled-pages-external",
        external_hops=1, max_depth=4,
    )
    # stored to main bucket
    assert storage.upload_document.called
    args, kwargs = storage.upload_document.call_args
    assert kwargs.get("bucket", args[2] if len(args) > 2 else None) in (None, "crawled-pages")
    # frontier marked done
    assert fr.count_by_status()["done"] == 1
    # internal child + external child enqueued (external within 1 hop)
    counts = fr.count_by_status()
    assert counts["pending"] == 2
    # documents collected from links returned for later processing
    assert isinstance(new_docs, set)


def test_process_result_empty_returns_needs_js(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    storage = MagicMock()
    item = FrontierItem(
        url="https://dining.vt.edu/menus", scope="vt_page", depth=0,
        hop_history=["https://dining.vt.edu/menus"], discovery_source="seed",
    )
    result = _fake_result("https://dining.vt.edu/menus", fit="loading...")
    needs_js = process_result(
        result, item, fr, storage,
        main_bucket="crawled-pages", external_bucket="crawled-pages-external",
        external_hops=1, max_depth=4, tier="fast",
    )
    # fast tier + empty -> signalled for JS retry; nothing stored, not marked done
    assert needs_js == "needs_js"
    assert fr.count_by_status().get("done", 0) == 0
```

> Note: `process_result` has two return modes used by the test — when content is usable it returns the set of document URLs discovered; when content looks empty on the fast tier it returns the sentinel string `"needs_js"`. Keep that contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/engine.py
"""Frontier-driven crawl engine: lease URLs, render with arun_many, store, expand.

Each streamed result is checkpointed immediately (mark_done / mark_failed) so a
restart resumes from the frontier. Children are classified and enqueued with the
full hop history. Empty fast-tier pages are signalled for a JS-tier retry.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from crawl4ai import AsyncWebCrawler, BrowserConfig, MemoryAdaptiveDispatcher, RateLimiter

from crawler.documents import collect_document_links
from crawler.extract import looks_empty, select_content, word_count
from crawler.frontier import Frontier, FrontierItem
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.metadata import PageMetadata, doc_id_for_url
from crawler.parser_hint import infer_page_type  # see Step 3b
from crawler.render import bucket_for_scope, fast_run_config, js_run_config
from crawler.scope import Scope, classify_url, registered_domain, subdomain_of

logger = logging.getLogger(__name__)


def process_result(
    result,
    item: FrontierItem,
    frontier: Frontier,
    storage,
    *,
    main_bucket: str,
    external_bucket: str,
    external_hops: int,
    max_depth: int,
    tier: str = "fast",
):
    """Store one crawl result and expand its links. Returns either a set of
    discovered document URLs, or the sentinel 'needs_js' for fast-tier empties."""
    if not result.success:
        frontier.mark_failed(item.url, error=getattr(result, "error_message", "unknown"))
        return set()

    text, used = select_content(result.markdown)
    if looks_empty(text):
        if tier == "fast":
            return "needs_js"  # retry in JS tier; leave row leased for now
        frontier.mark_skipped(item.url, reason="empty after JS render")
        return set()

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    if frontier.content_hash_seen(content_hash):
        frontier.mark_done(item.url, content_hash=content_hash, http_status=200)
        return collect_document_links(result.links)

    scope = Scope(item.scope)
    now = datetime.now(timezone.utc)
    title = (result.metadata or {}).get("title")
    page_type = infer_page_type(item.url)
    host = registered_domain(item.url) and (
        __import__("urllib.parse", fromlist=["urlparse"]).urlparse(item.url).hostname
    )
    raw = (getattr(result.markdown, "raw_markdown", "") or "")

    document = build_markdown_document(
        url=item.url, title=title, markdown_content=text,
        crawl_depth=item.depth, crawl_timestamp=now,
        doc_id=doc_id_for_url(item.url), content_hash=content_hash,
        source_host=host, subdomain=subdomain_of(item.url),
        page_type=page_type, is_external=(scope is Scope.EXTERNAL_PAGE),
        last_modified=(result.response_headers or {}).get("last-modified"),
    )
    bucket = bucket_for_scope(scope, main_bucket, external_bucket)
    object_key = url_to_object_key(item.url)
    storage.upload_document(object_key, document, bucket=bucket)

    meta = PageMetadata(
        doc_id=doc_id_for_url(item.url), url=item.url, title=title,
        crawl_depth=item.depth, crawl_timestamp=now, content_hash=content_hash,
        markdown_size_bytes=len(text.encode()),
        status_code=getattr(result, "status_code", None),
        response_headers=result.response_headers or None,
        internal_links=[l["href"] for l in (result.links or {}).get("internal", []) if l.get("href")],
        external_links=[l["href"] for l in (result.links or {}).get("external", []) if l.get("href")],
        last_modified=(result.response_headers or {}).get("last-modified"),
        etag=(result.response_headers or {}).get("etag"),
        source_host=host, subdomain=subdomain_of(item.url), scope=scope.value,
        is_external=(scope is Scope.EXTERNAL_PAGE), hop_history=item.hop_history,
        discovery_source=item.discovery_source,
        content_type=(result.response_headers or {}).get("content-type"),
        raw_markdown_size_bytes=len(raw.encode()),
        fit_markdown_size_bytes=len(text.encode()),
        word_count=word_count(text), render_tier=tier, robots_allowed=True,
        fetched_at=now,
    )
    storage.upload_metadata(object_key, meta, bucket=bucket)
    frontier.mark_done(item.url, content_hash=content_hash,
                       http_status=getattr(result, "status_code", 200) or 200)

    # Expand: enqueue children (only from VT pages, to bound external fan-out).
    if scope is Scope.VT_PAGE and item.depth < max_depth:
        links = (result.links or {})
        for cat in ("internal", "external"):
            for link in links.get(cat, []):
                href = link.get("href")
                if not href:
                    continue
                child_scope, child_hops = classify_url(
                    href, parent_hops=item.hop_history, external_hops=external_hops,
                )
                if child_scope is Scope.OUT_OF_SCOPE:
                    continue
                frontier.enqueue(
                    href, scope=child_scope.value, depth=item.depth + 1,
                    hop_history=child_hops, discovery_source="bfs",
                )
    return collect_document_links(result.links)


def browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True, text_mode=True, light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )


def make_dispatcher(memory_pct: float, max_sessions: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=memory_pct,
        check_interval=1.0,
        max_session_permit=max_sessions,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 3.0), max_delay=60.0, max_retries=3,
            rate_limit_codes=[429, 503],
        ),
    )
```

- [ ] **Step 3b: Add the page-type helper the engine imports**

The chunker already has `infer_page_type`, but the crawler must not import across services. Create a tiny shared copy:

```python
# services/crawler/src/crawler/parser_hint.py
"""Coarse page-type inference from a URL path (mirrors chunker.parser)."""
from __future__ import annotations


def infer_page_type(url: str) -> str:
    path = url.lower()
    if any(k in path for k in ["/people/", "/faculty/", "faculty"]):
        return "faculty"
    if any(k in path for k in ["/courses/", "/classes/", "/course"]):
        return "course"
    if any(k in path for k in ["/research/", "/labs/"]):
        return "research"
    if any(k in path for k in ["/news/", "/events/", "/calendar"]):
        return "news"
    if any(k in path for k in ["/dining", "/menus", "/menu"]):
        return "dining"
    if any(k in path for k in ["/about", "/info", "/contact"]):
        return "about"
    return "general"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_engine.py tests/test_render.py -v`
Expected: PASS. (If `select_content`/links shapes need tweaks to satisfy the asserts, adjust the impl — not the test contract.)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/engine.py services/crawler/src/crawler/parser_hint.py services/crawler/tests/test_engine.py
git commit -m "feat(crawler): frontier-driven crawl engine with checkpointing + JS tier"
```

### Task 3.4: Drive loop — lease batches, run fast tier, requeue empties to JS tier

**Files:**
- Modify: `services/crawler/src/crawler/engine.py` (add `run_engine`)
- Test: `services/crawler/tests/test_engine_loop.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_engine_loop.py
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from crawler.engine import run_engine
from crawler.frontier import Frontier


def _result(url, fit):
    md = SimpleNamespace(fit_markdown=fit, raw_markdown=fit)
    return SimpleNamespace(url=url, success=True, markdown=md,
                           metadata={"title": "T"},
                           links={"internal": [], "external": []},
                           status_code=200, response_headers={}, error_message=None)


def test_run_engine_processes_until_frontier_drains(tmp_path):
    fr = Frontier(str(tmp_path / "f.db"))
    fr.enqueue("https://cs.vt.edu/a", scope="vt_page", depth=0,
               hop_history=["https://cs.vt.edu/a"], discovery_source="seed")
    storage = MagicMock()

    async def fake_arun_many(urls, config, dispatcher):
        async def gen():
            for u in urls:
                yield _result(u, "# A\n" + "word " * 40)
        return gen()

    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler.arun_many = AsyncMock(side_effect=fake_arun_many)

    with patch("crawler.engine.AsyncWebCrawler", return_value=crawler):
        stats = asyncio.run(run_engine(
            fr, storage, main_bucket="crawled-pages",
            external_bucket="crawled-pages-external",
            external_hops=1, max_depth=2, batch_size=10,
            memory_pct=70.0, max_sessions=5, threshold=0.5,
        ))
    assert stats["pages_stored"] >= 1
    assert fr.count_by_status().get("pending", 0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_engine_loop.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_engine'`

- [ ] **Step 3: Write minimal implementation**

Append to `services/crawler/src/crawler/engine.py`:

```python
async def run_engine(
    frontier: Frontier,
    storage,
    *,
    main_bucket: str,
    external_bucket: str,
    external_hops: int,
    max_depth: int,
    batch_size: int,
    memory_pct: float,
    max_sessions: int,
    threshold: float,
) -> dict:
    """Lease pending URLs in batches, render (fast tier), checkpoint, retry empties
    on the JS tier, until the frontier has no pending rows."""
    stats = {"pages_stored": 0, "pages_failed": 0, "needs_js": 0, "documents": 0}
    document_urls: set[str] = set()
    dispatcher = make_dispatcher(memory_pct, max_sessions)
    fast_cfg = fast_run_config(threshold)
    js_cfg = js_run_config(threshold)

    async with AsyncWebCrawler(config=browser_config()) as crawler:
        while True:
            batch = frontier.lease_pending(limit=batch_size)
            if not batch:
                break
            by_url = {it.url: it for it in batch}

            # --- fast tier ---
            js_retry: list = []
            results = await crawler.arun_many(
                urls=list(by_url.keys()), config=fast_cfg, dispatcher=dispatcher,
            )
            async for result in results:
                item = by_url.get(result.url)
                if item is None:
                    continue
                outcome = process_result(
                    result, item, frontier, storage,
                    main_bucket=main_bucket, external_bucket=external_bucket,
                    external_hops=external_hops, max_depth=max_depth, tier="fast",
                )
                if outcome == "needs_js":
                    js_retry.append(item)
                    stats["needs_js"] += 1
                else:
                    document_urls |= outcome
                    if frontier.count_by_status().get("done"):
                        stats["pages_stored"] += 1

            # --- JS tier for empties ---
            if js_retry:
                js_by_url = {it.url: it for it in js_retry}
                js_results = await crawler.arun_many(
                    urls=list(js_by_url.keys()), config=js_cfg, dispatcher=dispatcher,
                )
                async for result in js_results:
                    item = js_by_url.get(result.url)
                    if item is None:
                        continue
                    outcome = process_result(
                        result, item, frontier, storage,
                        main_bucket=main_bucket, external_bucket=external_bucket,
                        external_hops=external_hops, max_depth=max_depth, tier="js",
                    )
                    if outcome != "needs_js":
                        document_urls |= outcome

    counts = frontier.count_by_status()
    stats["pages_stored"] = counts.get("done", 0)
    stats["pages_failed"] = counts.get("failed", 0)
    stats["documents"] = len(document_urls)
    stats["document_urls"] = document_urls
    return stats
```

> The `pages_stored` counter is reconciled from the frontier's `done` count at the end (authoritative), so the inner increment is best-effort only.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_engine_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/engine.py services/crawler/tests/test_engine_loop.py
git commit -m "feat(crawler): batched fast+JS render drive loop over the frontier"
```

---

## Phase 4 — Wire new crawler into main.py, config, and run bounded validation

### Task 4.1: Extend config

**Files:**
- Modify: `services/crawler/src/crawler/config.py`
- Test: `services/crawler/tests/test_config.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_config.py  (append)
def test_config_vtedu_defaults(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "k")
    monkeypatch.setenv("MINIO_SECRET_KEY", "s")
    from crawler.config import CrawlerConfig
    cfg = CrawlerConfig.from_env()
    assert cfg.allowed_domains == ("vt.edu",)
    assert cfg.external_hops == 1
    assert cfg.minio_external_bucket == "crawled-pages-external"
    assert cfg.frontier_db_path  # has a default
    assert cfg.max_urls_per_host == -1
    assert cfg.batch_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_config.py::test_config_vtedu_defaults -v`
Expected: FAIL — `AttributeError`/assertion on `allowed_domains`.

- [ ] **Step 3: Write minimal implementation**

In `services/crawler/src/crawler/config.py`:
- Change the `allowed_domains` default from `"cs.vt.edu"` to `"vt.edu"` (both in the dataclass docstring and the `_domains("CRAWL_ALLOWED_DOMAINS", "vt.edu")` call).
- Remove `cs.vt.edu`-specific `blocked_domains` defaults (they reference cs subdomains we now want); default to empty unless `CRAWL_BLOCKED_DOMAINS` set.
- Add new fields to the dataclass and `from_env`:

```python
    # appended dataclass fields
    minio_external_bucket: str = "crawled-pages-external"
    external_hops: int = 1
    frontier_db_path: str = "/data/frontier.db"
    discovery_source: str = "sitemap+cc"
    max_hosts: int | None = None
    max_urls_per_host: int = -1
    batch_size: int = 200
    memory_threshold_percent: float = 70.0
    max_sessions: int = 10
    prune_content_threshold: float = 0.5
```

```python
    # appended to the cls(...) call in from_env
            minio_external_bucket=os.environ.get("MINIO_EXTERNAL_BUCKET", "crawled-pages-external"),
            external_hops=int(os.environ.get("CRAWL_EXTERNAL_HOPS", "1")),
            frontier_db_path=os.environ.get("FRONTIER_DB_PATH", "/data/frontier.db"),
            discovery_source=os.environ.get("CRAWL_DISCOVERY_SOURCE", "sitemap+cc"),
            max_hosts=(int(os.environ["CRAWL_MAX_HOSTS"]) if os.environ.get("CRAWL_MAX_HOSTS") else None),
            max_urls_per_host=int(os.environ.get("CRAWL_MAX_URLS_PER_HOST", "-1")),
            batch_size=int(os.environ.get("CRAWL_BATCH_SIZE", "200")),
            memory_threshold_percent=float(os.environ.get("CRAWL_MEMORY_PCT", "70.0")),
            max_sessions=int(os.environ.get("CRAWL_MAX_SESSIONS", "10")),
            prune_content_threshold=float(os.environ.get("CRAWL_PRUNE_CONTENT_THRESHOLD", "0.5")),
```

Update the existing `test_config_from_env_defaults` assertions that expect `cs.vt.edu` / the old blocked list to expect `("vt.edu",)` and the new defaults.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/config.py services/crawler/tests/test_config.py
git commit -m "feat(crawler): vt.edu-wide config with discovery/render/external settings"
```

### Task 4.2: New main orchestration

**Files:**
- Modify: `services/crawler/src/crawler/main.py`
- Modify: `services/crawler/src/crawler/storage.py` (add `minio_external_bucket` awareness — it already takes bucket per call, so only ensure buckets)
- Test: `services/crawler/tests/test_main.py` (replace/extend with an orchestration smoke test using mocks)

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_main.py  (add)
from unittest.mock import AsyncMock, MagicMock, patch

from crawler import main as cmain


def test_cli_runs_discover_then_engine_then_docs(monkeypatch, tmp_path):
    cfg = MagicMock()
    cfg.frontier_db_path = str(tmp_path / "f.db")
    cfg.minio_bucket = "crawled-pages"
    cfg.minio_cleaned_bucket = "crawled-pages-cleaned"
    cfg.minio_external_bucket = "crawled-pages-external"
    cfg.external_hops = 1
    cfg.max_depth = 2
    cfg.batch_size = 50
    cfg.memory_threshold_percent = 70.0
    cfg.max_sessions = 5
    cfg.prune_content_threshold = 0.5
    cfg.max_hosts = 5
    cfg.max_urls_per_host = 50
    cfg.request_delay = 0.1

    monkeypatch.setattr(cmain.CrawlerConfig, "from_env", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cmain, "MinioStorage", MagicMock())
    monkeypatch.setattr(cmain, "populate_frontier", AsyncMock(return_value={"urls_enqueued": 3}))
    monkeypatch.setattr(cmain, "run_engine", AsyncMock(return_value={
        "pages_stored": 3, "pages_failed": 0, "documents": 0, "document_urls": set(),
    }))
    monkeypatch.setattr(cmain, "download_and_process_documents",
                        AsyncMock(return_value={"documents_processed": 0, "documents_failed": 0}))

    cmain.cli()  # should not raise
    assert cmain.populate_frontier.called
    assert cmain.run_engine.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_main.py -v`
Expected: FAIL (current `cli` uses the old BFS path / different symbols).

- [ ] **Step 3: Write minimal implementation**

Rewrite `services/crawler/src/crawler/main.py` to the discovery→engine→documents flow. Key points: ensure the three buckets exist; run discovery to populate the frontier (skipped if frontier already has pending rows — resumability); run the engine; process documents discovered during the crawl into the **cleaned** bucket (so the chunker sees them) AND store their extracted markdown; write a run summary to the `logs` bucket. Drop the old per-page cs.vt.edu cleaning loop (fit_markdown already cleaned VT pages into the main bucket; mirror VT pages into the cleaned bucket since the chunker reads `crawled-pages-cleaned`).

```python
"""Crawler entrypoint — discover → frontier → render → documents → summary."""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from crawler.config import CrawlerConfig
from crawler.discovery.run import populate_frontier
from crawler.documents import download_and_process_documents
from crawler.engine import run_engine
from crawler.frontier import Frontier
from crawler.storage import MinioStorage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
LOGS_BUCKET = "logs"


def cli() -> None:
    logger.info("Starting HokieHelp crawler (vt.edu full-domain)")
    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = MinioStorage(config)
    for bucket in (config.minio_bucket, config.minio_cleaned_bucket,
                   config.minio_external_bucket, LOGS_BUCKET):
        storage.ensure_bucket(bucket)

    # 1) Discovery — populate frontier unless resuming an existing one with work left.
    frontier = Frontier(config.frontier_db_path)
    if frontier.count_by_status().get("pending", 0) == 0 and \
       frontier.count_by_status().get("done", 0) == 0:
        logger.info("Empty frontier — running discovery")
        asyncio.run(populate_frontier(
            config.frontier_db_path, max_hosts=config.max_hosts,
            max_urls_per_host=config.max_urls_per_host,
            external_hops=config.external_hops,
        ))
        frontier = Frontier(config.frontier_db_path)
    else:
        logger.info("Resuming existing frontier: %s", frontier.count_by_status())

    # 2) Render engine — VT pages → main bucket; external → external bucket.
    stats = asyncio.run(run_engine(
        frontier, storage, main_bucket=config.minio_bucket,
        external_bucket=config.minio_external_bucket,
        external_hops=config.external_hops, max_depth=config.max_depth,
        batch_size=config.batch_size, memory_pct=config.memory_threshold_percent,
        max_sessions=config.max_sessions, threshold=config.prune_content_threshold,
    ))
    logger.info("Render complete: %s",
                {k: v for k, v in stats.items() if k != "document_urls"})

    # 3) Mirror VT main-bucket pages into the cleaned bucket for the chunker.
    #    (fit_markdown already cleaned them; the chunker reads crawled-pages-cleaned.)
    mirrored = 0
    for key in storage.list_objects(config.minio_bucket):
        if not key.endswith(".md"):
            continue
        storage.upload_document(key, storage.download_document(key, bucket=config.minio_bucket),
                                bucket=config.minio_cleaned_bucket)
        try:
            storage.upload_metadata(
                key, storage.download_metadata(key, bucket=config.minio_bucket),
                bucket=config.minio_cleaned_bucket,
            )
        except Exception:
            pass
        mirrored += 1
    logger.info("Mirrored %d VT pages into %s", mirrored, config.minio_cleaned_bucket)

    # 4) Documents (PDF/DOCX from any host) → cleaned bucket (chunker input).
    doc_urls = stats.get("document_urls", set())
    if doc_urls:
        seen = set(storage.load_all_content_hashes(config.minio_cleaned_bucket).keys())
        doc_stats = asyncio.run(download_and_process_documents(
            doc_urls, storage, seen, set(), request_delay=config.request_delay,
        ))
        logger.info("Documents: %s", doc_stats)

    # 5) Run summary to logs bucket.
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    summary = (f"# Crawl run {ts}\n\n"
               f"Frontier: {frontier.count_by_status()}\n"
               f"Render: {stats.get('pages_stored')} stored, "
               f"{stats.get('pages_failed')} failed, {stats.get('needs_js')} JS-retried\n")
    storage.upload_document(f"crawl-summary-{ts}.md", summary, bucket=LOGS_BUCKET)
    frontier.close()


if __name__ == "__main__":
    cli()
```

> `download_and_process_documents` currently always writes to the default (main) bucket. For documents to reach the chunker, pass the cleaned bucket. Update its `storage.upload_document(object_key, document)` / `upload_metadata` calls to `bucket=storage._cleaned_or_default` — simplest: add a `target_bucket` parameter to `download_and_process_documents` and thread `config.minio_cleaned_bucket` through. Make that change here and update its unit test in `tests/test_documents.py` to pass/assert the bucket.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_main.py tests/test_documents.py -v`
Expected: PASS (fix the documents test for the new `target_bucket` param).

- [ ] **Step 5: Run the full crawler suite**

Run: `cd services/crawler && uv run pytest -v`
Expected: all green. Remove now-dead `cleaner.py` tests only if you also delete the module — defer deletion to Task 4.4.

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/main.py services/crawler/src/crawler/documents.py services/crawler/tests/test_main.py services/crawler/tests/test_documents.py
git commit -m "feat(crawler): discovery→frontier→render→documents orchestration"
```

### Task 4.3: Bounded validation run on the MacBook

**Files:** none (operational). Requires local MinIO (the project already runs MinIO in-cluster; for the laptop run use a local MinIO container or port-forward to the cluster MinIO in the `test` namespace).

- [ ] **Step 1: Start a local MinIO (or port-forward cluster MinIO)**

Show context first (project rule): `kubectl config current-context`
Then: `kubectl port-forward -n test svc/minio 9000:9000 &`
Set env: `export MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=<from minio-secret> MINIO_SECRET_KEY=<from minio-secret> MINIO_SECURE=false`

- [ ] **Step 2: Run a bounded discovery + crawl**

```bash
cd services/crawler
export FRONTIER_DB_PATH=/tmp/vt_validation.db
export CRAWL_MAX_HOSTS=10 CRAWL_MAX_URLS_PER_HOST=150 CRAWL_MAX_DEPTH=2 CRAWL_EXTERNAL_HOPS=1
uv run python -m crawler.main
```
Expected: logs show discovery stats, render batches progressing, "Mirrored N VT pages", a documents pass, and a final summary. Wall-clock should be minutes, not hours, for ~1–2k pages.

- [ ] **Step 3: Eyeball quality**

```bash
sqlite3 /tmp/vt_validation.db "SELECT status, COUNT(*) FROM frontier GROUP BY status;"
# Inspect a sample cleaned page in MinIO:
mc alias set localmin http://localhost:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
mc ls --recursive localmin/crawled-pages-cleaned | head
mc cat localmin/crawled-pages-cleaned/dining.vt.edu/index.md | head -60
```
Expected: frontmatter has `source_host`, `subdomain`, `page_type`; body is real content (no global VT nav/footer); `crawled-pages-external/` holds only non-VT hosts; documents present as `.md`.

- [ ] **Step 4: Decision gate**

If speed and quality look good → proceed to Phase 5/6. If a major site renders empty even on the JS tier, note it in a deferral list (`services/crawler/DEFERRED_SITES.md`) per the agreed "figure out later" policy and continue.

- [ ] **Step 5: Commit any threshold/config tweaks discovered**

```bash
git add -A && git commit -m "chore(crawler): tune thresholds after bounded validation run"
```

### Task 4.4: Retire the cs.vt.edu hand-tuned cleaner

**Files:**
- Delete: `services/crawler/src/crawler/cleaner.py`
- Delete: `services/crawler/tests/test_cleaner.py`
- Keep: `build_department_info_doc` content — move it into a small `services/crawler/src/crawler/seed_docs.py` so the `_department-info.md` reference doc is still written (call it from `main.py` step 3, uploading to the cleaned bucket).

- [ ] **Step 1: Move `build_department_info_doc` to `seed_docs.py`** (copy the function verbatim from `cleaner.py`).
- [ ] **Step 2: Call it in `main.py`** after the mirror step: `storage.upload_document("_department-info.md", build_department_info_doc(), bucket=config.minio_cleaned_bucket)`.
- [ ] **Step 3: Delete `cleaner.py` and `test_cleaner.py`.**
- [ ] **Step 4: Run suite:** `cd services/crawler && uv run pytest -v` → all green.
- [ ] **Step 5: Commit**

```bash
git rm services/crawler/src/crawler/cleaner.py services/crawler/tests/test_cleaner.py
git add services/crawler/src/crawler/seed_docs.py services/crawler/src/crawler/main.py
git commit -m "refactor(crawler): retire cs.vt.edu regex cleaner; fit_markdown is generic"
```

---

## Phase 5 — Propagate richer metadata downstream (chunker → embedder → Qdrant)

So retrieval can filter by subdomain / page_type and rank with better signal.

### Task 5.1: Chunker reads new frontmatter

**Files:**
- Modify: `services/chunker/src/chunker/parser.py`
- Test: `services/chunker/tests/test_parser.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# services/chunker/tests/test_parser.py  (append)
from chunker.parser import parse_frontmatter


def test_parse_frontmatter_reads_host_and_page_type():
    md = (
        "---\n"
        "doc_id: 'abc'\n"
        "url: 'https://dining.vt.edu/menus'\n"
        "title: 'Menus'\n"
        "content_hash: 'h'\n"
        "crawl_timestamp: '2026-06-10T00:00:00+00:00'\n"
        "source_host: 'dining.vt.edu'\n"
        "subdomain: 'dining'\n"
        "page_type: 'dining'\n"
        "last_modified: 'Wed, 01 Jan 2026 00:00:00 GMT'\n"
        "---\n\n# Menus\nbody"
    )
    fm, body = parse_frontmatter(md)
    assert fm.source_host == "dining.vt.edu"
    assert fm.subdomain == "dining"
    assert fm.page_type == "dining"
    assert fm.last_modified.startswith("Wed")
    assert body.strip().startswith("# Menus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/chunker && uv run pytest tests/test_parser.py -k frontmatter -v`
Expected: FAIL — `AttributeError: 'FrontmatterData' object has no attribute 'source_host'`

- [ ] **Step 3: Write minimal implementation**

In `services/chunker/src/chunker/parser.py`, extend `FrontmatterData` and `parse_frontmatter`:

```python
@dataclass
class FrontmatterData:
    doc_id: str
    url: str
    title: str
    content_hash: str
    crawl_timestamp: str
    source_host: str = ""
    subdomain: str = ""
    page_type: str = "general"
    last_modified: str = ""
```

At the end of `parse_frontmatter`, populate the new fields (fall back to URL-derived values so old docs still work):

```python
    from urllib.parse import urlparse
    source_host = fields.get("source_host") or (urlparse(url).hostname or "")
    subdomain = fields.get("subdomain", "")
    page_type = fields.get("page_type") or infer_page_type(url)
    last_modified = fields.get("last_modified", "")

    return FrontmatterData(
        doc_id=doc_id, url=url, title=title, content_hash=content_hash,
        crawl_timestamp=crawl_timestamp, source_host=source_host,
        subdomain=subdomain, page_type=page_type, last_modified=last_modified,
    ), body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/chunker && uv run pytest tests/test_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/parser.py services/chunker/tests/test_parser.py
git commit -m "feat(chunker): parse host/subdomain/page_type from richer frontmatter"
```

### Task 5.2: ChunkRecord carries host + subdomain

**Files:**
- Modify: `services/chunker/src/chunker/models.py`
- Modify: `services/chunker/src/chunker/splitter.py`
- Test: `services/chunker/tests/test_models.py` and `tests/test_splitter.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# services/chunker/tests/test_models.py  (append)
from chunker.models import ChunkRecord


def test_chunkrecord_roundtrip_with_host_fields():
    rec = ChunkRecord(
        chunk_id="d_0000", document_id="d", chunk_index=0, text="t",
        url="https://dining.vt.edu/x", title="T", page_type="dining",
        headings_path=["A"], content_hash="h", crawl_timestamp="ts",
        token_count=1, source_host="dining.vt.edu", subdomain="dining",
    )
    back = ChunkRecord.from_dict(rec.to_dict())
    assert back.source_host == "dining.vt.edu"
    assert back.subdomain == "dining"


def test_chunkrecord_from_dict_defaults_missing_host():
    d = {
        "chunk_id": "d_0000", "document_id": "d", "chunk_index": 0, "text": "t",
        "url": "u", "title": "T", "page_type": "general", "headings_path": [],
        "content_hash": "h", "crawl_timestamp": "ts", "token_count": 1,
    }
    rec = ChunkRecord.from_dict(d)
    assert rec.source_host == ""
    assert rec.subdomain == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/chunker && uv run pytest tests/test_models.py -v`
Expected: FAIL — `TypeError`/`KeyError` on `source_host`.

- [ ] **Step 3: Write minimal implementation**

In `models.py`, add fields + serialize/deserialize:

```python
    source_host: str = ""
    subdomain: str = ""
```
Add to `to_dict`:
```python
            "source_host": self.source_host,
            "subdomain": self.subdomain,
```
Add to `from_dict`:
```python
            source_host=data.get("source_host", ""),
            subdomain=data.get("subdomain", ""),
```

In `splitter.py` `build_chunks(...)`, thread `frontmatter.source_host` and `frontmatter.subdomain` into every `ChunkRecord(...)` constructed (alongside the existing `page_type=frontmatter.page_type` style fields). Find the `ChunkRecord(` call and add `source_host=frontmatter.source_host, subdomain=frontmatter.subdomain,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/chunker && uv run pytest -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/models.py services/chunker/src/chunker/splitter.py services/chunker/tests/test_models.py
git commit -m "feat(chunker): carry source_host/subdomain into chunk records"
```

### Task 5.3: Embedder payload + Qdrant indexes

**Files:**
- Modify: `services/embedder/src/embedder/indexer.py`
- Test: `services/embedder/tests/test_indexer.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

```python
# services/embedder/tests/test_indexer.py  (append/create)
from unittest.mock import MagicMock
from embedder.indexer import QdrantIndexer


def test_upsert_includes_host_fields(monkeypatch):
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [MagicMock(name="hokiehelp_chunks")]
    # force collection-exists path
    fake_client.get_collections.return_value.collections[0].name = "hokiehelp_chunks"
    monkeypatch.setattr("embedder.indexer.QdrantClient", lambda host, port: fake_client)

    idx = QdrantIndexer(host="h", port=1, collection="hokiehelp_chunks", vector_size=3)
    chunks = [{
        "chunk_id": "d_0000", "document_id": "d", "url": "https://dining.vt.edu/x",
        "title": "T", "page_type": "dining", "headings_path": [], "chunk_index": 0,
        "content_hash": "h", "crawl_timestamp": "ts", "token_count": 1, "text": "t",
        "source_host": "dining.vt.edu", "subdomain": "dining",
    }]
    idx.upsert_chunks(chunks, [[0.1, 0.2, 0.3]])
    points = fake_client.upsert.call_args.kwargs["points"]
    assert points[0].payload["source_host"] == "dining.vt.edu"
    assert points[0].payload["subdomain"] == "dining"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/embedder && uv run pytest tests/test_indexer.py -v`
Expected: FAIL — `KeyError: 'source_host'` (payload missing it).

- [ ] **Step 3: Write minimal implementation**

In `indexer.py` `upsert_chunks`, add to the payload dict:

```python
                    "source_host": chunk.get("source_host", ""),
                    "subdomain": chunk.get("subdomain", ""),
```

In `_ensure_collection`, after the existing `document_id` index, add KEYWORD indexes (only runs at collection creation; for an existing collection, add a one-off migration note):

```python
        for field_name in ("source_host", "subdomain", "page_type"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/embedder && uv run pytest tests/test_indexer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/embedder/src/embedder/indexer.py services/embedder/tests/test_indexer.py
git commit -m "feat(embedder): index source_host/subdomain/page_type in Qdrant payload"
```

---

## Phase 6 — Verification & QA tooling

Quantify completeness (did we get everything?) and content quality (are the MD files clean and non-empty?), and account for failures.

### Task 6.1: Verification logic

**Files:**
- Create: `services/crawler/src/crawler/verify.py`
- Test: `services/crawler/tests/test_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# services/crawler/tests/test_verify.py
from crawler.verify import assess_document, summarize


def test_assess_document_flags_empty_and_boilerplate():
    good = "---\nurl: 'https://x.vt.edu/a'\nsource_host: 'x.vt.edu'\n---\n\n# Real\n" + "word " * 60
    a = assess_document("x.vt.edu/a.md", good)
    assert a.ok is True
    assert a.word_count >= 50

    empty = "---\nurl: 'https://x.vt.edu/b'\n---\n\n   "
    b = assess_document("x.vt.edu/b.md", empty)
    assert b.ok is False
    assert "empty" in b.reasons[0]

    no_fm = "# Orphan\nbody words " * 5
    c = assess_document("x.vt.edu/c.md", no_fm)
    assert c.ok is False
    assert any("frontmatter" in r for r in c.reasons)


def test_summarize_counts():
    from crawler.verify import DocAssessment
    rows = [
        DocAssessment("a", True, 100, []),
        DocAssessment("b", False, 0, ["empty body"]),
        DocAssessment("c", False, 5, ["too short"]),
    ]
    s = summarize(rows)
    assert s["total"] == 3
    assert s["ok"] == 1
    assert s["failed"] == 2
    assert "empty body" in s["reason_histogram"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && uv run pytest tests/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crawler.verify'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/crawler/src/crawler/verify.py
"""Post-ingestion verification: completeness + content quality + failure accounting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_WORDS = 25


@dataclass
class DocAssessment:
    key: str
    ok: bool
    word_count: int
    reasons: list[str] = field(default_factory=list)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def assess_document(key: str, content: str) -> DocAssessment:
    reasons: list[str] = []
    if not content.startswith("---"):
        reasons.append("missing frontmatter")
        body = content
    else:
        end = content.find("---", 3)
        body = content[end + 3:] if end != -1 else content
    stripped = body.strip()
    wc = _word_count(stripped)
    if not stripped:
        reasons.append("empty body")
    elif wc < MIN_WORDS:
        reasons.append(f"too short ({wc} words)")
    # crude global-nav leakage check (should have been pruned by fit_markdown)
    if "Skip to main content" in body or "Universal Access" in body:
        reasons.append("boilerplate leaked")
    ok = len(reasons) == 0
    return DocAssessment(key=key, ok=ok, word_count=wc, reasons=reasons)


def summarize(rows: list[DocAssessment]) -> dict:
    histogram: dict[str, int] = {}
    for r in rows:
        for reason in r.reasons:
            key = re.sub(r"\(\d+ words\)", "(N words)", reason)
            histogram[key] = histogram.get(key, 0) + 1
    return {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "failed": sum(1 for r in rows if not r.ok),
        "reason_histogram": histogram,
    }


def render_report(summary: dict, sample_failures: list[DocAssessment]) -> str:
    lines = ["# Ingestion Verification Report", "",
             f"- Total documents: {summary['total']}",
             f"- Clean (ok): {summary['ok']}",
             f"- Failed: {summary['failed']}", "",
             "## Failure reasons", ""]
    for reason, count in sorted(summary["reason_histogram"].items(), key=lambda x: -x[1]):
        lines.append(f"- {reason}: {count}")
    lines += ["", "## Sample failures", ""]
    for a in sample_failures[:25]:
        lines.append(f"- `{a.key}` — {', '.join(a.reasons)}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/crawler && uv run pytest tests/test_verify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/verify.py services/crawler/tests/test_verify.py
git commit -m "feat(crawler): ingestion verification (quality + failure accounting)"
```

### Task 6.2: Verification CLI over MinIO + frontier

**Files:**
- Create: `services/crawler/scripts/verify_ingestion.py`
- Test: covered by `test_verify.py` logic; CLI is a thin shell (smoke-run manually).

- [ ] **Step 1: Write the CLI**

```python
# services/crawler/scripts/verify_ingestion.py
#!/usr/bin/env python3
"""Verify ingested markdown quality + completeness; write a report to the logs bucket.

    uv run python scripts/verify_ingestion.py --bucket crawled-pages-cleaned [--frontier f.db]
"""
import argparse
import logging
from datetime import datetime, timezone

from crawler.config import CrawlerConfig
from crawler.frontier import Frontier
from crawler.storage import MinioStorage
from crawler.verify import assess_document, render_report, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="crawled-pages-cleaned")
    p.add_argument("--frontier", default=None)
    args = p.parse_args()

    config = CrawlerConfig.from_env()
    storage = MinioStorage(config)

    rows = []
    for key in storage.list_objects(args.bucket):
        if not key.endswith(".md"):
            continue
        rows.append(assess_document(key, storage.download_document(key, bucket=args.bucket)))

    summary = summarize(rows)
    failures = [r for r in rows if not r.ok]
    report = render_report(summary, failures)

    if args.frontier:
        fr = Frontier(args.frontier)
        report += f"\n## Frontier status\n\n```\n{fr.count_by_status()}\n```\n"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    storage.ensure_bucket("logs")
    storage.upload_document(f"verification-{ts}.md", report, bucket="logs")
    logger.info("Verification: %s", summary)
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run against the bounded validation buckets**

Run (env from Task 4.3): `cd services/crawler && uv run python scripts/verify_ingestion.py --bucket crawled-pages-cleaned --frontier /tmp/vt_validation.db`
Expected: prints a report with total/ok/failed and a reason histogram; report saved to the `logs` bucket. Target: failed rate is low and dominated by genuinely-empty source pages, not boilerplate leakage. If "boilerplate leaked" is high, raise `prune_content_threshold` (e.g. 0.5 → 0.6) and re-run a bounded crawl.

- [ ] **Step 3: Commit**

```bash
git add services/crawler/scripts/verify_ingestion.py
git commit -m "feat(crawler): verification CLI writing a report to the logs bucket"
```

---

## Phase 7 — Kubernetes: resumable Job, configmaps, full-domain test-pod run, promotion

Operational runbook. Project rules: always show `kubectl config current-context` before applying; default namespace `test`; never prod without explicit instruction; manifests in `k8s/`; verify with get pods/svc/rollout.

### Task 7.1: Frontier PVC + configmap updates

**Files:**
- Create: `k8s/crawler-pvc.yaml`
- Modify: `k8s/crawler-configmap.yaml`
- Modify: `k8s/pipeline-configmap.yaml`

- [ ] **Step 1: Write `k8s/crawler-pvc.yaml`**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: crawler-frontier
  namespace: test
  labels:
    app: hokiehelp-crawler
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 5Gi
```

- [ ] **Step 2: Update `k8s/crawler-configmap.yaml`** — replace the cs.vt.edu values:

```yaml
data:
  MINIO_ENDPOINT: "minio:9000"
  MINIO_BUCKET: "crawled-pages"
  MINIO_CLEANED_BUCKET: "crawled-pages-cleaned"
  MINIO_EXTERNAL_BUCKET: "crawled-pages-external"
  MINIO_SECURE: "false"
  CRAWL_ALLOWED_DOMAINS: "vt.edu"
  CRAWL_EXTERNAL_HOPS: "1"
  CRAWL_MAX_DEPTH: "10"   # backstop for sitemap-less hosts; coverage comes from discovery, see "On crawl depth" above
  CRAWL_DISCOVERY_SOURCE: "sitemap+cc"
  CRAWL_MAX_URLS_PER_HOST: "-1"
  CRAWL_BATCH_SIZE: "200"
  CRAWL_MEMORY_PCT: "70.0"
  CRAWL_MAX_SESSIONS: "10"
  CRAWL_PRUNE_CONTENT_THRESHOLD: "0.5"
  CRAWL_REQUEST_DELAY: "0.5"
  FRONTIER_DB_PATH: "/data/frontier.db"
  CRAWL_BLOCKED_PATHS: "/content/,/editor.html"
```

- [ ] **Step 3: Update `k8s/pipeline-configmap.yaml`** — set `CRAWL_SEED_URL` unused note + matching domain default (leave depth/pages as overrides). Set:

```yaml
  CRAWL_ALLOWED_DOMAINS: "vt.edu"
  CRAWL_MAX_DEPTH: "10"
```

- [ ] **Step 4: Mount the PVC in the crawler Job spec**

Modify `services/pipeline/src/pipeline/jobs.py` `build_crawler_job`: add a volume + mount so `/data` is the PVC, and raise memory limits (full-domain crawl is heavier):

```python
        volume_mounts=[client.V1VolumeMount(name="frontier", mount_path="/data")],
```
on the container, and on the pod spec:
```python
                    volumes=[client.V1Volume(
                        name="frontier",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name="crawler-frontier"),
                    )],
```
Bump `resources.requests/limits` memory to `6Gi`/`12Gi`. Add a unit test in `services/pipeline/tests/test_jobs.py` asserting the volume mount + claim name are present.

- [ ] **Step 5: Apply (show context first) and verify**

```bash
kubectl config current-context
kubectl apply -f k8s/crawler-pvc.yaml -f k8s/crawler-configmap.yaml -f k8s/pipeline-configmap.yaml
kubectl get pvc -n test
```
Expected: `crawler-frontier` PVC `Bound`.

- [ ] **Step 6: Commit**

```bash
git add k8s/crawler-pvc.yaml k8s/crawler-configmap.yaml k8s/pipeline-configmap.yaml services/pipeline/src/pipeline/jobs.py services/pipeline/tests/test_jobs.py
git commit -m "feat(k8s): frontier PVC + vt.edu crawler config + mount in crawler Job"
```

### Task 7.2: Build/push the new crawler image and run the full-domain test pod

**Files:**
- Create: `k8s/crawler-test-job.yaml`

- [ ] **Step 1: Build & load the image** (match how other images get to the cluster — confirm the registry/loader the project uses; for a local cluster this is typically `docker build` + load).

```bash
docker build -t hokiehelp-crawler:vtedu-test services/crawler
# load into the cluster per its mechanism (e.g. kind load / k3d import / push to registry)
```

- [ ] **Step 2: Write `k8s/crawler-test-job.yaml`** — points at scratch buckets so it never touches main data:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: crawler-fulltest
  namespace: test
  labels: { app: hokiehelp-crawler, run: fulltest }
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 86400
  template:
    metadata:
      labels: { app: hokiehelp-crawler, run: fulltest }
    spec:
      restartPolicy: Never
      volumes:
        - name: frontier
          persistentVolumeClaim: { claimName: crawler-frontier }
      containers:
        - name: crawler
          image: hokiehelp-crawler:vtedu-test
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef: { name: crawler-config }
          env:
            - { name: MINIO_BUCKET, value: "test-crawled-pages" }
            - { name: MINIO_CLEANED_BUCKET, value: "test-crawled-pages-cleaned" }
            - { name: MINIO_EXTERNAL_BUCKET, value: "test-crawled-pages-external" }
            - name: MINIO_ACCESS_KEY
              valueFrom: { secretKeyRef: { name: minio-credentials, key: access-key } }
            - name: MINIO_SECRET_KEY
              valueFrom: { secretKeyRef: { name: minio-credentials, key: secret-key } }
          volumeMounts:
            - { name: frontier, mountPath: /data }
          resources:
            requests: { memory: "6Gi", cpu: "1000m" }
            limits: { memory: "12Gi", cpu: "2000m" }
```

- [ ] **Step 3: Run it and watch**

```bash
kubectl config current-context
kubectl apply -f k8s/crawler-test-job.yaml
kubectl get pods -n test -l run=fulltest
kubectl logs -n test -l run=fulltest -f
```
Expected: discovery enumerates thousands of hosts, render batches progress, periodic summaries. The Job is resumable: if the pod restarts, it resumes from the frontier on the PVC.

- [ ] **Step 4: Verify completeness + quality on scratch buckets**

```bash
kubectl exec -n test deploy/<a-pod-with-uv> -- \
  uv run python scripts/verify_ingestion.py --bucket test-crawled-pages-cleaned --frontier /data/frontier.db
```
(or run the verifier as a short Job). Expected report: high ok-rate; failures dominated by genuinely empty source pages; frontier shows few `failed`. Run `requeue_failed()` once and re-run to mop up transient failures:
`sqlite3 /data/frontier.db "UPDATE frontier SET status='pending', error=NULL WHERE status='failed';"` then re-apply the Job.

- [ ] **Step 5: Decision gate before promotion**

Gate to promote to main: ok-rate ≥ agreed threshold (e.g. ≥ 95% of VT pages clean), `crawled-pages-external` contains only non-VT hosts, documents present, and spot-checks of dining/registrar/SPA pages contain real extracted content. Record results in the verification report.

- [ ] **Step 6: Commit the test Job manifest**

```bash
git add k8s/crawler-test-job.yaml
git commit -m "feat(k8s): full-domain crawler test Job against scratch buckets"
```

> **STOP — executor hands off here.** Promotion to the **main** buckets and the main Qdrant collection is performed by the repo owner, not the executor. For reference, promotion means: re-run the standard pipeline (crawler→chunker→embedder) against the default (main) configmap values, then validate retrieval with `k8s/retriever-job.yaml` (the retriever tester already logs full metadata; confirm the new `source_host`/`subdomain`/`page_type` payload fields appear and are filterable). Do not run this step automatically — present the verification report from Step 5 and wait for the owner to promote.

---

## Self-Review

**1. Spec coverage**
- "Discover the whole vt.edu domain incl. subdomains" → Phase 2 (crt.sh + CC + per-host seeding) ✓
- "Whatever pages and external links it has, one hop, stored separately with hop history" → `scope.py` (Phase 0), separate `crawled-pages-external` bucket + `hop_history` in metadata/frontier (Phases 0/1/3) ✓
- "Richer metadata for better indexing/retrieval" → Phases 0.2/0.3 + Phase 5 (chunker→embedder→Qdrant payload + indexes) ✓
- "Linked docs ingested" → documents pass routes PDF/DOCX to cleaned bucket (Task 4.2) ✓
- "SPA/JS pages — try, defer if impossible" → JS tier (Tasks 3.2–3.4) + `DEFERRED_SITES.md` policy (Task 4.3) ✓
- "Generic clean (no per-site regex)" → `fit_markdown`/PruningContentFilter, retire `cleaner.py` (Phase 3 + Task 4.4) ✓
- "Connect to pipeline, verify it works" → cleaned-bucket mirror keeps chunker contract; Phase 5 propagation; Phase 7 pipeline run ✓
- "Run on MacBook if speed good" → Task 4.3 bounded validation ✓
- "Full ingestion test on k8s test pod, verify before main" → Task 7.2 scratch-bucket Job + gates ✓
- "Confirm MD files have correct data, how much failed, get it perfect" → Phase 6 verification + failure accounting + requeue loop ✓

**2. Placeholder scan** — no "TBD/implement later"; each code step has full code. Two integration seams are called out explicitly with the exact change to make (documents `target_bucket` param in Task 4.2; pipeline Job volume in Task 7.1) rather than left vague. Phase 7 is an operational runbook (commands + gates), appropriate for ops, not code.

**3. Type consistency** — `Scope` enum values (`vt_page`/`external_page`/`document`/`out_of_scope`) are used consistently across `scope.py`, `frontier` (stored as `.value`), `engine.process_result`, and `render.bucket_for_scope`. `FrontierItem` fields match `lease_pending` output. `PageMetadata` new fields match `to_json`/`from_json`. `build_markdown_document` new kwargs match `engine` call site and chunker `parse_frontmatter` reads. `ChunkRecord.source_host/subdomain` match embedder payload keys.

**Note on scope size:** this plan spans five subsystems (discovery, frontier, render/extract, downstream metadata, verification) plus k8s ops. Each phase is independently testable and committable; you can also split Phases 1–3 (crawler core), Phase 5 (downstream), and Phases 6–7 (verify/ops) into separate execution sessions if preferred.
