# cs.vt.edu Full Domain Crawl Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the HokieHelp crawler from `website.cs.vt.edu` only to every subdomain of `cs.vt.edu`, remove the website.cs.vt.edu redirect-rewrite logic, fix the `#vt_main` CSS selector so non-VT-CMS pages are not silently dropped, unblock `wiki.cs.vt.edu`, fix a hardcoded URL in the cleaner, and scale K8s resources for the larger crawl scope.

**Architecture:** A new generic `_is_allowed_host(host, allowed_domains)` helper replaces hardcoded `website.cs.vt.edu` checks — it returns True for exact matches or any subdomain depth (e.g. `students.website.cs.vt.edu` correctly matches `cs.vt.edu`). The `css_selector="#vt_main"` is removed from both crawl configs so pages on non-VT-CMS subdomains (wiki, research labs, student sites) are not silently discarded; the cleaner handles boilerplate for all pages. Documents are collected from any host if the linking page is on cs.vt.edu — the trust anchor is the source page, not the document host. Config defaults, K8s ConfigMaps, pipeline defaults, and crawler job resources all update together.

**Tech Stack:** Python 3.12, pytest, crawl4ai, Kubernetes (kubectl apply -f), MinIO

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Remove `css_selector="#vt_main"` | Only VT CMS pages have this element. Any other cs.vt.edu subdomain returns empty markdown and silently fails. The cleaner handles boilerplate regardless. |
| No domain filter on document URLs | A document linked FROM a cs.vt.edu page is in scope regardless of where it is hosted. arxiv PDFs, vt.edu forms, publisher servers — all valid. |
| Unblock `wiki.cs.vt.edu` | CS department wikis contain structured knowledge (course docs, tool guides, policies) directly useful for RAG. |
| Keep `forum.cs.vt.edu` blocked | Forums are authentication-walled, noisy, and off-topic for an academic assistant. |
| Sub-sub-domains work automatically | `_is_allowed_host` uses `.endswith("." + domain)`, which matches any depth: `students.website.cs.vt.edu` ends with `.cs.vt.edu`. No special handling needed. |

---

## File Map

**Modify:**
- `services/crawler/src/crawler/crawl.py` — add `_is_allowed_host`, remove `_rewrite_to_website`, remove `css_selector` from both crawl configs, simplify phase 1/2 host validation, update external link scanning
- `services/crawler/src/crawler/cleaner.py:195-199` — update hardcoded `website.cs.vt.edu` regex to also match `cs.vt.edu`
- `services/crawler/src/crawler/config.py` — update defaults: seed URL → `https://cs.vt.edu`; allowed_domains → `cs.vt.edu`; blocked_domains → remove `students.cs.vt.edu`, remove `wiki.cs.vt.edu`
- `services/crawler/tests/conftest.py` — update `crawler_config` fixture defaults
- `services/crawler/tests/test_config.py` — update expected defaults assertions
- `services/crawler/tests/test_crawl.py` — remove rewrite-specific tests; add `_is_allowed_host` unit tests; update phase tests
- `services/crawler/tests/test_cleaner.py` — add test for updated URL regex
- `k8s/crawler-configmap.yaml` — update CRAWL_SEED_URL, CRAWL_ALLOWED_DOMAINS, CRAWL_BLOCKED_DOMAINS
- `k8s/pipeline-configmap.yaml` — update CRAWL_SEED_URL
- `services/pipeline/src/pipeline/config.py` — update default `crawl_seed_url`
- `services/pipeline/src/pipeline/jobs.py` — increase crawler job `active_deadline_seconds` and memory limit

---

### Task 1: Add `_is_allowed_host` helper and remove `_rewrite_to_website` from `crawl.py`

**Files:**
- Modify: `services/crawler/src/crawler/crawl.py:24-28`
- Test: `services/crawler/tests/test_crawl.py`

- [ ] **Step 1: Write failing unit tests for `_is_allowed_host`**

Add these tests to `services/crawler/tests/test_crawl.py` after the existing `_is_blocked_path` tests at the bottom of the file:

```python
from crawler.crawl import _is_allowed_host


def test_is_allowed_host_exact_match():
    assert _is_allowed_host("cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_subdomain():
    assert _is_allowed_host("website.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_deep_subdomain():
    assert _is_allowed_host("students.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_sub_sub_subdomain():
    # students.website.cs.vt.edu ends with .cs.vt.edu — must be accepted
    assert _is_allowed_host("students.website.cs.vt.edu", ("cs.vt.edu",)) is True


def test_is_allowed_host_unrelated_domain():
    assert _is_allowed_host("eng.vt.edu", ("cs.vt.edu",)) is False


def test_is_allowed_host_partial_suffix_no_match():
    # "notcs.vt.edu" must NOT match "cs.vt.edu"
    assert _is_allowed_host("notcs.vt.edu", ("cs.vt.edu",)) is False


def test_is_allowed_host_none_returns_false():
    assert _is_allowed_host(None, ("cs.vt.edu",)) is False


def test_is_allowed_host_empty_domains_returns_false():
    assert _is_allowed_host("cs.vt.edu", ()) is False
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd services/crawler && uv run pytest tests/test_crawl.py::test_is_allowed_host_exact_match -v
```

Expected: `ImportError` or `AttributeError` — `_is_allowed_host` does not exist yet.

- [ ] **Step 3: Add `_is_allowed_host` and remove `_rewrite_to_website` in `crawl.py`**

In `services/crawler/src/crawler/crawl.py`, replace the `_rewrite_to_website` function (lines 24–27) with:

```python
def _is_allowed_host(host: str | None, allowed_domains: tuple[str, ...]) -> bool:
    """Return True if host exactly matches or is a subdomain of any allowed domain.

    Works at any nesting depth: students.website.cs.vt.edu matches cs.vt.edu
    because the string ends with '.cs.vt.edu'.
    """
    if not host:
        return False
    for domain in allowed_domains:
        if host == domain or host.endswith("." + domain):
            return True
    return False
```

Remove `urlunparse` from the urllib.parse import (it was only used by `_rewrite_to_website`):

```python
from urllib.parse import urlparse
```

- [ ] **Step 4: Run the helper tests**

```bash
cd services/crawler && uv run pytest tests/test_crawl.py::test_is_allowed_host_exact_match tests/test_crawl.py::test_is_allowed_host_subdomain tests/test_crawl.py::test_is_allowed_host_deep_subdomain tests/test_crawl.py::test_is_allowed_host_sub_sub_subdomain tests/test_crawl.py::test_is_allowed_host_unrelated_domain tests/test_crawl.py::test_is_allowed_host_partial_suffix_no_match tests/test_crawl.py::test_is_allowed_host_none_returns_false tests/test_crawl.py::test_is_allowed_host_empty_domains_returns_false -v
```

Expected: All 8 PASS.

- [ ] **Step 5: Commit**

```bash
cd services/crawler && git add src/crawler/crawl.py tests/test_crawl.py
git commit -m "feat(crawler): add _is_allowed_host helper, remove _rewrite_to_website"
```

---

### Task 2: Remove `#vt_main` CSS selector and fix cleaner hardcoded URL

**Files:**
- Modify: `services/crawler/src/crawler/crawl.py:36-43` and `crawl.py:145-153`
- Modify: `services/crawler/src/crawler/cleaner.py:195-199`
- Test: `services/crawler/tests/test_cleaner.py`

`css_selector="#vt_main"` silently kills every page on a non-VT-CMS subdomain. When crawl4ai finds no matching element it returns empty markdown; `_store_result` then counts that page as failed. Removing the selector means all pages get their full markdown, and the cleaner handles boilerplate stripping for every domain.

The cleaner also has a regex that only strips bare "Computer Science" breadcrumb links pointing at `website.cs.vt.edu` — it needs to also strip the same link pointing at `cs.vt.edu`.

- [ ] **Step 1: Write failing cleaner test**

Add to `services/crawler/tests/test_cleaner.py`:

```python
def test_clean_markdown_removes_cs_vt_edu_breadcrumb_link():
    """Bare [Computer Science](https://cs.vt.edu) breadcrumb is stripped."""
    from crawler.cleaner import clean_markdown
    doc = "---\nurl: 'https://cs.vt.edu/about'\n---\n\n[Computer Science](https://cs.vt.edu)\n\n# About\n\nContent here."
    result = clean_markdown(doc)
    assert "[Computer Science](https://cs.vt.edu)" not in result
    assert "# About" in result
    assert "Content here." in result


def test_clean_markdown_still_removes_website_cs_vt_edu_breadcrumb_link():
    """Original website.cs.vt.edu breadcrumb link is still stripped."""
    from crawler.cleaner import clean_markdown
    doc = "---\nurl: 'https://website.cs.vt.edu/about'\n---\n\n[Computer Science](https://website.cs.vt.edu)\n\n# About\n\nContent here."
    result = clean_markdown(doc)
    assert "[Computer Science](https://website.cs.vt.edu)" not in result
    assert "# About" in result
```

- [ ] **Step 2: Run to confirm the cs.vt.edu test fails**

```bash
cd services/crawler && uv run pytest tests/test_cleaner.py::test_clean_markdown_removes_cs_vt_edu_breadcrumb_link -v
```

Expected: FAIL — the current regex only matches `website.cs.vt.edu`.

- [ ] **Step 3: Update the breadcrumb regex in `cleaner.py`**

In `services/crawler/src/crawler/cleaner.py`, replace lines 195–199:

```python
        if re.match(
            r"^\s*\[?\s*Computer Science\s*\]?\s*\(https://(?:website\.)?cs\.vt\.edu/?\)\s*$",
            line,
        ):
            continue
```

The change is `website\.cs\.vt\.edu` → `(?:website\.)?cs\.vt\.edu`, which matches both `https://cs.vt.edu` and `https://website.cs.vt.edu`.

- [ ] **Step 4: Remove `css_selector="#vt_main"` from both crawl configs in `crawl.py`**

In `services/crawler/src/crawler/crawl.py`, update `_make_markdown_config` (lines 36–43):

```python
def _make_markdown_config(request_delay: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        verbose=False,
        delay_before_return_html=request_delay,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        markdown_generator=DefaultMarkdownGenerator(),
    )
```

In the `bfs_config` construction inside `run_crawl` (around line 145), remove the `css_selector` line:

```python
    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=config.request_delay,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        markdown_generator=DefaultMarkdownGenerator(),
    )
```

- [ ] **Step 5: Run cleaner and crawler tests**

```bash
cd services/crawler && uv run pytest tests/test_cleaner.py tests/test_crawl.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd services/crawler && git add src/crawler/crawl.py src/crawler/cleaner.py tests/test_cleaner.py
git commit -m "fix(crawler): remove #vt_main selector so all cs.vt.edu subdomains are crawled, fix cleaner breadcrumb regex"
```

---

### Task 3: Rewrite `run_crawl` phase logic in `crawl.py`

**Files:**
- Modify: `services/crawler/src/crawler/crawl.py:113-295`
- Test: `services/crawler/tests/test_crawl.py`

Removes the phase 1 cs.vt.edu-rewrite block, the `crawl_allowed` manipulation hack, and the phase 2 redirect-rewrite. Documents are collected with the original signature — no domain filter on document URLs since the trust anchor is the page that linked them.

- [ ] **Step 1: Write failing tests for new behavior**

Remove these two tests from `services/crawler/tests/test_crawl.py` — their behavior is gone:
- `test_run_crawl_rewrites_cs_vt_edu_redirect_and_fetches`
- `test_run_crawl_queues_cs_subdomain_external_links`

Add these replacement tests:

```python
@pytest.mark.asyncio
async def test_run_crawl_accepts_any_cs_vt_edu_subdomain(crawler_config):
    """BFS results from any *.cs.vt.edu subdomain are stored directly — no rewrite."""
    mock_storage = _make_mock_storage()
    results = [
        _make_crawl_result("https://cs.vt.edu/about", "About", "# About CS", depth=0),
        _make_crawl_result("https://website.cs.vt.edu/people", "People", "# People", depth=1),
        _make_crawl_result("https://students.cs.vt.edu/portal", "Portal", "# Portal", depth=1),
        _make_crawl_result("https://wiki.cs.vt.edu/courses", "Courses", "# Courses", depth=1),
    ]

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            for r in results:
                yield r
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 4
    assert stats["pages_crawled"] == 4
    assert stats["pages_failed"] == 0


@pytest.mark.asyncio
async def test_run_crawl_skips_non_cs_vt_edu_domains(crawler_config):
    """Pages on non-cs.vt.edu domains are skipped."""
    mock_storage = _make_mock_storage()
    results = [
        _make_crawl_result("https://eng.vt.edu/page", "Eng", "# Eng", depth=0),
        _make_crawl_result("https://cs.vt.edu", "CS", "# CS VT", depth=0),
    ]

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            for r in results:
                yield r
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert stats["pages_crawled"] == 1   # only cs.vt.edu
    assert stats["pages_failed"] == 1    # eng.vt.edu skipped


@pytest.mark.asyncio
async def test_run_crawl_queues_cs_subdomain_external_links_phase2(crawler_config):
    """External *.cs.vt.edu links that BFS doesn't follow are queued and fetched in phase 2."""
    mock_storage = _make_mock_storage()

    bfs_result = _make_crawl_result(
        "https://cs.vt.edu",
        "Home",
        "# Home",
        depth=0,
        links={"external": [{"href": "https://research.cs.vt.edu/labs", "text": "Labs"}]},
    )
    phase2_result = _make_crawl_result(
        "https://research.cs.vt.edu/labs", "Labs", "# Labs", depth=0
    )

    mock_crawler_instance = AsyncMock()
    call_count = {"n": 0}

    async def fake_arun(url, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            async def _gen():
                yield bfs_result
            return _gen()
        else:
            return phase2_result

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 2
    assert stats["pages_crawled"] == 2


@pytest.mark.asyncio
async def test_run_crawl_collects_external_host_documents(crawler_config):
    """Documents linked from a cs.vt.edu page are collected even if hosted elsewhere."""
    mock_storage = _make_mock_storage()

    bfs_result = _make_crawl_result(
        "https://cs.vt.edu/research",
        "Research",
        "# Research",
        depth=0,
        links={
            "internal": [],
            "external": [{"href": "https://arxiv.org/pdf/2301.00001.pdf", "text": "Paper"}],
        },
    )

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield bfs_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    collected_doc_urls: list[set] = []

    async def fake_download(doc_urls, *args, **kwargs):
        collected_doc_urls.append(set(doc_urls))
        return {"documents_processed": 0, "documents_failed": 0}

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance), \
         patch("crawler.crawl.download_and_process_documents", side_effect=fake_download):
        await run_crawl(crawler_config, mock_storage)

    assert len(collected_doc_urls) == 1
    assert "https://arxiv.org/pdf/2301.00001.pdf" in collected_doc_urls[0]
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
cd services/crawler && uv run pytest tests/test_crawl.py::test_run_crawl_accepts_any_cs_vt_edu_subdomain tests/test_crawl.py::test_run_crawl_skips_non_cs_vt_edu_domains -v
```

Expected: FAIL — current code still has the cs.vt.edu rewrite path.

- [ ] **Step 3: Rewrite `run_crawl` in `crawl.py`**

Replace the entire `run_crawl` function (lines 113–295) with:

```python
async def run_crawl(config: CrawlerConfig, storage: MinioStorage) -> dict:
    """Execute a deep crawl and upload each page to storage.

    Returns a stats dict with pages_crawled, pages_failed, pages_skipped_duplicate counts.

    Two-phase strategy:
    1. BFS crawl seeded from config.seed_url. DomainFilter allows all config.allowed_domains
       (and their subdomains at any nesting depth). Results on any allowed host are stored
       directly — no URL rewriting.
    2. External links to allowed subdomains found during phase 1 that BFS did not follow
       (include_external=False) are individually fetched.
    3. Documents (PDF, Word) linked from any crawled page are downloaded regardless of
       the document host — the trust anchor is the cs.vt.edu page that linked them.
    """
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=list(config.allowed_domains),
            blocked_domains=list(config.blocked_domains),
        ),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=config.max_depth,
        include_external=False,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
    )

    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=config.request_delay,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        markdown_generator=DefaultMarkdownGenerator(),
    )

    stats = {"pages_crawled": 0, "pages_failed": 0, "pages_skipped_duplicate": 0,
             "documents_processed": 0, "documents_failed": 0}
    seen_content_hashes: set[str] = set(storage.load_all_content_hashes().keys())
    stored_urls: set[str] = set()
    # URLs to individually fetch in phase 2 (external subdomain links not followed by BFS)
    pending_fetches: set[str] = set()
    document_urls: set[str] = set()

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=config.seed_url, config=bfs_config):
            if not result.success:
                logger.warning(
                    "Failed to crawl %s: %s",
                    result.url,
                    getattr(result, "error_message", "unknown error"),
                )
                stats["pages_failed"] += 1
                continue

            parsed_url = urlparse(result.url)
            if parsed_url.scheme not in ("http", "https"):
                logger.warning("Skipping non-HTTP URL: %s", result.url)
                stats["pages_failed"] += 1
                continue

            if _is_blocked_path(result.url, config.blocked_paths):
                logger.debug("Blocked path, skipping: %s", result.url)
                stats["pages_failed"] += 1
                continue

            final_host = parsed_url.hostname

            if not _is_allowed_host(final_host, config.allowed_domains):
                logger.info(
                    "Skipping %s — host %s not in allowed domains",
                    result.url,
                    final_host,
                )
                stats["pages_failed"] += 1
                continue

            depth = result.metadata.get("depth", 0)
            _store_result(result, storage, stats, seen_content_hashes, stored_urls, depth)

            # Collect document links from this page — any host is valid, the
            # cs.vt.edu page that linked it is the trust anchor.
            document_urls.update(collect_document_links(result.links))

            # Queue external allowed-domain links the BFS won't follow (include_external=False)
            for link in (result.links or {}).get("external", []):
                href = link.get("href", "")
                if not href:
                    continue
                parsed = urlparse(href)
                host = parsed.hostname or ""
                if (
                    _is_allowed_host(host, config.allowed_domains)
                    and host not in config.blocked_domains
                    and href not in pending_fetches
                    and href not in stored_urls
                ):
                    logger.info("Queuing external cs.vt.edu link for phase 2: %s", href)
                    pending_fetches.add(href)

        # Phase 2: individually fetch queued external subdomain URLs
        if pending_fetches:
            logger.info("Phase 2: fetching %d queued URLs", len(pending_fetches))
            single_config = _make_markdown_config(config.request_delay)

            for url in pending_fetches:
                if url in stored_urls:
                    continue

                if _is_blocked_path(url, config.blocked_paths):
                    logger.debug("Blocked path (phase 2), skipping: %s", url)
                    continue

                result = await crawler.arun(url=url, config=single_config)
                if not result.success:
                    logger.warning(
                        "Phase 2 failed: %s — %s", url, getattr(result, "error_message", "")
                    )
                    stats["pages_failed"] += 1
                    continue

                final_host = urlparse(result.url).hostname

                if not _is_allowed_host(final_host, config.allowed_domains):
                    logger.info(
                        "Phase 2 skipping %s — host %s not allowed", result.url, final_host
                    )
                    stats["pages_failed"] += 1
                    continue

                _store_result(result, storage, stats, seen_content_hashes, stored_urls, depth=0)

    # Phase 3: download and process documents
    if document_urls:
        doc_stats = await download_and_process_documents(
            document_urls, storage, seen_content_hashes, stored_urls,
            request_delay=config.request_delay,
        )
        stats["documents_processed"] = doc_stats["documents_processed"]
        stats["documents_failed"] = doc_stats["documents_failed"]

    logger.info(
        "Deduplication: %d duplicate pages skipped",
        stats["pages_skipped_duplicate"],
    )
    logger.info(
        "Documents: %d processed, %d failed",
        stats["documents_processed"],
        stats["documents_failed"],
    )
    return stats
```

- [ ] **Step 4: Run all crawler tests**

```bash
cd services/crawler && uv run pytest tests/test_crawl.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd services/crawler && git add src/crawler/crawl.py tests/test_crawl.py
git commit -m "feat(crawler): expand to all cs.vt.edu subdomains, remove website rewrite logic"
```

---

### Task 4: Update `config.py` defaults

**Files:**
- Modify: `services/crawler/src/crawler/config.py:55-67`
- Test: `services/crawler/tests/test_config.py`
- Modify: `services/crawler/tests/conftest.py`

- [ ] **Step 1: Update failing assertions in `test_config.py`**

Replace the `test_config_from_env_defaults` test in `services/crawler/tests/test_config.py`:

```python
def test_config_from_env_defaults(monkeypatch):
    """Config loads with sensible defaults when only required vars are set."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")

    config = CrawlerConfig.from_env()

    assert config.minio_endpoint == "localhost:9000"
    assert config.minio_bucket == "crawled-pages"
    assert config.minio_cleaned_bucket == "crawled-pages-cleaned"
    assert config.minio_secure is False
    assert config.seed_url == "https://cs.vt.edu"
    assert config.max_depth == 4
    assert config.max_pages == 9999999
    assert config.allowed_domains == ("cs.vt.edu",)
    assert "git.cs.vt.edu" in config.blocked_domains
    assert "students.cs.vt.edu" not in config.blocked_domains
    assert "wiki.cs.vt.edu" not in config.blocked_domains
    assert config.prune_threshold == 0.45
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd services/crawler && uv run pytest tests/test_config.py::test_config_from_env_defaults -v
```

Expected: FAIL — current defaults still point at website.cs.vt.edu and block wiki/students.

- [ ] **Step 3: Update defaults in `config.py`**

In `services/crawler/src/crawler/config.py`, update the `from_env` classmethod:

Line 55 — change seed URL:
```python
seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
```

Lines 58–61 — change allowed_domains:
```python
allowed_domains=_domains(
    "CRAWL_ALLOWED_DOMAINS",
    "cs.vt.edu",
),
```

Lines 62–67 — change blocked_domains (remove `students.cs.vt.edu`, `wiki.cs.vt.edu`, `wordpress.cs.vt.edu`):
```python
blocked_domains=_domains(
    "CRAWL_BLOCKED_DOMAINS",
    "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,"
    "portal.cs.vt.edu,api.cs.vt.edu,forum.cs.vt.edu,login.cs.vt.edu",
),
```

- [ ] **Step 4: Update `conftest.py` fixture**

In `services/crawler/tests/conftest.py`, update the `crawler_config` fixture:

```python
@pytest.fixture
def crawler_config():
    """Minimal crawler config for testing."""
    return CrawlerConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="test-bucket",
        minio_cleaned_bucket="test-bucket-cleaned",
        minio_secure=False,
        seed_url="https://cs.vt.edu",
        max_depth=2,
        max_pages=100,
        allowed_domains=("cs.vt.edu",),
        blocked_domains=("git.cs.vt.edu", "login.cs.vt.edu"),
        blocked_paths=(),
        prune_threshold=0.45,
        request_delay=0.0,
    )
```

- [ ] **Step 5: Run all crawler tests**

```bash
cd services/crawler && uv run pytest tests/ -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd services/crawler && git add src/crawler/config.py tests/test_config.py tests/conftest.py
git commit -m "feat(crawler): default to cs.vt.edu, unblock students/wiki/wordpress subdomains"
```

---

### Task 5: Update K8s ConfigMaps

**Files:**
- Modify: `k8s/crawler-configmap.yaml`
- Modify: `k8s/pipeline-configmap.yaml`

- [ ] **Step 1: Update `k8s/crawler-configmap.yaml`**

Replace the file with:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: crawler-config
  namespace: test
  labels:
    app: hokiehelp-crawler
data:
  MINIO_ENDPOINT: "minio:9000"
  MINIO_BUCKET: "crawled-pages"
  MINIO_CLEANED_BUCKET: "crawled-pages-cleaned"
  MINIO_SECURE: "false"
  CRAWL_SEED_URL: "https://cs.vt.edu"
  CRAWL_MAX_DEPTH: "4"
  CRAWL_MAX_PAGES: "9999999"
  CRAWL_ALLOWED_DOMAINS: "cs.vt.edu"
  CRAWL_BLOCKED_DOMAINS: "git.cs.vt.edu,gitlab.cs.vt.edu,mail.cs.vt.edu,webmail.cs.vt.edu,portal.cs.vt.edu,api.cs.vt.edu,forum.cs.vt.edu,login.cs.vt.edu"
  CRAWL_BLOCKED_PATHS: "/content/,/editor.html,/cs-root.html,/cs-source.html"
  CRAWL_REQUEST_DELAY: "0.5"
  CRAWL_PRUNE_THRESHOLD: "0.45"
```

- [ ] **Step 2: Update `k8s/pipeline-configmap.yaml`**

Change `CRAWL_SEED_URL` to `https://cs.vt.edu`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pipeline-config
  namespace: test
  labels:
    app: hokiehelp-pipeline
data:
  CRAWL_SEED_URL: "https://cs.vt.edu"
  CRAWL_MAX_DEPTH: "4"
  CRAWL_MAX_PAGES: "9999999"
  PIPELINE_NAMESPACE: "test"
  PIPELINE_POLL_INTERVAL: "30"
```

- [ ] **Step 3: Commit**

```bash
git add k8s/crawler-configmap.yaml k8s/pipeline-configmap.yaml
git commit -m "feat(k8s): update crawler and pipeline configmaps to cs.vt.edu domain"
```

---

### Task 6: Update pipeline service config default and crawler job resources

**Files:**
- Modify: `services/pipeline/src/pipeline/config.py:36`
- Modify: `services/pipeline/src/pipeline/jobs.py:64-80`

- [ ] **Step 1: Update default `crawl_seed_url` in `services/pipeline/src/pipeline/config.py`**

Change line 36:

```python
crawl_seed_url=(
    seed_url
    or os.getenv("CRAWL_SEED_URL", "https://cs.vt.edu")
),
```

- [ ] **Step 2: Increase crawler job resources and deadline in `services/pipeline/src/pipeline/jobs.py`**

In `build_crawler_job`, replace the resources block (around line 64):

```python
resources=client.V1ResourceRequirements(
    requests={"memory": "2Gi", "cpu": "1000m"},
    limits={"memory": "6Gi", "cpu": "2000m"},
),
```

Change `active_deadline_seconds` from `28800` (8h) to `43200` (12h) in the same function:

```python
active_deadline_seconds=43200,
```

- [ ] **Step 3: Run pipeline tests**

```bash
cd services/pipeline && uv run pytest tests/ -v 2>/dev/null || echo "no pipeline unit tests"
```

- [ ] **Step 4: Commit**

```bash
git add services/pipeline/src/pipeline/config.py services/pipeline/src/pipeline/jobs.py
git commit -m "feat(pipeline): update seed URL default, increase crawler job resources for cs.vt.edu scale"
```

---

### Task 7: Apply to Kubernetes cluster and verify

- [ ] **Step 1: Show current kubectl context**

```bash
kubectl config current-context
```

Confirm you are NOT pointed at a production cluster.

- [ ] **Step 2: Apply updated ConfigMaps**

```bash
kubectl apply -f k8s/crawler-configmap.yaml
kubectl apply -f k8s/pipeline-configmap.yaml
```

Expected:
```
configmap/crawler-config configured
configmap/pipeline-config configured
```

- [ ] **Step 3: Verify ConfigMaps**

```bash
kubectl get configmap crawler-config -n test -o yaml
kubectl get configmap pipeline-config -n test -o yaml
```

Confirm `CRAWL_SEED_URL=https://cs.vt.edu`, `CRAWL_ALLOWED_DOMAINS=cs.vt.edu`, and `wiki.cs.vt.edu` absent from `CRAWL_BLOCKED_DOMAINS`.

- [ ] **Step 4: Verify running pods are unaffected**

```bash
kubectl get pods -n test
kubectl get svc -n test
```

- [ ] **Step 5: Optional smoke test — trigger a pipeline run**

Only run this if you have budget for a full crawl:

```bash
kubectl create job --from=cronjob/hokiehelp-pipeline hokiehelp-pipeline-manual-test -n test
kubectl get pods -n test -l managed-by=hokiehelp-pipeline -w
# Once crawler pod appears:
kubectl logs -f <crawler-pod-name> -n test
```

Confirm log lines like:
```
Crawling https://cs.vt.edu (max_depth=4, max_pages=9999999)
Stored page 1: https://cs.vt.edu (depth 0)
Stored page 2: https://website.cs.vt.edu/... (depth 1)
Stored page 3: https://wiki.cs.vt.edu/... (depth 1)
```

- [ ] **Step 6: Check chatbot is still healthy**

```bash
kubectl rollout status deployment/hokiehelp-chatbot -n test
```

- [ ] **Step 7: Commit any post-deployment adjustments**

```bash
git add -p
git commit -m "fix(k8s): post-deployment adjustments after cs.vt.edu crawl expansion"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|---|---|
| Accept any `*.cs.vt.edu` subdomain at any depth | Task 1 (`_is_allowed_host` with `.endswith`) |
| Accept `cs.vt.edu` itself | Task 1 (exact match branch) |
| Remove website.cs.vt.edu redirect-rewrite | Task 3 (`_rewrite_to_website` removed) |
| Documents collected regardless of host | Task 3 (`collect_document_links(result.links)` — no domain filter) |
| Non-VT-CMS pages no longer silently fail | Task 2 (remove `css_selector="#vt_main"`) |
| Unblock wiki.cs.vt.edu | Task 4 + Task 5 |
| Fix cleaner hardcoded URL | Task 2 (`cleaner.py` regex) |
| Update K8s manifests | Task 5 |
| Work with all services (pipeline) | Task 6 |
| Scale crawler resources | Task 6 |
| Deploy to test namespace | Task 7 |

### Type Consistency

- `_is_allowed_host(host: str | None, allowed_domains: tuple[str, ...]) -> bool` — defined Task 1, used in Task 3 (`run_crawl`). Match confirmed.
- `collect_document_links(result.links)` in Task 3 — original signature, no new params introduced. Match confirmed.
- `crawler_config` fixture uses `allowed_domains=("cs.vt.edu",)` — matches `CrawlerConfig.allowed_domains: tuple[str, ...]`. Match confirmed.

### Placeholder Scan

No TBD, TODO, or placeholder steps. All code blocks are complete.
