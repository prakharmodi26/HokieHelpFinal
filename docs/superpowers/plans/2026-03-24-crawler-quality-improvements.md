# Crawler Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve RAG quality through a three-phase approach: (1) deep URL discovery to map the full domain, (2) human review to confirm block/allow path patterns, (3) implement those patterns in the crawler + staleness filtering in the chunker for weekly recrawls.

**Architecture:**
- Phase A (Tasks 1–2): Standalone discovery script does a deep BFS collecting only URLs (no content), outputs grouped report for human review. Human confirms block patterns before any code changes.
- Phase B (Tasks 3–4): Implement confirmed block patterns in `CrawlerConfig` + `crawl.py`; update k8s configmap to depth=4, max_pages=999999.
- Phase C (Tasks 5–6): Add `is_stale_time_sensitive_page()` to chunker `parser.py`; call it in `main.py` to skip seminar/news pages older than 6 months before embedding.

**⚠ Hard Stop after Task 2:** Do NOT proceed to Task 3 until the human has reviewed the discovery report and explicitly confirmed the block path patterns.

**Tech Stack:** Python 3.11, Crawl4AI, MinIO, pytest, kubectl/k8s ConfigMaps

---

## File Map

| File | Change |
|---|---|
| `services/crawler/scripts/discover_urls.py` | New — standalone URL discovery script (not part of main service) |
| `services/crawler/src/crawler/config.py` | Add `blocked_paths: tuple[str, ...]` field |
| `services/crawler/tests/conftest.py` | Add `blocked_paths=()` to `crawler_config` fixture |
| `services/crawler/src/crawler/crawl.py` | Add `_is_blocked_path()` helper; skip blocked pages in BFS and Phase 2 |
| `services/crawler/tests/test_crawl.py` | Tests for `_is_blocked_path()` |
| `k8s/crawler-configmap.yaml` | depth=4, max_pages=999999, add CRAWL_BLOCKED_PATHS |
| `services/chunker/src/chunker/parser.py` | Add `is_stale_time_sensitive_page(url, body)` |
| `services/chunker/src/chunker/main.py` | Call staleness check before chunking; update import |
| `services/chunker/tests/test_parser.py` | Tests for staleness detection |
| `services/chunker/tests/test_main.py` | Integration test for stale seminar skipping |

---

## PHASE A — URL Discovery

---

## Task 1: Write and run the URL discovery script

**Files:**
- Create: `services/crawler/scripts/discover_urls.py`

This is a standalone script — not part of the main crawler service, no tests needed. It uses Crawl4AI to do a deep BFS collecting only URLs (no content processing, no MinIO storage), then prints a report grouped by path prefix.

- [ ] **Step 1: Create the scripts directory**

```bash
mkdir -p services/crawler/scripts
```

- [ ] **Step 2: Write `discover_urls.py`**

```python
#!/usr/bin/env python3
"""URL discovery script — maps the full website.cs.vt.edu domain.

Run from the services/crawler directory:
    python scripts/discover_urls.py [--depth N] [--max-pages N]

Outputs all discovered URLs grouped by top-level path prefix.
Use this report to decide which paths to block in CRAWL_BLOCKED_PATHS.
"""
import argparse
import asyncio
import sys
from collections import defaultdict
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain


SEED_URL = "https://website.cs.vt.edu"
ALLOWED_DOMAIN = "website.cs.vt.edu"


async def discover(max_depth: int, max_pages: int) -> list[str]:
    """Run BFS and return all successfully reached URLs."""
    filter_chain = FilterChain([
        DomainFilter(allowed_domains=[ALLOWED_DOMAIN]),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        include_external=False,
        max_pages=max_pages,
        filter_chain=filter_chain,
    )

    # Minimal config — we only care about URLs, not content
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=0.5,
    )

    urls: list[str] = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=SEED_URL, config=run_config):
            if result.success:
                urls.append(result.url)
                print(f"  [{len(urls)}] {result.url}", file=sys.stderr)

    return urls


def report(urls: list[str]) -> None:
    """Print a grouped report of all discovered URLs."""
    by_prefix: dict[str, list[str]] = defaultdict(list)

    for url in sorted(urls):
        path = urlparse(url).path.lstrip("/")
        prefix = path.split("/")[0] if "/" in path else path
        by_prefix[prefix].append(url)

    print(f"\n{'='*60}")
    print(f"DISCOVERY REPORT — {len(urls)} pages found")
    print(f"{'='*60}\n")

    for prefix, group in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        print(f"[{len(group):4d}]  /{prefix}/")
        for url in group[:5]:
            print(f"          {url}")
        if len(group) > 5:
            print(f"          ... and {len(group) - 5} more")
        print()

    print(f"\nFull URL list:")
    for url in sorted(urls):
        print(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover all URLs on website.cs.vt.edu")
    parser.add_argument("--depth", type=int, default=5, help="BFS max depth (default: 5)")
    parser.add_argument("--max-pages", type=int, default=5000, help="Max pages to crawl (default: 5000)")
    args = parser.parse_args()

    print(f"Starting discovery: depth={args.depth}, max_pages={args.max_pages}", file=sys.stderr)
    urls = asyncio.run(discover(args.depth, args.max_pages))
    report(urls)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the discovery crawl**

From the `services/crawler` directory (needs the crawler virtualenv with crawl4ai installed):

```bash
cd services/crawler
pip install -e ".[dev]" -q
python scripts/discover_urls.py --depth 5 --max-pages 5000 2>discovery_progress.log | tee discovery_report.txt
```

This will take 15–30 minutes. Progress logs go to `discovery_progress.log`, the final report to `discovery_report.txt`.

- [ ] **Step 4: Commit the script (not the output files)**

```bash
git add services/crawler/scripts/discover_urls.py
git commit --author="prakharmodi26 <prakharmodi25@gmail.com>" -m "feat(crawler): add URL discovery script for path analysis"
```

---

## Task 2: Review discovery report and confirm block patterns

**⚠ HARD STOP — Human review required before proceeding.**

- [ ] **Step 1: Review the discovery report**

Open `discovery_report.txt`. For each path prefix in the grouped output, decide:
- **Block**: pure noise (tag indexes, admin forms, login pages)
- **Keep**: useful content for the RAG (faculty, courses, research, news, seminars)
- **Review further**: unclear — look at sample URLs

Expected noise candidates based on prior analysis (but confirm against actual report):
- `/content/` — CMS tag/category index pages
- `/Graduate/grforceadd` — admin form
- Any new sections not previously seen

- [ ] **Step 2: Present findings to the human**

Show the grouped report output. For each potential block pattern, state:
1. How many pages it covers
2. Sample URLs
3. Why it's noise or useful

**Do not proceed to Task 3 until the human explicitly confirms:**
- Which paths to add to `CRAWL_BLOCKED_PATHS`
- Which paths to keep (even if large — e.g., seminars are filtered by date in Phase C)

---

## PHASE B — Implement Block Patterns

*(Only start after human confirms patterns in Task 2)*

---

## Task 3: Add blocked-path config to the crawler (TDD)

**Files:**
- Modify: `services/crawler/src/crawler/config.py`
- Modify: `services/crawler/tests/conftest.py`
- Modify: `services/crawler/src/crawler/crawl.py`
- Modify: `services/crawler/tests/test_crawl.py`

Use the exact block patterns confirmed by the human in Task 2 for all `CRAWL_BLOCKED_PATHS` defaults below.

- [ ] **Step 1: Add `blocked_paths` field to `CrawlerConfig`**

In `config.py`, add the field after `blocked_domains` and load it via the existing `_domains()` helper (which splits comma-separated strings — reusable for path prefixes):

```python
# Add field after blocked_domains:
blocked_paths: tuple[str, ...]

# Add inside from_env() after blocked_domains= line:
blocked_paths=_domains(
    "CRAWL_BLOCKED_PATHS",
    "<CONFIRMED_PATHS_FROM_TASK_2>",   # e.g. "/content/,/Graduate/grforceadd"
),
```

- [ ] **Step 2: Update `conftest.py` to include the new field**

The `crawler_config` fixture constructs `CrawlerConfig(...)` directly. Without updating it, all existing tests break. Add `blocked_paths=()` to the fixture:

```python
return CrawlerConfig(
    minio_endpoint="localhost:9000",
    minio_access_key="minioadmin",
    minio_secret_key="minioadmin",
    minio_bucket="test-bucket",
    minio_cleaned_bucket="test-bucket-cleaned",
    minio_secure=False,
    seed_url="https://website.cs.vt.edu",
    max_depth=2,
    max_pages=100,
    allowed_domains=("website.cs.vt.edu",),
    blocked_domains=("git.cs.vt.edu", "login.cs.vt.edu", "students.cs.vt.edu"),
    blocked_paths=(),          # new field — empty for tests
    prune_threshold=0.45,
    request_delay=0.0,
)
```

- [ ] **Step 3: Run existing crawler tests to confirm no regressions**

```bash
cd services/crawler
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Write failing tests for `_is_blocked_path`**

Add to `services/crawler/tests/test_crawl.py`:

```python
from crawler.crawl import _is_blocked_path


def test_blocked_path_matches_content_prefix():
    blocked = ("/content/", "/Graduate/grforceadd")
    assert _is_blocked_path("https://website.cs.vt.edu/content/tags/foo", blocked) is True


def test_blocked_path_matches_admin_form():
    blocked = ("/content/", "/Graduate/grforceadd")
    assert _is_blocked_path("https://website.cs.vt.edu/Graduate/grforceadd.html", blocked) is True


def test_blocked_path_allows_faculty_page():
    blocked = ("/content/", "/Graduate/grforceadd")
    assert _is_blocked_path("https://website.cs.vt.edu/people/faculty/denis-gracanin.html", blocked) is False


def test_blocked_path_allows_seminar_page():
    blocked = ("/content/", "/Graduate/grforceadd")
    assert _is_blocked_path("https://website.cs.vt.edu/research/Seminars/Ali_Butt.html", blocked) is False


def test_blocked_path_empty_blocks_nothing():
    assert _is_blocked_path("https://website.cs.vt.edu/content/anything", ()) is False
```

- [ ] **Step 5: Run to confirm they fail**

```bash
python -m pytest tests/test_crawl.py -v -k "blocked_path"
```

Expected: `ImportError` — `_is_blocked_path` not defined yet.

- [ ] **Step 6: Implement `_is_blocked_path` and wire it into the crawl loop**

In `crawl.py`, add after `_rewrite_to_website`:

```python
def _is_blocked_path(url: str, blocked_paths: tuple[str, ...]) -> bool:
    """Return True if the URL path starts with any blocked prefix."""
    path = urlparse(url).path
    return any(path.startswith(prefix) for prefix in blocked_paths)
```

In the BFS `async for result in ...` loop, immediately after `if not result.success:` block:

```python
if _is_blocked_path(result.url, config.blocked_paths):
    logger.debug("Blocked path, skipping: %s", result.url)
    stats["pages_failed"] += 1
    continue
```

In the Phase 2 `for url in pending_fetches:` loop, before `_store_result(...)`:

```python
if _is_blocked_path(url, config.blocked_paths):
    logger.debug("Blocked path (phase 2), skipping: %s", url)
    continue
```

- [ ] **Step 7: Run all crawler tests**

```bash
python -m pytest tests/ -v
```

Expected: all pass including the 5 new `blocked_path` tests.

- [ ] **Step 8: Commit**

```bash
git add services/crawler/src/crawler/config.py \
        services/crawler/tests/conftest.py \
        services/crawler/src/crawler/crawl.py \
        services/crawler/tests/test_crawl.py
git commit --author="prakharmodi26 <prakharmodi25@gmail.com>" -m "feat(crawler): add blocked_paths config and path filtering in crawl loop"
```

---

## Task 4: Update k8s crawler configmap

**Files:**
- Modify: `k8s/crawler-configmap.yaml`

- [ ] **Step 1: Update the configmap with confirmed settings**

Replace `k8s/crawler-configmap.yaml`:

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
  CRAWL_SEED_URL: "https://website.cs.vt.edu"
  CRAWL_MAX_DEPTH: "4"
  CRAWL_MAX_PAGES: "999999"
  CRAWL_ALLOWED_DOMAINS: "website.cs.vt.edu"
  CRAWL_BLOCKED_PATHS: "<CONFIRMED_PATHS_FROM_TASK_2>"
  CRAWL_REQUEST_DELAY: "0.5"
  CRAWL_PRUNE_THRESHOLD: "0.45"
```

- [ ] **Step 2: Apply and verify**

```bash
kubectl config current-context   # confirm: endeavour
kubectl apply -f k8s/crawler-configmap.yaml
kubectl get configmap crawler-config -n test -o jsonpath='{.data.CRAWL_MAX_DEPTH}'
# Expected: 4
kubectl get configmap crawler-config -n test -o jsonpath='{.data.CRAWL_BLOCKED_PATHS}'
# Expected: confirmed block patterns
```

- [ ] **Step 3: Commit**

```bash
git add k8s/crawler-configmap.yaml
git commit --author="prakharmodi26 <prakharmodi25@gmail.com>" -m "feat(k8s): update crawler config — depth=4, max_pages=999999, blocked paths"
```

---

## PHASE C — Staleness Filtering in Chunker

---

## Task 5: Add staleness detection to the chunker parser (TDD)

**Files:**
- Modify: `services/chunker/src/chunker/parser.py`
- Modify: `services/chunker/tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

Add to `services/chunker/tests/test_parser.py`:

```python
from datetime import datetime, timezone, timedelta
from chunker.parser import is_stale_time_sensitive_page


SEMINAR_URL = "https://website.cs.vt.edu/research/Seminars/Ali_Butt.html"
NEWS_URL = "https://website.cs.vt.edu/News/Seminars/someone.html"
FACULTY_URL = "https://website.cs.vt.edu/people/faculty/denis-gracanin.html"


def _body_with_date(date: datetime) -> str:
    """Fake seminar body with date in the VT CS site format."""
    return (
        f"### {date.strftime('%A')}, {date.strftime('%B')} {date.day}, {date.year}"
        " 2:30 - 3:45 p.m. Room 260\n### Abstract\nSome content here."
    )


def test_non_seminar_page_never_stale():
    body = _body_with_date(datetime.now(timezone.utc) - timedelta(days=400))
    assert is_stale_time_sensitive_page(FACULTY_URL, body) is False


def test_recent_seminar_not_stale():
    recent = datetime.now(timezone.utc) - timedelta(days=30)
    assert is_stale_time_sensitive_page(SEMINAR_URL, _body_with_date(recent)) is False


def test_old_seminar_is_stale():
    old = datetime.now(timezone.utc) - timedelta(days=200)
    assert is_stale_time_sensitive_page(SEMINAR_URL, _body_with_date(old)) is True


def test_recent_news_seminar_not_stale():
    recent = datetime.now(timezone.utc) - timedelta(days=45)
    assert is_stale_time_sensitive_page(NEWS_URL, _body_with_date(recent)) is False


def test_old_news_seminar_is_stale():
    old = datetime.now(timezone.utc) - timedelta(days=210)
    assert is_stale_time_sensitive_page(NEWS_URL, _body_with_date(old)) is True


def test_seminar_no_date_is_not_stale():
    """If date cannot be parsed, keep the page (fail safe)."""
    assert is_stale_time_sensitive_page(SEMINAR_URL, "### Abstract\nNo date here.") is False
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd services/chunker
python -m pytest tests/test_parser.py -v -k "stale"
```

Expected: `ImportError` — function doesn't exist yet.

- [ ] **Step 3: Implement `is_stale_time_sensitive_page` in `parser.py`**

Add new imports at the top of `parser.py` (after existing imports):

```python
import calendar
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
```

Add constants and function after the existing `_CMS_ERROR_RE` constant:

```python
_TIME_SENSITIVE_PATHS = (
    "/research/Seminars/",
    "/News/Seminars/",
)

# Matches "Friday, November 7, 2025" — date format on VT CS seminar/news pages
_EVENT_DATE_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r",\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})",
)

_STALENESS_DAYS = 180  # ~6 months


def is_stale_time_sensitive_page(url: str, body: str) -> bool:
    """Return True if a seminar/news page's event date is older than 6 months.

    Only applies to /research/Seminars/ and /News/Seminars/ paths.
    All other pages return False. If date cannot be extracted, returns False
    (fail safe — keep the page rather than silently dropping it).
    """
    path = urlparse(url).path
    if not any(path.startswith(prefix) for prefix in _TIME_SENSITIVE_PATHS):
        return False

    match = _EVENT_DATE_RE.search(body[:2000])  # date is near the top
    if not match:
        return False

    month_name, day_str, year_str = match.group(1), match.group(2), match.group(3)
    try:
        month_num = list(calendar.month_name).index(month_name)
        event_date = datetime(int(year_str), month_num, int(day_str), tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return False

    return (datetime.now(timezone.utc) - event_date).days > _STALENESS_DAYS
```

- [ ] **Step 4: Run staleness tests**

```bash
python -m pytest tests/test_parser.py -v -k "stale"
```

Expected: all 6 pass.

- [ ] **Step 5: Run full chunker test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add services/chunker/src/chunker/parser.py services/chunker/tests/test_parser.py
git commit --author="prakharmodi26 <prakharmodi25@gmail.com>" -m "feat(chunker): detect stale seminar/news pages by event date"
```

---

## Task 6: Call staleness check in the chunker main loop (TDD)

**Files:**
- Modify: `services/chunker/src/chunker/main.py`
- Modify: `services/chunker/tests/test_main.py`

- [ ] **Step 1: Write failing integration test**

Add to `services/chunker/tests/test_main.py` (using the existing `_make_storage` helper and `chunker_config` fixture):

```python
def test_stale_seminar_is_skipped(chunker_config):
    """A seminar page older than 6 months must be skipped, not chunked."""
    from datetime import datetime, timezone, timedelta
    old = datetime.now(timezone.utc) - timedelta(days=200)
    stale_doc = (
        "---\n"
        "doc_id: 'stale00000000001'\n"
        "url: 'https://website.cs.vt.edu/research/Seminars/OldSeminar.html'\n"
        "title: 'Old Seminar'\n"
        "content_hash: 'abc123'\n"
        "crawl_timestamp: '2024-01-01T00:00:00+00:00'\n"
        "---\n"
        f"### {old.strftime('%A')}, {old.strftime('%B')} {old.day}, {old.year} 2:30 p.m.\n"
        "### Abstract\nOld content that should not be indexed."
    )
    storage = _make_storage({"website.cs.vt.edu/research/Seminars/OldSeminar.md": stale_doc})
    stats = run_chunking(storage, chunker_config)
    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert not storage.upload_chunks.called
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd services/chunker
python -m pytest tests/test_main.py::test_stale_seminar_is_skipped -v
```

Expected: FAIL — `processed == 1` (staleness check not wired in yet).

- [ ] **Step 3: Update import and add staleness check in `main.py`**

Update the import at the top of `main.py`:

```python
from chunker.parser import parse_frontmatter, split_sections, is_cms_error_page, is_stale_time_sensitive_page
```

In the `for key in keys:` loop, after the `is_cms_error_page` block:

```python
if is_stale_time_sensitive_page(frontmatter.url, body):
    logger.info("Skipping stale time-sensitive page: %s", frontmatter.url)
    stats["skipped"] += 1
    continue
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass including `test_stale_seminar_is_skipped`.

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/main.py services/chunker/tests/test_main.py
git commit --author="prakharmodi26 <prakharmodi25@gmail.com>" -m "feat(chunker): skip stale seminar/news pages in chunking loop"
```

---

## Task 7: Push and verify CI

- [ ] **Step 1: Push all commits**

```bash
git push
```

- [ ] **Step 2: Confirm CI passes**

```bash
gh run list --workflow="Crawler CI" --limit=2
gh run list --workflow="Chunker CI" --limit=2
```

Expected: both show `success`.

---

## Summary

| Phase | What | Result |
|---|---|---|
| A — Discovery | Deep BFS to map all URLs | Report for human review |
| A — Review | Human confirms block patterns | Agreed path list |
| B — Crawler | `blocked_paths` in config + crawl loop | Noise paths never stored |
| B — K8s | depth=4, max_pages=999999 | Full site coverage on recrawl |
| C — Chunker | Date-based filtering for seminars/news | Only last 6 months in Qdrant |
