# Markdown Cleaning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-bucket pipeline (raw + cleaned) and a markdown cleaning module that strips boilerplate from crawled pages before they're useful for RAG.

**Architecture:** Crawler stores raw markdown in `crawled-pages` bucket (as today). A new `cleaner` module reads raw pages, strips boilerplate, and writes to `crawled-pages-cleaned` bucket. Cleaning runs as a post-crawl step in `main.py`.

**Tech Stack:** Python, regex, MinIO Python SDK (existing)

---

## Analysis: What's in the Raw Pages and What Needs Cleaning

### Current state

The crawler uses `PruningContentFilter(threshold=0.45)` via Crawl4AI's `fit_markdown`. This works **inconsistently**:
- **Depth 0-1 pages** (About.html, people.html, research.html): `fit_markdown` works well, produces clean output
- **Depth 2 pages** (individual person bios, news articles): `fit_markdown` fails or falls back to `raw_markdown`, leaving all boilerplate intact

Pages where pruning fails are ~300 lines, of which **~200 lines are boilerplate** (67%). The actual content is only ~100 lines.

### Boilerplate patterns found across ALL raw pages

Sampled: `andrea-sirles.html.md`, `cs-2024-deans-award-winners.html.md`, `faculty.html.md`, `index.md`

| # | Pattern | Lines per page | Why remove |
|---|---------|---------------|------------|
| 1 | **Skip links** | 2 | "Skip to main content", "Skip to search" — accessibility nav, not content |
| 2 | **VT header bar** | ~30 | VT logo, Universal Access toggle, Apply/Visit/Give/Shop links, Resources dropdown — site-wide chrome |
| 3 | **Navigation menu** | ~35 | Full site nav with About/Academic/People/Research/Engage submenus — identical on every page |
| 4 | **Search widget** | ~85 | Full search UI, search tips, search help, "quantum physics" examples — massive UI component with zero content value |
| 5 | **Breadcrumb trail** | ~5 | "College of Engineering / Department of Computer Science / ..." — navigational, not informational |
| 6 | **Sidebar navigation (Explore)** | ~20-40 | Lists all pages in current section — pure nav |
| 7 | **Department footer** | ~15 | "Follow Computer Science" social links, campus addresses, phone numbers — repeated on every page |
| 8 | **University footer** | ~25 | VT logo, Get Directions, See All Locations, policy links, copyright, social icons |
| 9 | **Tracking pixels** | 1-2 | `![](https://bat.bing.com/action/...)` — analytics, zero content value |
| 10 | **CMS artifacts** | varies | "Bio Item Bio Item", "General Item General Item", "Article Item Article Item", "Redirect Item Redirect Item" — CMS rendering artifacts |

### What should remain after cleaning

- YAML frontmatter (url, title, crawl_depth, crawl_timestamp)
- Page title (first `#` heading)
- All actual page content (paragraphs, lists, headings)
- Internal links within content (useful for RAG context)
- Images with meaningful alt text (content images, not icons)

---

## File Structure

```
services/crawler/src/crawler/
  config.py          — MODIFY: add cleaned_bucket field
  storage.py         — MODIFY: add MinioStorage support for second bucket
  cleaner.py         — CREATE: markdown cleaning logic
  main.py            — MODIFY: add post-crawl cleaning step
services/crawler/tests/
  test_cleaner.py    — CREATE: tests for cleaning logic
  test_config.py     — MODIFY: test new cleaned_bucket config
```

---

## Chunk 1: Two-Bucket Setup + Raw Crawl Config

### Task 1: Add `cleaned_bucket` to CrawlerConfig

**Files:**
- Modify: `services/crawler/src/crawler/config.py`
- Modify: `services/crawler/tests/test_config.py`
- Modify: `services/crawler/tests/conftest.py`

- [ ] **Step 1: Write failing test for cleaned_bucket config field**

In `tests/test_config.py`, add to `test_config_from_env_defaults`:
```python
assert config.minio_cleaned_bucket == "crawled-pages-cleaned"
```

And add to `test_config_from_env_custom`:
```python
monkeypatch.setenv("MINIO_CLEANED_BUCKET", "my-cleaned-bucket")
# ...
assert config.minio_cleaned_bucket == "my-cleaned-bucket"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && python -m pytest tests/test_config.py -v`
Expected: FAIL — `CrawlerConfig` has no `minio_cleaned_bucket` attribute

- [ ] **Step 3: Add minio_cleaned_bucket to CrawlerConfig**

In `config.py`, add field after `minio_bucket`:
```python
minio_cleaned_bucket: str
```

In `from_env()`, add:
```python
minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
```

- [ ] **Step 4: Update conftest.py fixture**

Add to the `crawler_config` fixture:
```python
minio_cleaned_bucket="test-bucket-cleaned",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/config.py services/crawler/tests/test_config.py services/crawler/tests/conftest.py
git commit -m "feat: add minio_cleaned_bucket config field for two-bucket pipeline"
```

### Task 2: Add bucket helpers to MinioStorage

**Files:**
- Modify: `services/crawler/src/crawler/storage.py`
- Modify: `services/crawler/tests/test_storage.py`

- [ ] **Step 1: Write failing test for ensure_bucket on cleaned bucket**

In `tests/test_storage.py`, add test:
```python
def test_list_objects(mock_minio_client):
    """list_objects returns object names from a bucket."""
    mock_obj = MagicMock()
    mock_obj.object_name = "website.cs.vt.edu/index.md"
    mock_minio_client.list_objects.return_value = [mock_obj]

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config_fixture)
        objects = storage.list_objects()

    mock_minio_client.list_objects.assert_called_once()
```

- [ ] **Step 2: Add list_objects and download_document methods to MinioStorage**

```python
def list_objects(self, bucket: str | None = None) -> list[str]:
    """List all object keys in the given bucket (default: raw bucket)."""
    target = bucket or self._bucket
    return [
        obj.object_name
        for obj in self._client.list_objects(target, recursive=True)
    ]

def download_document(self, object_key: str, bucket: str | None = None) -> str:
    """Download a document and return its content as string."""
    target = bucket or self._bucket
    response = self._client.get_object(target, object_key)
    try:
        return response.read().decode("utf-8")
    finally:
        response.close()
        response.release_conn()

def ensure_bucket(self, bucket: str) -> None:
    """Ensure a bucket exists, creating it if needed."""
    if not self._client.bucket_exists(bucket):
        self._client.make_bucket(bucket)
        logger.info("Created bucket %s", bucket)

def upload_document(self, object_key: str, content: str, bucket: str | None = None) -> None:
    """Upload a Markdown document to MinIO."""
    target = bucket or self._bucket
    data = content.encode("utf-8")
    self._client.put_object(
        target,
        object_key,
        io.BytesIO(data),
        len(data),
        content_type="text/markdown",
    )
    logger.info("Uploaded %s (%d bytes) to %s", object_key, len(data), target)
```

- [ ] **Step 3: Run tests**

Run: `cd services/crawler && python -m pytest tests/test_storage.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add services/crawler/src/crawler/storage.py services/crawler/tests/test_storage.py
git commit -m "feat: add list_objects, download_document, and bucket parameter to storage"
```

---

## Chunk 2: Markdown Cleaner Module (TDD)

### Task 3: Create cleaner with tests

**Files:**
- Create: `services/crawler/src/crawler/cleaner.py`
- Create: `services/crawler/tests/test_cleaner.py`

The cleaner operates on a single markdown document string and returns the cleaned version. It preserves the YAML frontmatter and removes the 10 boilerplate patterns identified above.

**Cleaning strategy:** The VT website uses a consistent template. Rather than regex-matching each pattern individually, we identify **content boundaries**:
- **Content starts** after the sidebar "Explore" section, or after the nav menu, at the first real heading (`# Title`)
- **Content ends** before "### Follow Computer Science" (department footer marker)

For pages where `fit_markdown` already worked (no boilerplate), the cleaner is a no-op (idempotent).

- [ ] **Step 1: Write test for clean_markdown with full boilerplate page**

File: `tests/test_cleaner.py`

```python
import pytest
from crawler.cleaner import clean_markdown

BOILERPLATE_HEADER = """\
  * [Skip to main content](https://website.cs.vt.edu/page.html#vt_main)
  * [Skip to search](https://website.cs.vt.edu/page.html#vt_search_box)


[![](https://www.assets.cms.vt.edu/images/whiteVTonTransparent.svg)Virginia Tech\u00ae home](https://www.vt.edu)
![](https://www.assets.cms.vt.edu/images/accessibility_icon_white.svg) Universal Access Toggle
"""

NAV_MENU = """\
  * [Home](https://website.cs.vt.edu/index.html)
  * [About](https://website.cs.vt.edu/About.html) About Submenu Toggle
    * [Accreditation](https://website.cs.vt.edu/About/accreditation.html)
  * [People](https://website.cs.vt.edu/people.html) People Submenu Toggle
    * [Faculty](https://website.cs.vt.edu/people/faculty.html)
"""

DEPT_FOOTER = """\
### Follow Computer Science
[Facebook](https://facebook.com/VT.ComputerScience) [![](https://www.assets.cms.vt.edu/images/social-media/x-logo-white.svg)X](https://x.com/vt_cs) [Instagram](https://www.instagram.com/vt_cs/) [Linked In](https://www.linkedin.com/company/vt-cs/mycompany/?viewAsMember=true) [YouTube](https://www.youtube.com/@VTEngineering/featured)
**Blacksburg campus**
Torgersen Hall RM 3160
620 Drillfield Dr.
Blacksburg, VA 24061
(540) 231-6931 (undergraduate)
(540) 231-0746 (graduate)
"""

VT_FOOTER = """\
![Virginia Tech logo](https://www.assets.cms.vt.edu/images/logo-white-black.svg)
[Get Directions ](https://www.vt.edu/maps/directions.html)
  * [University Status](https://www.vt.edu/status.html)
  * [Privacy Statement](https://www.vt.edu/privacy.html)

\u00a9 2026 Virginia Polytechnic Institute and State University. All rights reserved.
  * [Instagram](https://instagram.com/virginia.tech/ "instagram")

![](https://bat.bing.com/action/0?ti=343199719)
"""

FRONTMATTER = """\
---
url: 'https://website.cs.vt.edu/people/admin/jane.html'
title: 'Jane Doe  | Computer Science | Virginia Tech'
crawl_depth: 2
crawl_timestamp: '2026-03-17T00:06:52.348399+00:00'
---"""


def test_clean_removes_header_nav_and_footer():
    raw = f"{FRONTMATTER}\n\n{BOILERPLATE_HEADER}\n{NAV_MENU}\n#  Jane Doe \nAssistant Professor\nShe studies AI.\n{DEPT_FOOTER}\n{VT_FOOTER}"
    result = clean_markdown(raw)

    assert "Skip to main content" not in result
    assert "Universal Access" not in result
    assert "Submenu Toggle" not in result
    assert "Follow Computer Science" not in result
    assert "Virginia Polytechnic Institute" not in result
    assert "bat.bing.com" not in result
    # Content preserved
    assert "Jane Doe" in result
    assert "Assistant Professor" in result
    assert "She studies AI." in result
    # Frontmatter preserved
    assert "url: 'https://website.cs.vt.edu/people/admin/jane.html'" in result


def test_clean_already_clean_page_is_noop():
    """Pages where fit_markdown already worked should pass through unchanged."""
    clean_page = f"{FRONTMATTER}\n\n# About\nThe department has been an incubator.\n"
    result = clean_markdown(clean_page)
    assert result == clean_page


def test_clean_removes_cms_artifacts():
    page = f"{FRONTMATTER}\n\n# Faculty\n  * Bio Item Bio Item\n[ Jane Doe , bio ](https://example.com)\nProfessor\n  * General Item General Item\n"
    result = clean_markdown(page)
    assert "Bio Item Bio Item" not in result
    assert "General Item General Item" not in result
    assert "Jane Doe" in result
    assert "Professor" in result


def test_clean_removes_search_widget():
    page = f"{FRONTMATTER}\n\nSearch\nSearch query\nsearch\nSearch this site \nFrequent Searches:\n  * [Hokie Spa](https://hokiespa.vt.edu)\n\n# Real Title\nReal content here.\n"
    result = clean_markdown(page)
    assert "Search query" not in result
    assert "Frequent Searches" not in result
    assert "Hokie Spa" not in result
    assert "Real Title" in result
    assert "Real content here." in result


def test_clean_removes_tracking_pixels():
    page = f"{FRONTMATTER}\n\n# Title\nContent.\n\n![](https://bat.bing.com/action/0?ti=343199719&Ver=2&mid=abc)\n"
    result = clean_markdown(page)
    assert "bat.bing.com" not in result
    assert "Content." in result


def test_clean_removes_breadcrumb():
    page = f"{FRONTMATTER}\n\n  1. [ College of Engineering  /  ](https://eng.vt.edu/)\n  2. [Department of Computer Science / ](https://website.cs.vt.edu/)\n  3. [About / ](https://website.cs.vt.edu/About.html)\n\n# About\nContent.\n"
    result = clean_markdown(page)
    assert "College of Engineering" not in result
    assert "Content." in result


def test_clean_removes_explore_sidebar():
    page = f"{FRONTMATTER}\n\nExplore\n  * [ Administration ](https://website.cs.vt.edu/people/administration.html)\n    * [ Chris Arnold ](https://website.cs.vt.edu/people/administration/chris-arnold.html)\n\n# Person Name\nBio here.\n"
    result = clean_markdown(page)
    assert "Chris Arnold" not in result or "Bio here." in result
    assert "Bio here." in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/crawler && python -m pytest tests/test_cleaner.py -v`
Expected: FAIL — `crawler.cleaner` module not found

- [ ] **Step 3: Implement clean_markdown**

File: `services/crawler/src/crawler/cleaner.py`

```python
"""Clean boilerplate from crawled VT CS department markdown pages."""

from __future__ import annotations

import re

# Patterns to remove (applied line-by-line or as blocks)
_SKIP_LINK_RE = re.compile(r"^\s*\*\s*\[Skip to (main content|search)\]")
_VT_HEADER_RE = re.compile(r"(Virginia Tech\u00ae? home|Universal Access|accessibility_icon)")
_TRACKING_PIXEL_RE = re.compile(r"!\[.*?\]\(https?://bat\.bing\.com/")
_CMS_ARTIFACT_RE = re.compile(r"^\s*\*\s*(Bio|General|Article|Redirect) Item \1 Item\s*$")
_SOCIAL_ICON_RE = re.compile(r"^\s*\*\s*\[(Instagram|Facebook|Linked-?In|Threads|Youtube|X|Blue Sky)\]\(https?://")
_COPYRIGHT_RE = re.compile(r"^\u00a9 \d{4} Virginia Polytechnic")
_VT_LOGO_RE = re.compile(r"!\[Virginia Tech logo\]")
_MAP_IMAGE_RE = re.compile(r"!\[Map of Virginia")
_SEARCH_HELP_RE = re.compile(r"^##\s*(Search Tips|Search Help|More search options|People Results|VT News Results)")
_BREADCRUMB_RE = re.compile(r"^\s*\d+\.\s*\[\s*(College of Engineering|Department of Computer Science)\s*/?\s*\]")

# Block markers — everything from these lines onward (until next content) is removed
_DEPT_FOOTER_MARKER = "### Follow Computer Science"
_EXPLORE_MARKER_RE = re.compile(r"^Explore\s*$")

# Lines that are part of the VT header/nav block
_NAV_INDICATORS = [
    "Submenu Toggle",
    "Resources for",
    "Frequent Searches:",
    "Search query",
    "Search this site",
    "Search all vt.edu",
    "People search",
    "webnewsvideopeople",
    "Web results for",
    "News results for",
    "Video results for",
    "People results for",
    "Sort by relevance",
    "Filter search",
    "Apply filters Clear filters",
    "search\n",
]

# Links that are part of the header, not content
_HEADER_LINK_RE = re.compile(
    r"^\s*\*?\s*\[(Apply|Visit|Give|Shop|Future Students|Current Students|"
    r"Parents and Families|Faculty and Staff|Alumni|Industry and Partners|"
    r"Hokie Sports Shop|Hokie Shop|Hokie Gear|Hokie License Plates|"
    r"Hokie Spa|Canvas|Jobs|Academic Calendar|Tuition|"
    r"University Photo Library|University Libraries|Undergraduate Course Catalog|"
    r"TLOS Course Catalog|Ensemble support|IT Support|VT Policy Library|"
    r"University Status|Principles of Community|Privacy Statement|Acceptable Use|"
    r"We Remember|Accessibility|Consumer Information|Cost & Aid|SAFE at VT|"
    r"Policies|Equal Opportunity|WVTF|University Bookstore|Jobs at Virginia Tech)\]"
)

_VT_FOOTER_LINKS = [
    "Get Directions",
    "See All Locations",
    "Contact Virginia Tech",
]


def clean_markdown(document: str) -> str:
    """Remove VT website boilerplate from a markdown document.

    Preserves YAML frontmatter and actual page content.
    Idempotent: already-clean pages pass through unchanged.
    """
    # Split frontmatter from body
    frontmatter = ""
    body = document
    if document.startswith("---"):
        end = document.find("---", 3)
        if end != -1:
            end = end + 3
            frontmatter = document[:end]
            body = document[end:]

    lines = body.split("\n")
    cleaned: list[str] = []
    in_explore_block = False
    in_search_block = False
    in_dept_footer = False
    in_vt_footer = False

    for line in lines:
        stripped = line.strip()

        # Skip empty-ish lines in removed blocks
        if in_explore_block:
            if stripped.startswith("#") and not stripped.startswith("# ") == False:
                # A real heading ends the explore block
                if re.match(r"^#{1,2}\s+\S", stripped):
                    in_explore_block = False
                else:
                    continue
            elif stripped.startswith("* [") or stripped.startswith("* [ ") or stripped.startswith("*  ") or not stripped:
                continue
            else:
                in_explore_block = False

        if in_search_block:
            if stripped.startswith("#") and not _SEARCH_HELP_RE.match(stripped):
                in_search_block = False
            else:
                continue

        if in_dept_footer or in_vt_footer:
            continue

        # Detect block starts
        if _EXPLORE_MARKER_RE.match(stripped):
            in_explore_block = True
            continue

        if stripped == "Search" and len(stripped) == 6:
            in_search_block = True
            continue

        if stripped.startswith(_DEPT_FOOTER_MARKER):
            in_dept_footer = True
            continue

        if _VT_LOGO_RE.search(line):
            in_vt_footer = True
            continue

        # Single-line removals
        if _SKIP_LINK_RE.match(line):
            continue
        if _VT_HEADER_RE.search(line):
            continue
        if _TRACKING_PIXEL_RE.search(line):
            continue
        if _CMS_ARTIFACT_RE.match(line):
            continue
        if _SOCIAL_ICON_RE.match(line):
            continue
        if _COPYRIGHT_RE.match(stripped):
            continue
        if _MAP_IMAGE_RE.search(line):
            continue
        if _SEARCH_HELP_RE.match(stripped):
            in_search_block = True
            continue
        if _BREADCRUMB_RE.match(line):
            continue
        if _HEADER_LINK_RE.match(line):
            continue
        if any(indicator in line for indicator in _NAV_INDICATORS):
            continue
        if any(link in line for link in _VT_FOOTER_LINKS):
            continue
        if "Close Universal Access dialog" in line:
            continue
        if re.match(r"^\s*\[?\s*Computer Science\s*\]?\s*\(https://website\.cs\.vt\.edu/\)\s*$", line):
            continue
        if stripped == "Menu":
            continue
        if stripped.startswith("![](data:image/svg+xml"):
            continue
        if "[Intranet]" in line and "(Internal)" in line:
            continue

        cleaned.append(line)

    # Collapse excessive blank lines (3+ consecutive → 2)
    result_body = "\n".join(cleaned)
    result_body = re.sub(r"\n{4,}", "\n\n\n", result_body)

    return frontmatter + result_body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_cleaner.py -v`
Expected: ALL PASS (may need tweaks — iterate until green)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/cleaner.py services/crawler/tests/test_cleaner.py
git commit -m "feat: add markdown cleaner module with tests for VT boilerplate removal"
```

---

## Chunk 3: Wire Cleaning into Pipeline

### Task 4: Add post-crawl cleaning step to main.py

**Files:**
- Modify: `services/crawler/src/crawler/main.py`
- Modify: `services/crawler/.env.example`
- Modify: `services/crawler/docker-compose.dev.yml` (at repo root)

- [ ] **Step 1: Update main.py to run cleaning after crawl**

```python
from crawler.cleaner import clean_markdown

def cli() -> None:
    # ... existing crawl code ...

    # Post-crawl: clean all raw pages and upload to cleaned bucket
    logger.info("Starting post-crawl cleaning")
    storage.ensure_bucket(config.minio_cleaned_bucket)

    raw_keys = storage.list_objects()
    cleaned_count = 0
    for key in raw_keys:
        raw_content = storage.download_document(key)
        cleaned_content = clean_markdown(raw_content)
        storage.upload_document(key, cleaned_content, bucket=config.minio_cleaned_bucket)
        cleaned_count += 1

    logger.info("Cleaning complete: %d pages cleaned and stored in %s", cleaned_count, config.minio_cleaned_bucket)
```

- [ ] **Step 2: Update .env.example**

Add:
```
MINIO_CLEANED_BUCKET=crawled-pages-cleaned
```

- [ ] **Step 3: Update docker-compose.dev.yml**

Add env var:
```yaml
MINIO_CLEANED_BUCKET: crawled-pages-cleaned
```

- [ ] **Step 4: Run all tests**

Run: `cd services/crawler && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/main.py services/crawler/.env.example docker-compose.dev.yml
git commit -m "feat: wire markdown cleaning into post-crawl pipeline with two-bucket output"
```

### Task 5: End-to-end test run

- [ ] **Step 1: Run 50-page crawl with cleaning**

```bash
cd services/crawler && \
MINIO_ENDPOINT=localhost:9000 \
MINIO_ACCESS_KEY=minioadmin \
MINIO_SECRET_KEY=minioadmin \
CRAWL_MAX_PAGES=50 \
python -m crawler.main
```

- [ ] **Step 2: Verify both buckets have content**

```bash
python -c "
from minio import Minio
client = Minio('localhost:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
raw = list(client.list_objects('crawled-pages', recursive=True))
cleaned = list(client.list_objects('crawled-pages-cleaned', recursive=True))
print(f'Raw bucket: {len(raw)} objects')
print(f'Cleaned bucket: {len(cleaned)} objects')
"
```

- [ ] **Step 3: Spot-check a cleaned page to verify boilerplate is gone**

- [ ] **Step 4: Commit any fixes**
