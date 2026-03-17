# HokieHelp Phase 1: Web Crawler & Object Storage Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kubernetes-deployable crawling service that deep-crawls cs.vt.edu, extracts clean Markdown, and stores it in MinIO object storage.

**Architecture:** A Python-based crawler service using Crawl4AI's BFS deep crawling strategy with domain filtering. The crawler runs as a batch job (K8s CronJob), streams results as they arrive, and uploads each page as a Markdown document with YAML frontmatter metadata to MinIO. Configuration is driven by environment variables for K8s compatibility.

**Tech Stack:** Python 3.11+, Crawl4AI (AsyncWebCrawler, BFSDeepCrawlStrategy), MinIO Python SDK, pytest, Docker, Kubernetes

---

## File Structure

```
services/
  crawler/
    src/
      crawler/
        __init__.py              # Package init
        config.py                # Configuration from env vars
        storage.py               # MinIO upload client
        crawl.py                 # Deep crawl orchestration
        markdown_doc.py          # Markdown document builder with metadata
        main.py                  # Entrypoint: wire config → crawl → store
    tests/
      __init__.py
      conftest.py                # Shared fixtures (fake MinIO, mock crawl results)
      test_config.py             # Config loading tests
      test_storage.py            # MinIO storage tests
      test_markdown_doc.py       # Markdown document building tests
      test_crawl.py              # Crawl orchestration tests
      test_main.py               # Integration/entrypoint tests
    pyproject.toml               # Project metadata and dependencies
    Dockerfile                   # Container image for K8s deployment
    .env.example                 # Example environment variables

infrastructure/
  kubernetes/
    crawler-job.yaml             # K8s CronJob manifest
    minio.yaml                   # MinIO deployment manifest (dev)
```

### Responsibilities

| File | Responsibility |
|------|---------------|
| `config.py` | Load and validate all settings from environment variables |
| `storage.py` | Connect to MinIO, ensure bucket exists, upload markdown documents |
| `markdown_doc.py` | Build a Markdown string with YAML frontmatter from crawl result metadata |
| `crawl.py` | Configure Crawl4AI BFS deep crawl, stream results, call storage for each page |
| `main.py` | Parse CLI args, create config, run crawl, handle graceful shutdown |

---

## Chunk 1: Project Scaffolding and Configuration

### Task 1: Project Setup

**Files:**
- Create: `services/crawler/pyproject.toml`
- Create: `services/crawler/src/crawler/__init__.py`
- Create: `services/crawler/tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "hokiehelp-crawler"
version = "0.1.0"
description = "Web crawler for cs.vt.edu that stores Markdown in MinIO"
requires-python = ">=3.11"
dependencies = [
    "crawl4ai>=0.6.0",
    "minio>=7.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
hokiehelp-crawl = "crawler.main:cli"
```

- [ ] **Step 2: Create package init files**

`services/crawler/src/crawler/__init__.py`:
```python
"""HokieHelp web crawler service."""
```

`services/crawler/tests/__init__.py`:
```python
```

- [ ] **Step 3: Install the project in dev mode**

Run: `cd services/crawler && pip install -e ".[dev]"`
Expected: Successful installation with crawl4ai and minio

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `cd services/crawler && python -m pytest tests/ -v`
Expected: "no tests ran" with exit code 5 (no tests collected)

- [ ] **Step 5: Commit**

```bash
git add services/crawler/pyproject.toml services/crawler/src/crawler/__init__.py services/crawler/tests/__init__.py
git commit -m "feat: scaffold crawler service with pyproject.toml"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `services/crawler/src/crawler/config.py`
- Create: `services/crawler/tests/test_config.py`
- Create: `services/crawler/.env.example`

- [ ] **Step 1: Write failing tests for config loading**

`services/crawler/tests/test_config.py`:
```python
import os
import pytest
from crawler.config import CrawlerConfig


def test_config_from_env_defaults(monkeypatch):
    """Config loads with sensible defaults when only required vars are set."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")

    config = CrawlerConfig.from_env()

    assert config.minio_endpoint == "localhost:9000"
    assert config.minio_access_key == "minioadmin"
    assert config.minio_secret_key == "minioadmin"
    assert config.minio_bucket == "crawled-pages"
    assert config.minio_secure is False
    assert config.seed_url == "https://cs.vt.edu"
    assert config.max_depth == 2
    assert config.max_pages == 500
    assert config.allowed_domain == "cs.vt.edu"


def test_config_from_env_custom(monkeypatch):
    """Config respects custom environment variable overrides."""
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.prod:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "prodkey")
    monkeypatch.setenv("MINIO_SECRET_KEY", "prodsecret")
    monkeypatch.setenv("MINIO_BUCKET", "my-bucket")
    monkeypatch.setenv("MINIO_SECURE", "true")
    monkeypatch.setenv("CRAWL_SEED_URL", "https://cs.vt.edu/academics")
    monkeypatch.setenv("CRAWL_MAX_DEPTH", "5")
    monkeypatch.setenv("CRAWL_MAX_PAGES", "1000")

    config = CrawlerConfig.from_env()

    assert config.minio_endpoint == "minio.prod:9000"
    assert config.minio_bucket == "my-bucket"
    assert config.minio_secure is True
    assert config.seed_url == "https://cs.vt.edu/academics"
    assert config.max_depth == 5
    assert config.max_pages == 1000


def test_config_missing_required_var(monkeypatch):
    """Config raises ValueError when required vars are missing."""
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        CrawlerConfig.from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/crawler && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement config.py**

`services/crawler/src/crawler/config.py`:
```python
"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CrawlerConfig:
    """Immutable crawler configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool
    seed_url: str
    max_depth: int
    max_pages: int
    allowed_domain: str

    @classmethod
    def from_env(cls) -> CrawlerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_BUCKET, MINIO_SECURE, CRAWL_SEED_URL,
                  CRAWL_MAX_DEPTH, CRAWL_MAX_PAGES
        """

        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise ValueError(f"Required environment variable {name} is not set")
            return val

        return cls(
            minio_endpoint=_require("MINIO_ENDPOINT"),
            minio_access_key=_require("MINIO_ACCESS_KEY"),
            minio_secret_key=_require("MINIO_SECRET_KEY"),
            minio_bucket=os.environ.get("MINIO_BUCKET", "crawled-pages"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            seed_url=os.environ.get("CRAWL_SEED_URL", "https://cs.vt.edu"),
            max_depth=int(os.environ.get("CRAWL_MAX_DEPTH", "2")),
            max_pages=int(os.environ.get("CRAWL_MAX_PAGES", "500")),
            allowed_domain=os.environ.get("CRAWL_ALLOWED_DOMAIN", "cs.vt.edu"),
        )
```

- [ ] **Step 4: Create .env.example**

`services/crawler/.env.example`:
```bash
# MinIO connection
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=crawled-pages
MINIO_SECURE=false

# Crawler settings
CRAWL_SEED_URL=https://cs.vt.edu
CRAWL_MAX_DEPTH=2
CRAWL_MAX_PAGES=500
CRAWL_ALLOWED_DOMAIN=cs.vt.edu
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_config.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/config.py services/crawler/tests/test_config.py services/crawler/.env.example
git commit -m "feat: add crawler configuration module with env var loading"
```

---

## Chunk 2: Markdown Document Builder and MinIO Storage

### Task 3: Markdown Document Builder

**Files:**
- Create: `services/crawler/src/crawler/markdown_doc.py`
- Create: `services/crawler/tests/test_markdown_doc.py`

- [ ] **Step 1: Write failing tests for markdown document building**

`services/crawler/tests/test_markdown_doc.py`:
```python
import pytest
from datetime import datetime, timezone
from crawler.markdown_doc import build_markdown_document, url_to_object_key


def test_build_markdown_document():
    """Builds a markdown string with YAML frontmatter from crawl data."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/academics/courses.html",
        title="CS Courses",
        markdown_content="# Courses\n\nHere are our courses.",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert doc.startswith("---\n")
    assert "url: 'https://cs.vt.edu/academics/courses.html'" in doc
    assert "title: 'CS Courses'" in doc
    assert "crawl_depth: 1" in doc
    assert "crawl_timestamp: '2026-03-16T12:00:00+00:00'" in doc
    assert "---\n\n# Courses\n\nHere are our courses." in doc


def test_build_markdown_document_no_title():
    """Uses URL as fallback when title is missing."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/about",
        title=None,
        markdown_content="About page content.",
        crawl_depth=0,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "title: 'https://cs.vt.edu/about'" in doc


def test_build_markdown_document_special_chars():
    """Handles titles with YAML-special characters like colons."""
    doc = build_markdown_document(
        url="https://cs.vt.edu/courses/cs101",
        title="CS 101: Intro to CS",
        markdown_content="Course content.",
        crawl_depth=1,
        crawl_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "title: 'CS 101: Intro to CS'" in doc


def test_url_to_object_key_basic():
    """Converts a URL to a MinIO object key."""
    key = url_to_object_key("https://cs.vt.edu/academics/courses.html")
    assert key == "cs.vt.edu/academics/courses.html.md"


def test_url_to_object_key_trailing_slash():
    """Trailing slash is stripped so /about and /about/ map to the same key."""
    key = url_to_object_key("https://cs.vt.edu/academics/")
    assert key == "cs.vt.edu/academics.md"


def test_url_to_object_key_root():
    """Handles root URL."""
    key = url_to_object_key("https://cs.vt.edu")
    assert key == "cs.vt.edu/index.md"


def test_url_to_object_key_strips_query_and_fragment():
    """Strips query string and fragment from URL."""
    key = url_to_object_key("https://cs.vt.edu/page?foo=bar#section")
    assert key == "cs.vt.edu/page.md"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/crawler && python -m pytest tests/test_markdown_doc.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement markdown_doc.py**

`services/crawler/src/crawler/markdown_doc.py`:
```python
"""Build Markdown documents with YAML frontmatter from crawl results."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse


def build_markdown_document(
    *,
    url: str,
    title: str | None,
    markdown_content: str,
    crawl_depth: int,
    crawl_timestamp: datetime,
) -> str:
    """Return a Markdown string with YAML frontmatter metadata."""
    effective_title = title if title else url
    ts = crawl_timestamp.isoformat()

    frontmatter = (
        f"---\n"
        f"url: '{url}'\n"
        f"title: '{effective_title}'\n"
        f"crawl_depth: {crawl_depth}\n"
        f"crawl_timestamp: '{ts}'\n"
        f"---"
    )
    return f"{frontmatter}\n\n{markdown_content}"


def url_to_object_key(url: str) -> str:
    """Convert a URL to a structured MinIO object key.

    Example: https://cs.vt.edu/academics/courses.html -> cs.vt.edu/academics/courses.html.md
    """
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc
    path = parsed.path

    # Strip leading and trailing slashes to normalize /about and /about/
    path = path.strip("/")

    # Handle root URL (empty path)
    if not path:
        path = "index"

    # Add .md extension if not already a .md file
    if not path.endswith(".md"):
        path = path + ".md"

    return f"{host}/{path}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_markdown_doc.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/markdown_doc.py services/crawler/tests/test_markdown_doc.py
git commit -m "feat: add markdown document builder with YAML frontmatter"
```

---

### Task 4: MinIO Storage Client

**Files:**
- Create: `services/crawler/src/crawler/storage.py`
- Create: `services/crawler/tests/test_storage.py`
- Create: `services/crawler/tests/conftest.py`

- [ ] **Step 1: Write failing tests for storage**

`services/crawler/tests/conftest.py`:
```python
"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock
from crawler.config import CrawlerConfig


@pytest.fixture
def crawler_config():
    """Minimal crawler config for testing."""
    return CrawlerConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="test-bucket",
        minio_secure=False,
        seed_url="https://cs.vt.edu",
        max_depth=2,
        max_pages=100,
        allowed_domain="cs.vt.edu",
    )


@pytest.fixture
def mock_minio_client():
    """A MagicMock standing in for minio.Minio."""
    client = MagicMock()
    client.bucket_exists.return_value = True
    client.put_object.return_value = MagicMock(
        object_name="cs.vt.edu/index.md", etag="abc123"
    )
    return client
```

`services/crawler/tests/test_storage.py`:
```python
import io
import pytest
from unittest.mock import MagicMock, patch, call

from crawler.storage import MinioStorage


def test_init_creates_bucket_if_missing(crawler_config, mock_minio_client):
    """Storage creates the bucket when it doesn't exist."""
    mock_minio_client.bucket_exists.return_value = False

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)

    mock_minio_client.make_bucket.assert_called_once_with("test-bucket")


def test_init_skips_bucket_creation_if_exists(crawler_config, mock_minio_client):
    """Storage does not recreate an existing bucket."""
    mock_minio_client.bucket_exists.return_value = True

    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)

    mock_minio_client.make_bucket.assert_not_called()


def test_upload_document(crawler_config, mock_minio_client):
    """Uploads markdown content to the correct object key."""
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_document("cs.vt.edu/index.md", "# Hello")

    mock_minio_client.put_object.assert_called_once()
    args, kwargs = mock_minio_client.put_object.call_args
    assert args[0] == "test-bucket"
    assert args[1] == "cs.vt.edu/index.md"
    assert kwargs["content_type"] == "text/markdown"


def test_upload_document_content_bytes(crawler_config, mock_minio_client):
    """Uploaded data matches the input content."""
    with patch("crawler.storage.Minio", return_value=mock_minio_client):
        storage = MinioStorage(crawler_config)
        storage.upload_document("cs.vt.edu/page.md", "# Test Content")

    args, kwargs = mock_minio_client.put_object.call_args
    data_stream = args[2]
    data_length = args[3]
    content = data_stream.read()
    assert content == b"# Test Content"
    assert data_length == len(b"# Test Content")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/crawler && python -m pytest tests/test_storage.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement storage.py**

`services/crawler/src/crawler/storage.py`:
```python
"""MinIO object storage client for uploading crawled documents."""

from __future__ import annotations

import io
import logging

from minio import Minio

from crawler.config import CrawlerConfig

logger = logging.getLogger(__name__)


class MinioStorage:
    """Uploads Markdown documents to MinIO."""

    def __init__(self, config: CrawlerConfig) -> None:
        self._bucket = config.minio_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            logger.info("Created bucket %s", self._bucket)

    def upload_document(self, object_key: str, content: str) -> None:
        """Upload a Markdown document to MinIO.

        Args:
            object_key: The object path within the bucket (e.g. cs.vt.edu/index.md)
            content: The full Markdown document string
        """
        data = content.encode("utf-8")
        self._client.put_object(
            self._bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type="text/markdown",
        )
        logger.info("Uploaded %s (%d bytes)", object_key, len(data))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_storage.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/storage.py services/crawler/tests/test_storage.py services/crawler/tests/conftest.py
git commit -m "feat: add MinIO storage client with bucket management"
```

---

## Chunk 3: Crawl Orchestration and Entrypoint

### Task 5: Crawl Orchestration

**Files:**
- Create: `services/crawler/src/crawler/crawl.py`
- Create: `services/crawler/tests/test_crawl.py`

- [ ] **Step 1: Write failing tests for crawl orchestration**

`services/crawler/tests/test_crawl.py`:
```python
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from crawler.crawl import run_crawl


def _make_crawl_result(url: str, title: str, markdown: str, depth: int = 0):
    """Create a mock CrawlResult."""
    result = MagicMock()
    result.success = True
    result.url = url
    result.metadata = {"title": title, "depth": depth}
    result.markdown = MagicMock()
    result.markdown.raw_markdown = markdown
    return result


@pytest.mark.asyncio
async def test_run_crawl_processes_results(crawler_config):
    """run_crawl uploads each successful crawl result to storage."""
    mock_storage = MagicMock()
    results = [
        _make_crawl_result("https://cs.vt.edu", "Home", "# Home Page", depth=0),
        _make_crawl_result("https://cs.vt.edu/about", "About", "# About", depth=1),
    ]

    # Mock AsyncWebCrawler as async context manager returning async iterator
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

    assert mock_storage.upload_document.call_count == 2
    assert stats["pages_crawled"] == 2
    assert stats["pages_failed"] == 0


@pytest.mark.asyncio
async def test_run_crawl_skips_failed_results(crawler_config):
    """run_crawl skips results where success is False."""
    mock_storage = MagicMock()

    failed_result = MagicMock()
    failed_result.success = False
    failed_result.url = "https://cs.vt.edu/broken"
    failed_result.error_message = "404 Not Found"

    ok_result = _make_crawl_result("https://cs.vt.edu", "Home", "# Home", depth=0)

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield failed_result
            yield ok_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 1
    assert stats["pages_crawled"] == 1
    assert stats["pages_failed"] == 1


@pytest.mark.asyncio
async def test_run_crawl_skips_empty_markdown(crawler_config):
    """run_crawl skips results with empty markdown content."""
    mock_storage = MagicMock()

    empty_result = _make_crawl_result("https://cs.vt.edu/empty", "Empty", "", depth=0)

    mock_crawler_instance = AsyncMock()

    async def fake_arun(*args, **kwargs):
        async def _gen():
            yield empty_result
        return _gen()

    mock_crawler_instance.arun = fake_arun
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("crawler.crawl.AsyncWebCrawler", return_value=mock_crawler_instance):
        stats = await run_crawl(crawler_config, mock_storage)

    assert mock_storage.upload_document.call_count == 0
    assert stats["pages_crawled"] == 0
    assert stats["pages_failed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/crawler && python -m pytest tests/test_crawl.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement crawl.py**

`services/crawler/src/crawler/crawl.py`:
```python
"""Deep crawl orchestration using Crawl4AI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, ContentTypeFilter

from crawler.config import CrawlerConfig
from crawler.markdown_doc import build_markdown_document, url_to_object_key
from crawler.storage import MinioStorage

logger = logging.getLogger(__name__)


async def run_crawl(config: CrawlerConfig, storage: MinioStorage) -> dict:
    """Execute a deep crawl and upload each page to storage.

    Returns a stats dict with pages_crawled and pages_failed counts.
    """
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=[config.allowed_domain],
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=config.max_depth,
        include_external=False,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=True,
    )

    stats = {"pages_crawled": 0, "pages_failed": 0}

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            url=config.seed_url, config=run_config
        ):
            if not result.success:
                logger.warning(
                    "Failed to crawl %s: %s",
                    result.url,
                    getattr(result, "error_message", "unknown error"),
                )
                stats["pages_failed"] += 1
                continue

            depth = result.metadata.get("depth", 0)
            title = result.metadata.get("title")
            now = datetime.now(timezone.utc)

            markdown_content = result.markdown.raw_markdown
            if not markdown_content:
                logger.warning("Empty markdown for %s, skipping", result.url)
                stats["pages_failed"] += 1
                continue

            document = build_markdown_document(
                url=result.url,
                title=title,
                markdown_content=markdown_content,
                crawl_depth=depth,
                crawl_timestamp=now,
            )

            object_key = url_to_object_key(result.url)
            storage.upload_document(object_key, document)
            stats["pages_crawled"] += 1
            logger.info(
                "Stored page %d: %s (depth %d)",
                stats["pages_crawled"],
                result.url,
                depth,
            )

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_crawl.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/crawler/src/crawler/crawl.py services/crawler/tests/test_crawl.py
git commit -m "feat: add deep crawl orchestration with BFS strategy"
```

---

### Task 6: Main Entrypoint

**Files:**
- Create: `services/crawler/src/crawler/main.py`
- Create: `services/crawler/tests/test_main.py`

- [ ] **Step 1: Write failing tests for main entrypoint**

`services/crawler/tests/test_main.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from crawler.main import cli


def test_cli_runs_crawl(monkeypatch):
    """CLI loads config, creates storage, runs crawl, and exits."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")

    mock_storage_cls = MagicMock()
    mock_run_crawl = AsyncMock(return_value={"pages_crawled": 5, "pages_failed": 1})

    with patch("crawler.main.MinioStorage", mock_storage_cls), \
         patch("crawler.main.run_crawl", mock_run_crawl):
        cli()

    mock_storage_cls.assert_called_once()
    mock_run_crawl.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/crawler && python -m pytest tests/test_main.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement main.py**

`services/crawler/src/crawler/main.py`:
```python
"""Crawler entrypoint — wire config, storage, and crawl together."""

from __future__ import annotations

import asyncio
import logging
import sys

from crawler.config import CrawlerConfig
from crawler.crawl import run_crawl
from crawler.storage import MinioStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cli() -> None:
    """CLI entrypoint for the crawler."""
    logger.info("Starting HokieHelp crawler")

    try:
        config = CrawlerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    logger.info(
        "Crawling %s (max_depth=%d, max_pages=%d)",
        config.seed_url,
        config.max_depth,
        config.max_pages,
    )

    storage = MinioStorage(config)
    stats = asyncio.run(run_crawl(config, storage))

    logger.info(
        "Crawl complete: %d pages stored, %d failed",
        stats["pages_crawled"],
        stats["pages_failed"],
    )


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/crawler && python -m pytest tests/test_main.py -v`
Expected: 1 test PASS

- [ ] **Step 5: Run full test suite**

Run: `cd services/crawler && python -m pytest tests/ -v`
Expected: All tests PASS (18 total)

- [ ] **Step 6: Commit**

```bash
git add services/crawler/src/crawler/main.py services/crawler/tests/test_main.py
git commit -m "feat: add crawler CLI entrypoint"
```

---

## Chunk 4: Docker and Kubernetes Deployment

### Task 7: Dockerfile

**Files:**
- Create: `services/crawler/Dockerfile`

- [ ] **Step 1: Create the Dockerfile**

`services/crawler/Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Crawl4AI needs a browser for rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files and install
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Crawl4AI post-install: download browser
RUN crawl4ai-setup

# Run the crawler
CMD ["python", "-m", "crawler.main"]
```

- [ ] **Step 2: Verify Dockerfile syntax**

Run: `cd services/crawler && docker build --check .` (or `docker build -t hokiehelp-crawler . --dry-run` — just verify no syntax errors)

If `--check` is not supported, just visually verify the Dockerfile is valid.

- [ ] **Step 3: Commit**

```bash
git add services/crawler/Dockerfile
git commit -m "feat: add Dockerfile for crawler service"
```

---

### Task 8: Kubernetes Manifests

**Files:**
- Create: `infrastructure/kubernetes/minio.yaml`
- Create: `infrastructure/kubernetes/crawler-job.yaml`

- [ ] **Step 1: Create MinIO development deployment manifest**

`infrastructure/kubernetes/minio.yaml`:
```yaml
# MinIO deployment for local/dev use
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  labels:
    app: minio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
        - name: minio
          image: minio/minio:latest
          args: ["server", "/data", "--console-address", ":9001"]
          ports:
            - containerPort: 9000
              name: api
            - containerPort: 9001
              name: console
          env:
            - name: MINIO_ROOT_USER
              value: "minioadmin"
            - name: MINIO_ROOT_PASSWORD
              value: "minioadmin"
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: minio
spec:
  selector:
    app: minio
  ports:
    - port: 9000
      targetPort: 9000
      name: api
    - port: 9001
      targetPort: 9001
      name: console
```

- [ ] **Step 2: Create crawler CronJob manifest**

`infrastructure/kubernetes/crawler-job.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hokiehelp-crawler
  labels:
    app: hokiehelp-crawler
spec:
  schedule: "0 3 * * 0"  # Weekly on Sunday at 3am
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600  # 1 hour timeout
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: crawler
              image: hokiehelp-crawler:latest
              env:
                - name: MINIO_ENDPOINT
                  value: "minio:9000"
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
                - name: MINIO_BUCKET
                  value: "crawled-pages"
                - name: CRAWL_SEED_URL
                  value: "https://cs.vt.edu"
                - name: CRAWL_MAX_DEPTH
                  value: "2"
                - name: CRAWL_MAX_PAGES
                  value: "500"
              resources:
                requests:
                  memory: "512Mi"
                  cpu: "250m"
                limits:
                  memory: "2Gi"
                  cpu: "1000m"
```

- [ ] **Step 3: Note — create the K8s Secret before running the crawler job**

The crawler-job.yaml references a Secret `minio-credentials`. Create it with:
```bash
kubectl create secret generic minio-credentials \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin
```

- [ ] **Step 4: Commit**

```bash
git add infrastructure/kubernetes/minio.yaml infrastructure/kubernetes/crawler-job.yaml
git commit -m "feat: add Kubernetes manifests for MinIO and crawler CronJob"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Run full test suite one last time**

Run: `cd services/crawler && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify project structure**

Run: `find services/ infrastructure/ -type f | sort`
Expected output:
```
infrastructure/kubernetes/crawler-job.yaml
infrastructure/kubernetes/minio.yaml
services/crawler/.env.example
services/crawler/Dockerfile
services/crawler/pyproject.toml
services/crawler/src/crawler/__init__.py
services/crawler/src/crawler/config.py
services/crawler/src/crawler/crawl.py
services/crawler/src/crawler/main.py
services/crawler/src/crawler/markdown_doc.py
services/crawler/src/crawler/storage.py
services/crawler/tests/__init__.py
services/crawler/tests/conftest.py
services/crawler/tests/test_config.py
services/crawler/tests/test_crawl.py
services/crawler/tests/test_main.py
services/crawler/tests/test_markdown_doc.py
services/crawler/tests/test_storage.py
```

- [ ] **Step 3: Final commit for any remaining files**

```bash
git add -A
git status
# If anything unstaged, commit it:
git commit -m "chore: complete Phase 1 crawler service scaffolding"
```
