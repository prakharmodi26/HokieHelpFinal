# Chunker Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `services/chunker` microservice that reads cleaned markdown + metadata from MinIO, splits documents into semantically-aware chunks, and stores chunk records back to object storage for downstream embedding/indexing.

**Architecture:** The chunker reads `.md` files from the `crawled-pages-cleaned` bucket (written by the crawler), parses YAML frontmatter and markdown structure, applies heading-first + token-size chunking rules, and writes per-document chunk arrays to a new `chunks` bucket as `chunks/<doc_id>.json`. It runs as a weekly Kubernetes CronJob (Sunday 4 AM, one hour after the crawler) sharing the same MinIO credentials Secret. No external token-counting library — token count is approximated as `len(text) // 4` (4 chars ≈ 1 token for English).

**Tech Stack:** Python 3.11, `minio>=7.2.0`, `pytest>=8.0`, Docker (`python:3.11-slim`), Kubernetes CronJob, GitHub Actions CI/CD (GHCR push)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/chunker/pyproject.toml` | CREATE | Package metadata, `minio` dep, `hokiehelp-chunk` CLI entrypoint |
| `services/chunker/Dockerfile` | CREATE | `python:3.11-slim`, no browser needed |
| `services/chunker/src/chunker/__init__.py` | CREATE | Empty package marker |
| `services/chunker/src/chunker/config.py` | CREATE | `ChunkerConfig` frozen dataclass, `from_env()` |
| `services/chunker/src/chunker/models.py` | CREATE | `ChunkRecord` dataclass, `to_dict`/`from_dict` |
| `services/chunker/src/chunker/parser.py` | CREATE | Frontmatter parser, section splitter, `infer_page_type` |
| `services/chunker/src/chunker/splitter.py` | CREATE | Token estimator, merge-small, split-large-with-overlap |
| `services/chunker/src/chunker/storage.py` | CREATE | `ChunkerStorage`: list, download, upload chunks |
| `services/chunker/src/chunker/main.py` | CREATE | `cli()` entrypoint wiring all modules |
| `services/chunker/tests/__init__.py` | CREATE | Empty |
| `services/chunker/tests/conftest.py` | CREATE | Shared fixtures |
| `services/chunker/tests/test_config.py` | CREATE | Config loading + validation |
| `services/chunker/tests/test_models.py` | CREATE | ChunkRecord serialisation round-trip |
| `services/chunker/tests/test_parser.py` | CREATE | Frontmatter parsing, section splitting, page_type |
| `services/chunker/tests/test_splitter.py` | CREATE | Token estimation, merge, split-with-overlap |
| `services/chunker/tests/test_storage.py` | CREATE | Storage using MagicMock Minio |
| `services/chunker/tests/test_main.py` | CREATE | End-to-end orchestration with mocked storage |
| `k8s/chunker-configmap.yaml` | CREATE | Non-secret env vars for chunker |
| `k8s/chunker-cronjob.yaml` | CREATE | Weekly CronJob, same Secret as crawler |
| `.github/workflows/chunker-ci.yaml` | CREATE | Test on PR, build+push on main |

---

## Object Storage Contract

**Input** (reads from `crawled-pages-cleaned` bucket):
```
website.cs.vt.edu/about.md          ← cleaned markdown with YAML frontmatter
website.cs.vt.edu/about.meta.json   ← sidecar (not read by chunker — frontmatter is enough)
_department-info.md                 ← synthetic doc, may have incomplete frontmatter
```

**Output** (writes to `chunks` bucket):
```
chunks/<doc_id>.json   ← JSON array of ChunkRecord objects for that document
```

**ChunkRecord schema:**
```json
{
  "chunk_id": "a1b2c3d4e5f6a7b8_0000",
  "document_id": "a1b2c3d4e5f6a7b8",
  "chunk_index": 0,
  "text": "## Faculty\n\nThe CS department has...",
  "url": "https://website.cs.vt.edu/people/faculty",
  "title": "Faculty | CS | VT",
  "page_type": "faculty",
  "headings_path": ["People", "Faculty"],
  "content_hash": "3f7a9c12d4e8b001",
  "crawl_timestamp": "2026-03-16T03:00:00+00:00"
}
```

---

## Chunking Logic

1. **Strip frontmatter** — extract `doc_id`, `url`, `title`, `content_hash`, `crawl_timestamp` from YAML block.
2. **Split by headings** — scan body for `^#+ ` lines. Each heading starts a new section. Track heading hierarchy for `headings_path`.
3. **Apply size rules** per section (token estimate = `len(text) // 4`):
   - Section < 120 tokens → accumulate with next section (forward merge).
   - Section in [120, 400] tokens → emit as one chunk.
   - Section > 400 tokens → split into paragraph-boundary windows (≤400 tokens each) with 64-token character overlap between adjacent windows.
4. **Assign IDs** — `chunk_id = f"{doc_id}_{chunk_index:04d}"`.
5. **headings_path** for a merged chunk = path of the *first* constituent section.

---

## Task 1: Service scaffold

**Files:**
- Create: `services/chunker/pyproject.toml`
- Create: `services/chunker/src/chunker/__init__.py`
- Create: `services/chunker/tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p services/chunker/src/chunker
mkdir -p services/chunker/tests
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "hokiehelp-chunker"
version = "0.1.0"
description = "Markdown chunker for HokieHelp RAG pipeline"
requires-python = ">=3.11"
dependencies = [
    "minio>=7.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[project.scripts]
hokiehelp-chunk = "chunker.main:cli"
```

- [ ] **Step 3: Create empty `__init__.py` files**

`services/chunker/src/chunker/__init__.py` and `services/chunker/tests/__init__.py` — both empty.

- [ ] **Step 4: Verify package installs**

```bash
cd services/chunker && pip install -e ".[dev]"
```

Expected: `Successfully installed hokiehelp-chunker-0.1.0`

- [ ] **Step 5: Commit**

```bash
git add services/chunker/pyproject.toml services/chunker/src/chunker/__init__.py services/chunker/tests/__init__.py
git commit -m "feat: scaffold chunker service package"
```

---

## Task 2: Config module

**Files:**
- Create: `services/chunker/src/chunker/config.py`
- Create: `services/chunker/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

File: `services/chunker/tests/test_config.py`

```python
"""Tests for ChunkerConfig."""
import os
import pytest
from chunker.config import ChunkerConfig


def _base_env():
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
    }


def test_from_env_loads_required(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    assert cfg.minio_endpoint == "localhost:9000"
    assert cfg.minio_access_key == "minioadmin"
    assert cfg.minio_secret_key == "minioadmin"


def test_from_env_defaults(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    assert cfg.minio_cleaned_bucket == "crawled-pages-cleaned"
    assert cfg.minio_chunks_bucket == "chunks"
    assert cfg.minio_secure is False
    assert cfg.chunk_preferred_tokens == 400
    assert cfg.chunk_overlap_tokens == 64
    assert cfg.chunk_min_tokens == 120


def test_from_env_overrides(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHUNK_PREFERRED_TOKENS", "200")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "32")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "60")
    cfg = ChunkerConfig.from_env()
    assert cfg.chunk_preferred_tokens == 200
    assert cfg.chunk_overlap_tokens == 32
    assert cfg.chunk_min_tokens == 60


def test_from_env_missing_required_raises(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        ChunkerConfig.from_env()


def test_config_is_frozen(monkeypatch):
    for k, v in _base_env().items():
        monkeypatch.setenv(k, v)
    cfg = ChunkerConfig.from_env()
    with pytest.raises((AttributeError, TypeError)):
        cfg.minio_endpoint = "other"  # type: ignore
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker.config'`

- [ ] **Step 3: Implement `config.py`**

```python
"""Configuration loaded from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkerConfig:
    """Immutable chunker configuration."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_cleaned_bucket: str
    minio_chunks_bucket: str
    chunk_preferred_tokens: int
    chunk_overlap_tokens: int
    chunk_min_tokens: int

    @classmethod
    def from_env(cls) -> ChunkerConfig:
        """Load configuration from environment variables.

        Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        Optional: MINIO_SECURE, MINIO_CLEANED_BUCKET, MINIO_CHUNKS_BUCKET,
                  CHUNK_PREFERRED_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS
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
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            minio_cleaned_bucket=os.environ.get("MINIO_CLEANED_BUCKET", "crawled-pages-cleaned"),
            minio_chunks_bucket=os.environ.get("MINIO_CHUNKS_BUCKET", "chunks"),
            chunk_preferred_tokens=int(os.environ.get("CHUNK_PREFERRED_TOKENS", "400")),
            chunk_overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "64")),
            chunk_min_tokens=int(os.environ.get("CHUNK_MIN_TOKENS", "120")),
        )
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/chunker && pytest tests/test_config.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/config.py services/chunker/tests/test_config.py
git commit -m "feat: add chunker config module"
```

---

## Task 3: ChunkRecord model

**Files:**
- Create: `services/chunker/src/chunker/models.py`
- Create: `services/chunker/tests/test_models.py`

- [ ] **Step 1: Write failing tests**

File: `services/chunker/tests/test_models.py`

```python
"""Tests for ChunkRecord model."""
import pytest
from chunker.models import ChunkRecord


def _sample_chunk(**overrides) -> ChunkRecord:
    defaults = dict(
        chunk_id="abc1234567890abc_0000",
        document_id="abc1234567890abc",
        chunk_index=0,
        text="## Faculty\n\nThe CS department has world-class faculty.",
        url="https://website.cs.vt.edu/people/faculty",
        title="Faculty | CS | VT",
        page_type="faculty",
        headings_path=["People", "Faculty"],
        content_hash="deadbeef01234567",
        crawl_timestamp="2026-03-16T03:00:00+00:00",
    )
    defaults.update(overrides)
    return ChunkRecord(**defaults)


def test_chunk_record_fields():
    c = _sample_chunk()
    assert c.chunk_id == "abc1234567890abc_0000"
    assert c.chunk_index == 0
    assert c.headings_path == ["People", "Faculty"]


def test_to_dict_has_all_keys():
    d = _sample_chunk().to_dict()
    for key in (
        "chunk_id", "document_id", "chunk_index", "text", "url", "title",
        "page_type", "headings_path", "content_hash", "crawl_timestamp",
    ):
        assert key in d


def test_to_dict_values_match():
    c = _sample_chunk()
    d = c.to_dict()
    assert d["chunk_id"] == c.chunk_id
    assert d["headings_path"] == ["People", "Faculty"]
    assert d["chunk_index"] == 0


def test_from_dict_round_trips():
    c = _sample_chunk()
    restored = ChunkRecord.from_dict(c.to_dict())
    assert restored.chunk_id == c.chunk_id
    assert restored.text == c.text
    assert restored.headings_path == c.headings_path
    assert restored.crawl_timestamp == c.crawl_timestamp


def test_from_dict_handles_missing_optional_fields():
    d = _sample_chunk().to_dict()
    # headings_path is always required in schema, but test robustness
    d["page_type"] = "general"
    restored = ChunkRecord.from_dict(d)
    assert restored.page_type == "general"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker.models'`

- [ ] **Step 3: Implement `models.py`**

```python
"""ChunkRecord dataclass — the output unit of the chunking service."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ChunkRecord:
    """A single chunk produced from one crawled page."""

    chunk_id: str          # "{document_id}_{chunk_index:04d}"
    document_id: str       # 16-hex doc ID from crawler
    chunk_index: int       # 0-based position within the document
    text: str              # Chunk content (includes heading)
    url: str               # Source page URL
    title: str             # Page title
    page_type: str         # "faculty" | "course" | "research" | "news" | "about" | "general"
    headings_path: List[str]  # e.g. ["People", "Faculty", "Professor Jane Doe"]
    content_hash: str      # SHA-256[:16] of this chunk's text
    crawl_timestamp: str   # ISO timestamp from frontmatter

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "page_type": self.page_type,
            "headings_path": self.headings_path,
            "content_hash": self.content_hash,
            "crawl_timestamp": self.crawl_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChunkRecord:
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            url=data["url"],
            title=data["title"],
            page_type=data["page_type"],
            headings_path=data["headings_path"],
            content_hash=data["content_hash"],
            crawl_timestamp=data["crawl_timestamp"],
        )
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/chunker && pytest tests/test_models.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/models.py services/chunker/tests/test_models.py
git commit -m "feat: add ChunkRecord model"
```

---

## Task 4: Markdown parser

**Files:**
- Create: `services/chunker/src/chunker/parser.py`
- Create: `services/chunker/tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

File: `services/chunker/tests/test_parser.py`

```python
"""Tests for markdown frontmatter parser and section splitter."""
import pytest
from chunker.parser import (
    FrontmatterData,
    Section,
    parse_frontmatter,
    split_sections,
    infer_page_type,
)

# --- parse_frontmatter ---

FULL_DOC = """\
---
doc_id: 'abc1234567890abc'
url: 'https://website.cs.vt.edu/people/faculty'
title: 'Faculty | CS | VT'
crawl_depth: 1
crawl_timestamp: '2026-03-16T03:00:00+00:00'
content_hash: 'deadbeef01234567'
---

# People

## Faculty

Professor Smith does research in systems.
"""

INCOMPLETE_DOC = """\
---
url: '_department-info'
title: 'CS Department Contact Information'
crawl_depth: 0
crawl_timestamp: 'generated'
---

# CS Department
"""

NO_FRONTMATTER_DOC = "# Heading\n\nSome text.\n"


def test_parse_frontmatter_extracts_all_fields():
    fm, body = parse_frontmatter(FULL_DOC)
    assert fm.doc_id == "abc1234567890abc"
    assert fm.url == "https://website.cs.vt.edu/people/faculty"
    assert fm.title == "Faculty | CS | VT"
    assert fm.content_hash == "deadbeef01234567"
    assert fm.crawl_timestamp == "2026-03-16T03:00:00+00:00"


def test_parse_frontmatter_body_is_text_after_delimiter():
    fm, body = parse_frontmatter(FULL_DOC)
    assert "# People" in body
    assert "---" not in body


def test_parse_frontmatter_incomplete_derives_doc_id():
    fm, body = parse_frontmatter(INCOMPLETE_DOC)
    assert len(fm.doc_id) == 16
    assert all(c in "0123456789abcdef" for c in fm.doc_id)


def test_parse_frontmatter_incomplete_derives_content_hash():
    fm, body = parse_frontmatter(INCOMPLETE_DOC)
    assert len(fm.content_hash) > 0


def test_parse_frontmatter_no_frontmatter():
    fm, body = parse_frontmatter(NO_FRONTMATTER_DOC)
    assert "# Heading" in body
    assert fm.url == ""


# --- split_sections ---

MULTI_HEADING_DOC = """\
# People

Intro text.

## Faculty

Professor Smith.

## Staff

Alice Bob.

### Admin Staff

Charlie.
"""

FLAT_DOC = """\
Some intro with no heading.

More text here.
"""

def test_split_sections_one_section_per_heading():
    sections = split_sections(MULTI_HEADING_DOC)
    # Should produce sections for: root (before first heading), # People, ## Faculty, ## Staff, ### Admin Staff
    # But root section may be empty — at minimum we get the 4 heading sections
    heading_texts = [s.text for s in sections]
    assert any("## Faculty" in t for t in heading_texts)
    assert any("## Staff" in t for t in heading_texts)
    assert any("### Admin Staff" in t for t in heading_texts)


def test_split_sections_headings_path_h1():
    sections = split_sections(MULTI_HEADING_DOC)
    people_section = next(s for s in sections if "# People" in s.text)
    assert people_section.headings_path == ["People"]


def test_split_sections_headings_path_h2():
    sections = split_sections(MULTI_HEADING_DOC)
    faculty_section = next(s for s in sections if "## Faculty" in s.text)
    assert faculty_section.headings_path == ["People", "Faculty"]


def test_split_sections_headings_path_h3():
    sections = split_sections(MULTI_HEADING_DOC)
    admin_section = next(s for s in sections if "### Admin Staff" in s.text)
    assert admin_section.headings_path == ["People", "Staff", "Admin Staff"]


def test_split_sections_no_headings_returns_one_section():
    sections = split_sections(FLAT_DOC)
    assert len(sections) == 1
    assert sections[0].headings_path == []


def test_split_sections_each_section_has_text():
    sections = split_sections(MULTI_HEADING_DOC)
    for s in sections:
        assert isinstance(s.text, str)
        assert len(s.text) > 0


# --- infer_page_type ---

def test_infer_page_type_faculty():
    assert infer_page_type("https://website.cs.vt.edu/people/faculty") == "faculty"


def test_infer_page_type_course():
    assert infer_page_type("https://website.cs.vt.edu/courses/cs4664") == "course"


def test_infer_page_type_research():
    assert infer_page_type("https://website.cs.vt.edu/research/labs") == "research"


def test_infer_page_type_news():
    assert infer_page_type("https://website.cs.vt.edu/news/2026-award") == "news"


def test_infer_page_type_about():
    assert infer_page_type("https://website.cs.vt.edu/about") == "about"


def test_infer_page_type_default():
    assert infer_page_type("https://website.cs.vt.edu/") == "general"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_parser.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker.parser'`

- [ ] **Step 3: Implement `parser.py`**

```python
"""Parse YAML frontmatter and split markdown into semantic sections."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


@dataclass
class FrontmatterData:
    """Fields extracted from a markdown YAML frontmatter block."""
    doc_id: str
    url: str
    title: str
    content_hash: str
    crawl_timestamp: str


@dataclass
class Section:
    """A single semantic section: one heading + its body text."""
    headings_path: List[str]
    text: str  # includes the heading line itself


def _parse_simple_yaml(block: str) -> dict:
    """Parse a minimal YAML block (key: 'value' or key: value lines only)."""
    result: dict = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        val = raw_val.strip().strip("'\"")
        result[key] = val
    return result


def _derive_doc_id(url: str) -> str:
    """Derive a 16-hex doc_id from a URL string (same algorithm as crawler)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _derive_content_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def parse_frontmatter(markdown: str) -> Tuple[FrontmatterData, str]:
    """Extract YAML frontmatter and return (FrontmatterData, body_text).

    If frontmatter is absent or fields are missing, sensible fallbacks are used
    so that synthetic documents like _department-info.md are handled gracefully.
    """
    body = markdown
    fields: dict = {}

    if markdown.startswith("---"):
        end = markdown.find("---", 3)
        if end != -1:
            fm_block = markdown[3:end].strip()
            body = markdown[end + 3:]
            fields = _parse_simple_yaml(fm_block)

    url = fields.get("url", "")
    doc_id = fields.get("doc_id") or _derive_doc_id(url)
    content_hash = fields.get("content_hash") or _derive_content_hash(body)
    title = fields.get("title", url or "Untitled")
    crawl_timestamp = fields.get("crawl_timestamp", "")

    return FrontmatterData(
        doc_id=doc_id,
        url=url,
        title=title,
        content_hash=content_hash,
        crawl_timestamp=crawl_timestamp,
    ), body


def split_sections(body: str) -> List[Section]:
    """Split markdown body into sections at h1/h2/h3 heading boundaries.

    Each section includes its heading line and all text until the next heading.
    Heading hierarchy is tracked to populate headings_path.
    Text before the first heading becomes a section with empty headings_path.
    """
    lines = body.splitlines(keepends=True)
    # Find all heading positions
    heading_positions: list[tuple[int, int, str]] = []  # (line_index, level, title)
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.+)$", line.rstrip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_positions.append((i, level, title))

    if not heading_positions:
        text = body.strip()
        if not text:
            return []
        return [Section(headings_path=[], text=text)]

    sections: List[Section] = []

    # Text before the first heading
    first_heading_line = heading_positions[0][0]
    pre_text = "".join(lines[:first_heading_line]).strip()
    if pre_text:
        sections.append(Section(headings_path=[], text=pre_text))

    # Build sections for each heading
    # heading_stack tracks (level, title) for the current path
    heading_stack: list[tuple[int, str]] = []

    for idx, (line_i, level, title) in enumerate(heading_positions):
        # Determine end of this section
        if idx + 1 < len(heading_positions):
            end_line = heading_positions[idx + 1][0]
        else:
            end_line = len(lines)

        section_text = "".join(lines[line_i:end_line]).strip()
        if not section_text:
            continue

        # Update heading stack: pop levels >= current
        heading_stack = [(l, t) for l, t in heading_stack if l < level]
        heading_stack.append((level, title))
        headings_path = [t for _, t in heading_stack]

        sections.append(Section(headings_path=headings_path, text=section_text))

    return sections


def infer_page_type(url: str) -> str:
    """Infer a coarse page type from the URL path."""
    path = url.lower()
    if any(k in path for k in ["/people/", "/faculty/", "faculty"]):
        return "faculty"
    if any(k in path for k in ["/courses/", "/classes/", "/course"]):
        return "course"
    if any(k in path for k in ["/research/", "/labs/"]):
        return "research"
    if any(k in path for k in ["/news/", "/events/"]):
        return "news"
    if any(k in path for k in ["/about", "/info"]):
        return "about"
    return "general"
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/chunker && pytest tests/test_parser.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/parser.py services/chunker/tests/test_parser.py
git commit -m "feat: add markdown frontmatter parser and section splitter"
```

---

## Task 5: Token-aware chunking

**Files:**
- Create: `services/chunker/src/chunker/splitter.py`
- Create: `services/chunker/tests/test_splitter.py`

- [ ] **Step 1: Write failing tests**

File: `services/chunker/tests/test_splitter.py`

```python
"""Tests for token-aware chunking logic."""
import pytest
from chunker.parser import Section
from chunker.splitter import (
    estimate_tokens,
    split_large_section,
    build_chunks,
)
from chunker.models import ChunkRecord


# --- estimate_tokens ---

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    assert estimate_tokens("Hello world") == len("Hello world") // 4


def test_estimate_tokens_400_char_string():
    text = "a" * 400
    assert estimate_tokens(text) == 100


# --- split_large_section ---

def _make_section(text: str) -> Section:
    return Section(headings_path=["Test"], text=text)


def test_split_large_section_small_input_stays_single():
    text = "Short text. " * 10  # ~12 tokens, well under 400
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_split_large_section_large_input_splits():
    # Create text > 400 tokens (> 1600 chars)
    para = "This is a paragraph with some content. " * 10  # ~390 chars per repeat
    text = (para + "\n\n") * 5  # ~2000+ chars → > 500 tokens
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) >= 2


def test_split_large_section_no_chunk_exceeds_limit():
    para = "Word " * 200  # 1000 chars = 250 tokens each
    text = (para + "\n\n") * 4   # 4 paragraphs, total ~4000 chars = ~1000 tokens
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 400 + 64  # allow a little slack for overlap


def test_split_large_section_overlap_present():
    # Two big paragraphs; the second chunk should start with some chars from the first
    para = "X" * 1800  # 450 tokens
    text = para + "\n\n" + "Y" * 1800
    chunks = split_large_section(_make_section(text), preferred_tokens=400, overlap_tokens=64)
    assert len(chunks) >= 2
    # Second chunk should contain some overlap from first (last 64*4=256 chars of first)
    overlap_chars = 64 * 4
    first_end = chunks[0][-overlap_chars:]
    assert first_end in chunks[1]


# --- build_chunks ---

def _make_sections(sizes_in_tokens: list[int]) -> list[Section]:
    """Create sections with approximate given token sizes."""
    sections = []
    for i, tokens in enumerate(sizes_in_tokens):
        text = f"## Section {i}\n\n" + "word " * (tokens * 4 // 5)
        sections.append(Section(headings_path=[f"Section {i}"], text=text))
    return sections


FAKE_FM = type("FM", (), {
    "doc_id": "abc1234567890abc",
    "url": "https://website.cs.vt.edu/test",
    "title": "Test Page",
    "content_hash": "deadbeef01234567",
    "crawl_timestamp": "2026-03-16T03:00:00",
})()

FAKE_CONFIG = type("C", (), {
    "chunk_preferred_tokens": 400,
    "chunk_overlap_tokens": 64,
    "chunk_min_tokens": 120,
})()


def test_build_chunks_returns_chunk_records():
    sections = _make_sections([200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], ChunkRecord)


def test_build_chunks_chunk_ids_sequential():
    sections = _make_sections([200, 200, 200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_id == f"abc1234567890abc_{i:04d}"


def test_build_chunks_small_sections_merged():
    # 3 small sections (50 tokens each) should be merged into fewer chunks
    sections = _make_sections([50, 50, 50])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) < 3  # should be merged


def test_build_chunks_large_section_split():
    # 1 section of ~600 tokens should produce 2+ chunks
    sections = _make_sections([600])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    assert len(chunks) >= 2


def test_build_chunks_metadata_fields():
    sections = _make_sections([200])
    chunks = build_chunks(sections, FAKE_FM, FAKE_CONFIG)
    c = chunks[0]
    assert c.document_id == "abc1234567890abc"
    assert c.url == "https://website.cs.vt.edu/test"
    assert c.title == "Test Page"
    assert c.crawl_timestamp == "2026-03-16T03:00:00"
    assert len(c.content_hash) == 16


def test_build_chunks_empty_sections_returns_empty():
    chunks = build_chunks([], FAKE_FM, FAKE_CONFIG)
    assert chunks == []
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_splitter.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker.splitter'`

- [ ] **Step 3: Implement `splitter.py`**

```python
"""Token-aware chunking: merge small sections, split large ones with overlap."""
from __future__ import annotations

import hashlib
from typing import Any, List

from chunker.models import ChunkRecord
from chunker.parser import FrontmatterData, Section, infer_page_type


def estimate_tokens(text: str) -> int:
    """Approximate token count as character count // 4 (4 chars ≈ 1 token)."""
    return len(text) // 4


def split_large_section(section: Section, preferred_tokens: int, overlap_tokens: int) -> List[str]:
    """Split an oversized section into windows at paragraph boundaries with overlap.

    Strategy:
    1. Split on blank lines (paragraphs).
    2. Accumulate paragraphs until the window would exceed preferred_tokens.
    3. Flush the window, then rewind by overlap_tokens chars for the next window.
    """
    preferred_chars = preferred_tokens * 4
    overlap_chars = overlap_tokens * 4

    if estimate_tokens(section.text) <= preferred_tokens:
        return [section.text.strip()]

    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]

    windows: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 > preferred_chars and current_parts:
            window_text = "\n\n".join(current_parts)
            windows.append(window_text)
            # Rewind: keep tail of current window for overlap
            overlap_text = window_text[-overlap_chars:]
            current_parts = [overlap_text] if overlap_text.strip() else []
            current_len = len(overlap_text)
        current_parts.append(para)
        current_len += para_len + 2  # +2 for the '\n\n' separator

    if current_parts:
        windows.append("\n\n".join(current_parts))

    return windows if windows else [section.text.strip()]


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_chunks(
    sections: List[Section],
    frontmatter: Any,  # FrontmatterData or duck-typed
    config: Any,       # ChunkerConfig or duck-typed
) -> List[ChunkRecord]:
    """Convert a list of sections into ChunkRecord objects.

    Rules:
    - section < chunk_min_tokens  → accumulate into a pending batch for forward merge
    - section in [min, preferred] → flush pending (if any), emit section as one chunk
    - section > preferred         → flush pending, split into windows with overlap
    """
    preferred = config.chunk_preferred_tokens
    overlap = config.chunk_overlap_tokens
    minimum = config.chunk_min_tokens
    page_type = infer_page_type(frontmatter.url)

    chunk_texts: List[tuple[str, List[str]]] = []  # (text, headings_path)
    pending_parts: List[Section] = []

    def _flush_pending() -> None:
        if not pending_parts:
            return
        merged_text = "\n\n".join(s.text for s in pending_parts)
        merged_path = pending_parts[0].headings_path
        chunk_texts.append((merged_text, merged_path))
        pending_parts.clear()

    for section in sections:
        tokens = estimate_tokens(section.text)

        if tokens > preferred:
            _flush_pending()
            windows = split_large_section(section, preferred, overlap)
            for window in windows:
                chunk_texts.append((window, section.headings_path))
        elif tokens < minimum:
            pending_parts.append(section)
            # If pending accumulated enough, flush
            pending_total = sum(estimate_tokens(s.text) for s in pending_parts)
            if pending_total >= minimum:
                _flush_pending()
        else:
            if pending_parts:
                combined = sum(estimate_tokens(s.text) for s in pending_parts) + tokens
                if combined <= preferred:
                    pending_parts.append(section)
                    _flush_pending()
                else:
                    _flush_pending()
                    chunk_texts.append((section.text, section.headings_path))
            else:
                chunk_texts.append((section.text, section.headings_path))

    _flush_pending()

    records: List[ChunkRecord] = []
    for i, (text, headings_path) in enumerate(chunk_texts):
        records.append(ChunkRecord(
            chunk_id=f"{frontmatter.doc_id}_{i:04d}",
            document_id=frontmatter.doc_id,
            chunk_index=i,
            text=text,
            url=frontmatter.url,
            title=frontmatter.title,
            page_type=page_type,
            headings_path=headings_path,
            content_hash=_chunk_hash(text),
            crawl_timestamp=frontmatter.crawl_timestamp,
        ))
    return records
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/chunker && pytest tests/test_splitter.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add services/chunker/src/chunker/splitter.py services/chunker/tests/test_splitter.py
git commit -m "feat: add token-aware chunking splitter"
```

---

## Task 6: Storage layer

**Files:**
- Create: `services/chunker/tests/conftest.py` (created here so test_storage.py can use the shared fixture)
- Create: `services/chunker/src/chunker/storage.py`
- Create: `services/chunker/tests/test_storage.py`

**Note:** `conftest.py` is created in this task (not Task 7) because `test_storage.py` needs the shared `chunker_config` fixture. Task 7 only adds `test_main.py`, which reuses the same conftest.

- [ ] **Step 1: Write `conftest.py`**

```python
"""Shared test fixtures for the chunker service."""
import pytest
from chunker.config import ChunkerConfig


@pytest.fixture
def chunker_config():
    return ChunkerConfig(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        minio_cleaned_bucket="crawled-pages-cleaned",
        minio_chunks_bucket="chunks",
        chunk_preferred_tokens=400,
        chunk_overlap_tokens=64,
        chunk_min_tokens=120,
    )
```

- [ ] **Step 2: Write failing tests**

File: `services/chunker/tests/test_storage.py`

```python
"""Tests for ChunkerStorage using a mocked Minio client."""
import json
import io
from unittest.mock import MagicMock, patch, call
import pytest

from chunker.storage import ChunkerStorage
from chunker.models import ChunkRecord


@pytest.fixture
def mock_minio():
    client = MagicMock()
    client.bucket_exists.return_value = True
    return client


@pytest.fixture
def storage(chunker_config, mock_minio):
    with patch("chunker.storage.Minio", return_value=mock_minio):
        s = ChunkerStorage(chunker_config)
    s._client = mock_minio
    return s


def test_list_markdown_keys_filters_md_only(storage, mock_minio):
    obj1 = MagicMock(); obj1.object_name = "website.cs.vt.edu/about.md"
    obj2 = MagicMock(); obj2.object_name = "website.cs.vt.edu/about.meta.json"
    obj3 = MagicMock(); obj3.object_name = "website.cs.vt.edu/people.md"
    mock_minio.list_objects.return_value = [obj1, obj2, obj3]
    keys = storage.list_markdown_keys()
    assert keys == ["website.cs.vt.edu/about.md", "website.cs.vt.edu/people.md"]


def test_download_document_returns_content(storage, mock_minio):
    resp = MagicMock()
    resp.read.return_value = b"# Hello\n\nWorld"
    mock_minio.get_object.return_value = resp
    content = storage.download_document("website.cs.vt.edu/about.md")
    assert content == "# Hello\n\nWorld"
    resp.close.assert_called_once()
    resp.release_conn.assert_called_once()


def test_upload_chunks_serialises_to_json(storage, mock_minio):
    chunks = [
        ChunkRecord(
            chunk_id="abc_0000", document_id="abc", chunk_index=0,
            text="Hello", url="https://example.com", title="Test",
            page_type="general", headings_path=[], content_hash="deadbeef01234567",
            crawl_timestamp="2026-03-16T00:00:00",
        )
    ]
    storage.upload_chunks("abc", chunks)
    assert mock_minio.put_object.called
    call_args = mock_minio.put_object.call_args
    bucket = call_args[0][0]
    key = call_args[0][1]
    assert bucket == "chunks"
    assert key == "chunks/abc.json"
    # Verify content is valid JSON array
    data_stream = call_args[0][2]
    content = json.loads(data_stream.read().decode())
    assert isinstance(content, list)
    assert content[0]["chunk_id"] == "abc_0000"


def test_init_creates_bucket_if_missing(chunker_config, mock_minio):
    """Verify bucket is created during __init__ when it does not exist."""
    mock_minio.bucket_exists.return_value = False
    with patch("chunker.storage.Minio", return_value=mock_minio):
        ChunkerStorage(chunker_config)
    mock_minio.make_bucket.assert_called_once_with("chunks")


def test_ensure_bucket_public_method(storage, mock_minio):
    """ChunkerStorage exposes ensure_bucket() for parity with MinioStorage."""
    mock_minio.bucket_exists.return_value = False
    storage.ensure_bucket("new-bucket")
    mock_minio.make_bucket.assert_called_with("new-bucket")
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_storage.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker.storage'`

- [ ] **Step 3: Implement `storage.py`**

```python
"""MinIO storage layer for the chunker service."""
from __future__ import annotations

import io
import json
import logging
from typing import List

from minio import Minio

from chunker.config import ChunkerConfig
from chunker.models import ChunkRecord

logger = logging.getLogger(__name__)


class ChunkerStorage:
    """Reads cleaned markdown from MinIO; writes chunk JSON to MinIO."""

    def __init__(self, config: ChunkerConfig) -> None:
        self._cleaned_bucket = config.minio_cleaned_bucket
        self._chunks_bucket = config.minio_chunks_bucket
        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self._ensure_bucket(self._chunks_bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info("Created bucket %s", bucket)

    def list_markdown_keys(self) -> List[str]:
        """Return all .md object keys from the cleaned bucket (excludes .meta.json)."""
        return [
            obj.object_name
            for obj in self._client.list_objects(self._cleaned_bucket, recursive=True)
            if obj.object_name.endswith(".md")
        ]

    def download_document(self, key: str) -> str:
        """Download a markdown document from the cleaned bucket."""
        response = self._client.get_object(self._cleaned_bucket, key)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure a bucket exists, creating it if needed (public, mirrors MinioStorage)."""
        self._ensure_bucket(bucket)

    def upload_chunks(self, doc_id: str, chunks: List[ChunkRecord]) -> None:
        """Upload all chunks for a document as chunks/<doc_id>.json."""
        key = f"chunks/{doc_id}.json"
        payload = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2)
        data = payload.encode("utf-8")
        self._client.put_object(
            self._chunks_bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        logger.info("Uploaded %s (%d chunks, %d bytes) to %s", key, len(chunks), len(data), self._chunks_bucket)
```

- [ ] **Step 4: Run to confirm PASS**

```bash
cd services/chunker && pytest tests/test_storage.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add services/chunker/tests/conftest.py services/chunker/src/chunker/storage.py services/chunker/tests/test_storage.py
git commit -m "feat: add chunker storage layer"
```

---

## Task 7: Main entrypoint and end-to-end test

**Files:**
- Create: `services/chunker/src/chunker/main.py`
- Create: `services/chunker/tests/test_main.py`

Note: `conftest.py` was already created in Task 6. `test_main.py` uses the `chunker_config` fixture from it.

- [ ] **Step 1: Write failing tests**

File: `services/chunker/tests/test_main.py`

```python
"""End-to-end tests for the chunker main orchestration."""
import json
from unittest.mock import MagicMock, patch
import pytest

from chunker.main import run_chunking


SAMPLE_DOC = """\
---
doc_id: 'abc1234567890abc'
url: 'https://website.cs.vt.edu/people/faculty'
title: 'Faculty | CS | VT'
crawl_depth: 1
crawl_timestamp: '2026-03-16T03:00:00+00:00'
content_hash: 'deadbeef01234567'
---

# People

## Faculty

Professor Smith researches distributed systems and has published extensively.
Professor Jones works on machine learning and has won several awards.
"""

EMPTY_DOC = """\
---
doc_id: 'empty000000000000'
url: 'https://website.cs.vt.edu/empty'
title: 'Empty'
crawl_depth: 1
crawl_timestamp: '2026-03-16T03:00:00+00:00'
content_hash: 'aaaa0000bbbb1111'
---

"""


def _make_storage(docs: dict) -> MagicMock:
    """Build a fake ChunkerStorage that serves given key→content mapping."""
    storage = MagicMock()
    storage.list_markdown_keys.return_value = list(docs.keys())
    storage.download_document.side_effect = lambda key: docs[key]
    return storage


def test_run_chunking_processes_document(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/faculty.md": SAMPLE_DOC})
    stats = run_chunking(storage, chunker_config)
    assert stats["processed"] == 1
    assert stats["total_chunks"] >= 1
    assert stats["failed"] == 0
    assert storage.upload_chunks.called


def test_run_chunking_skips_empty_body(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/empty.md": EMPTY_DOC})
    stats = run_chunking(storage, chunker_config)
    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert not storage.upload_chunks.called


def test_run_chunking_handles_failure(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/bad.md": "good content"})
    storage.upload_chunks.side_effect = RuntimeError("MinIO unavailable")
    # download succeeds but upload fails — should count as failed
    stats = run_chunking(storage, chunker_config)
    assert stats["failed"] == 1
    assert stats["processed"] == 0


def test_run_chunking_chunk_ids_use_doc_id(chunker_config):
    storage = _make_storage({"website.cs.vt.edu/faculty.md": SAMPLE_DOC})
    run_chunking(storage, chunker_config)
    call_args = storage.upload_chunks.call_args
    doc_id = call_args[0][0]
    chunks = call_args[0][1]
    assert doc_id == "abc1234567890abc"
    for c in chunks:
        assert c.chunk_id.startswith("abc1234567890abc_")


def test_run_chunking_multiple_docs(chunker_config):
    docs = {
        "website.cs.vt.edu/a.md": SAMPLE_DOC,
        "website.cs.vt.edu/b.md": SAMPLE_DOC,
    }
    storage = _make_storage(docs)
    stats = run_chunking(storage, chunker_config)
    assert stats["processed"] == 2
    assert storage.upload_chunks.call_count == 2
```

- [ ] **Step 3: Run to confirm FAIL**

```bash
cd services/chunker && pytest tests/test_main.py -v
```

Expected: `ImportError: cannot import name 'run_chunking'`

- [ ] **Step 4: Implement `main.py`**

```python
"""Chunker entrypoint — read cleaned docs, chunk, write chunk records."""
from __future__ import annotations

import logging
import sys

from chunker.config import ChunkerConfig
from chunker.parser import parse_frontmatter, split_sections
from chunker.splitter import build_chunks
from chunker.storage import ChunkerStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_chunking(storage: ChunkerStorage, config: ChunkerConfig) -> dict:
    """Process all cleaned documents and write chunk records.

    Returns a stats dict with keys: processed, skipped, total_chunks, failed.
    """
    keys = storage.list_markdown_keys()
    logger.info("Found %d markdown documents to chunk", len(keys))

    stats = {"processed": 0, "skipped": 0, "total_chunks": 0, "failed": 0}

    for key in keys:
        try:
            content = storage.download_document(key)
            frontmatter, body = parse_frontmatter(content)

            if not body.strip():
                logger.debug("Skipping %s — empty body", key)
                stats["skipped"] += 1
                continue

            sections = split_sections(body)
            if not sections:
                logger.debug("Skipping %s — no sections after split", key)
                stats["skipped"] += 1
                continue

            chunks = build_chunks(sections, frontmatter, config)
            storage.upload_chunks(frontmatter.doc_id, chunks)

            stats["processed"] += 1
            stats["total_chunks"] += len(chunks)
            logger.info("Chunked %s → doc_id=%s, %d chunks", key, frontmatter.doc_id, len(chunks))

        except Exception as exc:
            logger.error("Failed to chunk %s: %s", key, exc)
            stats["failed"] += 1

    logger.info(
        "Chunking complete: %d processed, %d skipped, %d total chunks, %d failed",
        stats["processed"], stats["skipped"], stats["total_chunks"], stats["failed"],
    )
    return stats


def cli() -> None:
    """CLI entrypoint for the chunker service."""
    logger.info("Starting HokieHelp chunker")

    try:
        config = ChunkerConfig.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    storage = ChunkerStorage(config)
    stats = run_chunking(storage, config)

    if stats["failed"] > 0:
        logger.warning("%d documents failed to chunk", stats["failed"])
        sys.exit(1)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 5: Run all tests to confirm PASS**

```bash
cd services/chunker && pytest -v
```

Expected: all tests pass, 0 failures

- [ ] **Step 6: Commit**

```bash
git add services/chunker/src/chunker/main.py services/chunker/tests/test_main.py
git commit -m "feat: add chunker main entrypoint and end-to-end tests"
```

---

## Task 8: Dockerfile

**Files:**
- Create: `services/chunker/Dockerfile`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

CMD ["python", "-m", "chunker.main"]
```

Note: No browser dependency — chunker only needs Python + minio SDK.

- [ ] **Step 2: Build locally to confirm it works (optional)**

```bash
cd services/chunker && docker build -t hokiehelp-chunker:local .
```

Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
git add services/chunker/Dockerfile
git commit -m "feat: add chunker Dockerfile"
```

---

## Task 9: Kubernetes manifests

**Files:**
- Create: `k8s/chunker-configmap.yaml`
- Create: `k8s/chunker-cronjob.yaml`

- [ ] **Step 1: Write `k8s/chunker-configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chunker-config
  namespace: test
  labels:
    app: hokiehelp-chunker
data:
  MINIO_ENDPOINT: "minio:9000"
  MINIO_CLEANED_BUCKET: "crawled-pages-cleaned"
  MINIO_CHUNKS_BUCKET: "chunks"
  MINIO_SECURE: "false"
  CHUNK_PREFERRED_TOKENS: "400"
  CHUNK_OVERLAP_TOKENS: "64"
  CHUNK_MIN_TOKENS: "120"
```

- [ ] **Step 2: Write `k8s/chunker-cronjob.yaml`**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hokiehelp-chunker
  namespace: test
  labels:
    app: hokiehelp-chunker
spec:
  schedule: "0 4 * * 0"  # Weekly on Sunday at 4am (1h after crawler at 3am)
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600  # 1 hour timeout
      template:
        metadata:
          labels:
            app: hokiehelp-chunker
        spec:
          restartPolicy: Never
          containers:
            - name: chunker
              image: ghcr.io/prakharmodi26/hokiehelp-chunker:latest
              imagePullPolicy: Always
              envFrom:
                - configMapRef:
                    name: chunker-config
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
                  memory: "256Mi"
                  cpu: "250m"
                limits:
                  memory: "1Gi"
                  cpu: "500m"
```

- [ ] **Step 3: Commit**

```bash
git add k8s/chunker-configmap.yaml k8s/chunker-cronjob.yaml
git commit -m "feat: add K8s manifests for chunker service"
```

---

## Task 10: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/chunker-ci.yaml`

- [ ] **Step 1: Write `.github/workflows/chunker-ci.yaml`**

```yaml
name: Chunker CI

on:
  push:
    branches: [main]
    paths:
      - "services/chunker/**"
      - ".github/workflows/chunker-ci.yaml"
  pull_request:
    branches: [main]
    paths:
      - "services/chunker/**"

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: prakharmodi26/hokiehelp-chunker

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/chunker
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ".[dev]"
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
          context: services/chunker
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/chunker-ci.yaml
git commit -m "feat: add GitHub Actions CI/CD for chunker service"
git push
```

Expected: CI pipeline triggers, tests run, Docker image pushed to GHCR on merge to `main`.

---

## Task 11: Deploy to cluster

- [ ] **Step 1: Show current kubectl context**

```bash
kubectl config current-context
```

Expected: `endeavour`

- [ ] **Step 2: Apply ConfigMap**

```bash
kubectl apply -f k8s/chunker-configmap.yaml
```

- [ ] **Step 3: Apply CronJob**

```bash
kubectl apply -f k8s/chunker-cronjob.yaml
```

- [ ] **Step 4: Verify resources**

```bash
kubectl get cronjobs -n test
kubectl get configmap chunker-config -n test
```

- [ ] **Step 5: Trigger a manual run to validate end-to-end**

```bash
kubectl create job --from=cronjob/hokiehelp-chunker chunker-test -n test
kubectl logs -n test -l app=hokiehelp-chunker -f
```

Expected logs:
```
INFO chunker.main: Found N markdown documents to chunk
INFO chunker.main: Chunked website.cs.vt.edu/... → doc_id=..., N chunks
INFO chunker.main: Chunking complete: N processed, 0 skipped, N total chunks, 0 failed
```

- [ ] **Step 6: Verify chunks in MinIO**

```bash
kubectl port-forward -n test svc/minio 9001:9001
```

Open MinIO console at `http://localhost:9001`, navigate to the `chunks` bucket. Verify `chunks/<doc_id>.json` files exist and contain valid JSON arrays.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Token estimate = `len // 4` | No extra dependencies; accurate enough for guardrail splits in English text |
| Heading-first split, token-size second | Preserves semantic meaning; avoids splitting a faculty profile mid-sentence |
| All chunks for a doc in one JSON file | Simpler to read/delete atomically; downstream embedder reads one file per doc |
| CronJob Sunday 4am (1h after crawler) | Crawler runs 3am; 1 hour buffer for crawler to finish before chunker starts |
| Same `minio-credentials` Secret | Reuse existing credentials — no new K8s secrets needed |
| `sys.exit(1)` on any failure | K8s Job sees non-zero exit, retries up to `backoffLimit: 2` |
