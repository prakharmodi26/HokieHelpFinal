"""Parse YAML frontmatter and split markdown into semantic sections."""
from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
from urllib.parse import urlparse


_CMS_ERROR_RE = re.compile(r"^#\s+Resource at '.*?' not found", re.MULTILINE)


def is_cms_error_page(body: str) -> bool:
    """Return True if the body is a CMS 'Resource not found' page (safety-net filter)."""
    return bool(_CMS_ERROR_RE.search(body[:500]))


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
