"""CourseLeaf catalog parsing and markdown cards (Task 8.4).

This module is intentionally fixture-testable: callers provide already
rendered CourseLeaf HTML, and these helpers parse structured course/program
records plus deterministic markdown cards. The browser-render crawl pass that
feeds this module is left to the parent integration task.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from html import unescape
from urllib.parse import urljoin, urlparse

import yaml
from bs4 import BeautifulSoup, Tag
from crawl4ai import AsyncWebCrawler

from crawler.render import js_run_config

logger = logging.getLogger(__name__)
CATALOG_BASE_URL = "https://catalog.vt.edu"
COURSE_INDEX_URL = f"{CATALOG_BASE_URL}/course-descriptions/"
PROGRAM_SEED_URLS = (
    f"{CATALOG_BASE_URL}/program-explorer/",
    f"{CATALOG_BASE_URL}/undergraduate/",
    f"{CATALOG_BASE_URL}/graduate/",
)

_COURSE_CODE_RE = re.compile(
    r"\b(?P<subject>[A-Z&]{2,8})\s*(?P<number>\d{4})(?P<suffix>[A-Z]?)\b"
)
_CREDITS_RE = re.compile(
    r"\(?\s*(?P<credits>\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+"
    r"(?:credit|credits|hours?)\s*\)?",
    re.IGNORECASE,
)
_TITLE_SPLIT_RE = re.compile(r"\s*(?:[:.-]\s+|\s+-\s+)")
_LABEL_RE = re.compile(
    r"^(?P<label>Prerequisite(?:\(s\))?|Corequisite(?:\(s\))?|"
    r"Pre-?/?Co-?requisite(?:\(s\))?|When Offered|Cross[- ]Listed|"
    r"Cross[- ]Listing(?:\(s\))?|Also Listed As)\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CourseRecord:
    subject: str
    number: str
    level: str
    title: str
    credits: str | None = None
    prerequisites: str | None = None
    corequisites: str | None = None
    when_offered: str | None = None
    cross_listings: list[str] = field(default_factory=list)
    description: str | None = None
    url: str | None = None
    key: str | None = None


@dataclass(frozen=True)
class ProgramRecord:
    title: str
    url: str
    degree: str | None = None
    college: str | None = None
    department: str | None = None
    description: str | None = None
    requirements: list[str] = field(default_factory=list)
    level: str | None = None
    key: str | None = None


def _clean_text(value: str | None) -> str:
    text = unescape(value or "")
    text = text.replace("\xa0", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _tag_text(tag: Tag | None, separator: str = " ") -> str:
    if tag is None:
        return ""
    return _clean_text(tag.get_text(separator, strip=True))


def _course_level(number: str) -> str:
    return "ug" if int(number[:4]) < 5000 else "grad"


def course_key(subject: str, number: str) -> str:
    """Stable key used for exact lookup payloads and document IDs."""
    return f"{subject.strip().upper()}-{number.strip().upper()}"


def program_key(url_or_title: str) -> str:
    """Stable program key from URL path when possible, title otherwise."""
    parsed = urlparse(url_or_title)
    source = parsed.path.strip("/") if parsed.scheme or parsed.netloc else url_or_title
    source = source or url_or_title
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-._").lower()
    return safe or "program"


def course_subject_url(subject: str, base_url: str = CATALOG_BASE_URL) -> str:
    """CourseLeaf subject page URL, e.g. /course-descriptions/cs/."""
    return urljoin(base_url.rstrip("/") + "/", f"course-descriptions/{subject.lower()}/")


def course_record_url(record: CourseRecord | dict) -> str:
    """Stable per-course URL for document identity and source linking."""
    data = _record_dict(record)
    key = data.get("key") or course_key(data["subject"], data["number"])
    base = data.get("url") or course_subject_url(data["subject"])
    return f"{base.rstrip('/')}#{key}"


def parse_subject_urls(html: str, source_url: str = COURSE_INDEX_URL) -> list[str]:
    """Discover CourseLeaf subject pages from the course descriptions index."""
    soup = BeautifulSoup(html or "", "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for link in soup.select('a[href*="/course-descriptions/"]'):
        url = urljoin(source_url, str(link.get("href") or ""))
        path = urlparse(url).path
        if not re.match(r"^/course-descriptions/[a-z0-9&-]+/$", path, re.I):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def course_object_key(record: CourseRecord | dict) -> str:
    data = _record_dict(record)
    return f"catalog.vt.edu/course-descriptions/{data['subject'].lower()}/{data['key']}.md"


def program_object_key(record: ProgramRecord | dict) -> str:
    data = _record_dict(record)
    return f"catalog.vt.edu/programs/{data['key']}.md"


def _record_dict(record) -> dict:
    if hasattr(record, "__dataclass_fields__"):
        return asdict(record)
    return dict(record)


def _parse_course_header(text: str) -> dict | None:
    text = _clean_text(text)
    code = _COURSE_CODE_RE.search(text)
    if not code:
        return None

    credits_match = None
    for match in _CREDITS_RE.finditer(text):
        credits_match = match
    credits = credits_match.group("credits").replace(" ", "") if credits_match else None

    subject = code.group("subject").upper()
    number = f"{code.group('number')}{code.group('suffix')}".upper()
    rest = text[code.end():]
    if credits_match:
        rest = rest[: credits_match.start() - code.end()]
    title = _TITLE_SPLIT_RE.sub("", rest, count=1).strip(" .:-")
    return {
        "subject": subject,
        "number": number,
        "level": _course_level(code.group("number")),
        "title": _clean_text(title),
        "credits": credits,
    }


def _split_labelled_extra(text: str) -> tuple[str | None, str | None]:
    match = _LABEL_RE.match(_clean_text(text))
    if not match:
        return None, None
    label = match.group("label").lower()
    value = _clean_text(match.group("value"))
    if label.startswith("prerequisite"):
        return "prerequisites", value
    if label.startswith("corequisite"):
        return "corequisites", value
    if label.startswith("pre") and "co" in label:
        return "corequisites", value
    if label.startswith("when"):
        return "when_offered", value
    if label.startswith("cross") or label.startswith("also"):
        return "cross_listings", value
    return None, None


def _extract_cross_listings(value: str) -> list[str]:
    listings: list[str] = []
    for match in _COURSE_CODE_RE.finditer(value.upper()):
        item = f"{match.group('subject')} {match.group('number')}{match.group('suffix')}"
        if item not in listings:
            listings.append(item)
    return listings


def parse_courseblocks(html: str, source_url: str | None = None) -> list[CourseRecord]:
    """Parse CourseLeaf courseblock HTML into structured course records."""
    soup = BeautifulSoup(html or "", "html.parser")
    records: list[CourseRecord] = []
    blocks = soup.select(".courseblock")
    if not blocks:
        blocks = soup.find_all(class_=lambda c: c and "courseblock" in str(c).split())

    for block in blocks:
        title_tag = block.select_one(".courseblocktitle")
        if title_tag is None:
            title_tag = block.find(["p", "div", "span"], class_=re.compile("title"))
        header = _parse_course_header(_tag_text(title_tag or block))
        if not header:
            continue

        description_parts: list[str] = []
        fields: dict[str, str | list[str] | None] = {
            "prerequisites": None,
            "corequisites": None,
            "when_offered": None,
            "cross_listings": [],
        }

        for child in block.find_all(["p", "div"], recursive=False):
            if child is title_tag:
                continue
            classes = set(child.get("class") or [])
            text = _tag_text(child)
            if not text:
                continue
            key, value = _split_labelled_extra(text)
            if key == "cross_listings" and value:
                fields[key] = _extract_cross_listings(value)
            elif key and value:
                fields[key] = value
            elif "courseblockdesc" in classes or "desc" in " ".join(classes):
                description_parts.append(text)
            elif "courseblockextra" not in classes:
                description_parts.append(text)

        key = course_key(header["subject"], header["number"])
        records.append(CourseRecord(
            **header,
            prerequisites=fields["prerequisites"] or None,
            corequisites=fields["corequisites"] or None,
            when_offered=fields["when_offered"] or None,
            cross_listings=list(fields["cross_listings"] or []),
            description="\n\n".join(description_parts) or None,
            url=source_url,
            key=key,
        ))
    return records


def _infer_program_level(url: str) -> str | None:
    path = urlparse(url).path.lower()
    if "/undergraduate/" in path:
        return "undergraduate"
    if "/graduate/" in path:
        return "graduate"
    return None


def _infer_degree(title: str) -> str | None:
    match = re.search(
        r"(?:\bB\.[AS]\.|\bM\.[AS]\.|\bPh\.?D\.?|\bMinor\b|\bCertificate\b|"
        r"\bBachelor of [^,;()]+|\bMaster of [^,;()]+|\bDoctor of [^,;()]+)",
        title,
        flags=re.IGNORECASE,
    )
    return _clean_text(match.group(0)) if match else None


def _nearest_heading_text(tag: Tag) -> str | None:
    for prev in tag.find_all_previous(["h2", "h3", "h4"]):
        text = _tag_text(prev)
        if text:
            return text
    return None


def parse_programs(html: str, source_url: str = CATALOG_BASE_URL) -> list[ProgramRecord]:
    """Parse CourseLeaf program listing or detail HTML into program records.

    Listing pages yield one record per program link. Detail pages without a
    program list yield one record for the page with a compact description and
    requirement headings/bullets when present.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    records: list[ProgramRecord] = []
    seen: set[str] = set()
    program_links = []
    for link in soup.select("a[href]"):
        href = str(link.get("href") or "")
        path = urlparse(urljoin(source_url, href)).path.lower()
        if (
            ("/undergraduate/" in path or "/graduate/" in path)
            and path.count("/") >= 3
            and not path.endswith(("/undergraduate/", "/graduate/"))
        ):
            program_links.append(link)

    for link in program_links:
        title = _tag_text(link)
        if not title:
            continue
        url = urljoin(source_url, str(link.get("href") or ""))
        key = program_key(url)
        if key in seen:
            continue
        seen.add(key)
        records.append(ProgramRecord(
            title=title,
            url=url,
            degree=_infer_degree(title),
            college=_nearest_heading_text(link),
            level=_infer_program_level(url),
            key=key,
        ))

    if records:
        return records

    title = _tag_text(soup.select_one("h1")) or _tag_text(soup.title)
    if not title:
        return []
    main = soup.select_one("#content, main, #textcontainer") or soup
    paragraphs = [
        _tag_text(p) for p in main.find_all("p")
        if _tag_text(p) and not _LABEL_RE.match(_tag_text(p))
    ]
    requirements = []
    for tag in main.find_all(["h2", "h3", "li"]):
        text = _tag_text(tag)
        if text and re.search(r"requirement|curriculum|plan of study|credit", text, re.I):
            requirements.append(text)
    url = source_url
    records.append(ProgramRecord(
        title=title,
        url=url,
        degree=_infer_degree(title),
        description="\n\n".join(paragraphs[:3]) or None,
        requirements=requirements,
        level=_infer_program_level(url),
        key=program_key(url or title),
    ))
    return records


def _frontmatter(data: dict) -> str:
    clean = {k: v for k, v in data.items() if v not in (None, [], "")}
    dumped = yaml.safe_dump(clean, sort_keys=False, allow_unicode=False).strip()
    return f"---\n{dumped}\n---\n\n"


def _content_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def course_card_markdown(record: CourseRecord | dict) -> str:
    """Markdown card with structured YAML fields for downstream payload."""
    data = _record_dict(record)
    title = f"{data['subject']} {data['number']}: {data['title']}"
    body_lines = [f"# {title}", ""]
    if data.get("credits"):
        body_lines.append(f"**Credits:** {data['credits']}")
    if data.get("prerequisites"):
        body_lines.append(f"**Prerequisites:** {data['prerequisites']}")
    if data.get("corequisites"):
        body_lines.append(f"**Corequisites:** {data['corequisites']}")
    if data.get("when_offered"):
        body_lines.append(f"**When offered:** {data['when_offered']}")
    if data.get("cross_listings"):
        body_lines.append(f"**Cross-listed:** {', '.join(data['cross_listings'])}")
    if data.get("description"):
        body_lines += ["", data["description"]]
    body = "\n".join(body_lines).rstrip() + "\n"
    key = data.get("key") or course_key(data["subject"], data["number"])
    url = course_record_url(data)
    frontmatter = {
        "doc_id": _doc_id(url),
        "page_type": "course",
        "key": key,
        "title": title,
        "url": url,
        "source_host": "catalog.vt.edu",
        "subdomain": "catalog",
        "subject": data["subject"],
        "course_number": data["number"],
        "course_level": data["level"],
        "credits": data.get("credits"),
        "prerequisites": data.get("prerequisites"),
        "corequisites": data.get("corequisites"),
        "when_offered": data.get("when_offered"),
        "cross_listings": data.get("cross_listings") or [],
        "content_hash": _content_hash(body),
    }
    return _frontmatter(frontmatter) + body


def program_card_markdown(record: ProgramRecord | dict) -> str:
    """Markdown card for CourseLeaf programs with frontmatter payload fields."""
    data = _record_dict(record)
    body_lines = [f"# {data['title']}", ""]
    for label, key in (
        ("Degree", "degree"),
        ("College", "college"),
        ("Department", "department"),
        ("Level", "level"),
    ):
        if data.get(key):
            body_lines.append(f"**{label}:** {data[key]}")
    if data.get("description"):
        body_lines += ["", data["description"]]
    if data.get("requirements"):
        body_lines += ["", "## Requirements", ""]
        body_lines += [f"- {item}" for item in data["requirements"]]
    body = "\n".join(body_lines).rstrip() + "\n"
    key = data.get("key") or program_key(data.get("url") or data["title"])
    frontmatter = {
        "doc_id": _doc_id(data["url"]),
        "page_type": "program",
        "key": key,
        "title": data["title"],
        "url": data["url"],
        "source_host": "catalog.vt.edu",
        "subdomain": "catalog",
        "program_name": data["title"],
        "degree_type": data.get("degree"),
        "college": data.get("college"),
        "department": data.get("department"),
        "program_level": data.get("level"),
        "requirements": data.get("requirements") or [],
        "content_hash": _content_hash(body),
    }
    return _frontmatter(frontmatter) + body


async def _render_html(urls: list[str], threshold: float = 0.5) -> dict[str, str]:
    """Browser-render CourseLeaf URLs and return {url: html}."""
    if not urls:
        return {}
    rendered: dict[str, str] = {}
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(urls=urls, config=js_run_config(threshold))
        async for result in results:
            if getattr(result, "success", False):
                rendered[result.url] = getattr(result, "html", "") or ""
            else:
                logger.warning(
                    "CourseLeaf render failed for %s: %s",
                    getattr(result, "url", ""),
                    getattr(result, "error_message", "unknown"),
                )
    return rendered


async def harvest_courseleaf_catalog(
    storage,
    target_bucket: str,
    *,
    fetch_html=None,
    threshold: float = 0.5,
    now: datetime | None = None,
) -> dict:
    """Render and store CourseLeaf course/program cards into the cleaned bucket.

    `fetch_html` is an async test seam accepting a list of URLs and returning
    {url: html}. Production uses crawl4ai's browser tier because CourseLeaf
    plain HTTP responses are bot-gated.
    """
    fetch_html = fetch_html or (lambda urls: _render_html(urls, threshold))
    now = now or datetime.now(timezone.utc)

    stats = {
        "courses_harvested": 0,
        "programs_harvested": 0,
        "pages_rendered": 0,
        "pages_failed": 0,
    }

    index_html = await fetch_html([COURSE_INDEX_URL])
    stats["pages_rendered"] += len(index_html)
    subject_urls = parse_subject_urls(index_html.get(COURSE_INDEX_URL, ""))

    subject_html = await fetch_html(subject_urls)
    stats["pages_rendered"] += len(subject_html)
    stats["pages_failed"] += max(0, len(subject_urls) - len(subject_html))
    for url, html in subject_html.items():
        for record in parse_courseblocks(html, source_url=url):
            storage.upload_document(
                course_object_key(record),
                course_card_markdown(record),
                bucket=target_bucket,
            )
            stats["courses_harvested"] += 1

    program_html = await fetch_html(list(PROGRAM_SEED_URLS))
    stats["pages_rendered"] += len(program_html)
    stats["pages_failed"] += max(0, len(PROGRAM_SEED_URLS) - len(program_html))
    seen_programs: set[str] = set()
    program_detail_urls: list[str] = []
    for url, html in program_html.items():
        for record in parse_programs(html, source_url=url):
            if record.url not in seen_programs:
                seen_programs.add(record.url)
                program_detail_urls.append(record.url)

    detail_html = await fetch_html(program_detail_urls)
    stats["pages_rendered"] += len(detail_html)
    stats["pages_failed"] += max(0, len(program_detail_urls) - len(detail_html))
    for url, html in detail_html.items():
        records = parse_programs(html, source_url=url)
        if not records:
            continue
        record = records[0]
        storage.upload_document(
            program_object_key(record),
            program_card_markdown(record),
            bucket=target_bucket,
        )
        stats["programs_harvested"] += 1

    return stats
