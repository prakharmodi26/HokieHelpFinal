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
from urllib.parse import unquote, urljoin, urlparse

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

from crawler.render import (
    DEFAULT_RENDER_CHUNK_SIZE,
    DEFAULT_RENDER_IDLE_TIMEOUT,
    js_run_config,
    render_in_chunks,
)
from crawler.validate import validate_program_record
from crawler.program_content import requirements_blocks

logger = logging.getLogger(__name__)
CATALOG_BASE_URL = "https://catalog.vt.edu"
COURSE_INDEX_URL = f"{CATALOG_BASE_URL}/course-descriptions/"
# program-explorer flattens every degree program (undergrad + grad, ~622) into
# one rendered page; the minors and graduate-certificate hubs cover the program
# types the explorer omits. The bare /undergraduate/ and /graduate/ indexes are
# NOT seeded — they only expose depth-2 college hubs (0 program-detail links).
PROGRAM_SEED_URLS = (
    f"{CATALOG_BASE_URL}/program-explorer/",
    f"{CATALOG_BASE_URL}/undergraduate/minors/",
    f"{CATALOG_BASE_URL}/graduate/graduate-certificates/",
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
    r"Pre-?/?Co-?requisite(?:\(s\))?|When Offered|"
    # VT's live template uses "Course Crosslist:"; older snapshots used
    # "Cross-listed with:" / "Also offered as:". Match all of them.
    r"Course Cross-?list(?:ing)?(?:\(s\))?|Cross-?list(?:ing|ed)?(?:\s+with)?|"
    r"Cross[- ]Listing(?:\(s\))?|Also Listed As|Also offered as|"
    r"Also Offered As)\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_EDITION_RE = re.compile(r"\b(\d{4}-\d{4})\b")


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
    # Any other labelled field the catalog publishes that has no dedicated slot
    # — e.g. "Pathway Concept Area(s)", "Instructional Contact Hours",
    # "Repeatability". Captured verbatim ({label: value}) so nothing the page
    # shows is dropped from the embedded card.
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramRecord:
    title: str
    url: str
    degree: str | None = None
    college: str | None = None
    department: str | None = None
    description: str | None = None
    requirements: list[str] = field(default_factory=list)
    requirements_markdown: bool = False
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
    if label.startswith("cross") or label.startswith("also") or "crosslist" in label:
        return "cross_listings", value
    return None, None


def _extract_cross_listings(value: str) -> list[str]:
    listings: list[str] = []
    for match in _COURSE_CODE_RE.finditer(value.upper()):
        item = f"{match.group('subject')} {match.group('number')}{match.group('suffix')}"
        if item not in listings:
            listings.append(item)
    return listings


def _has_label_strong(tag: Tag) -> bool:
    """Return True if the tag's first non-whitespace content is a <strong> ending in ':'."""
    strong = tag.find("strong")
    if strong is None:
        return False
    strong_text = (strong.get_text(strip=True) or "")
    return strong_text.endswith(":")


def parse_courseblocks(html: str, source_url: str | None = None) -> list[CourseRecord]:
    """Parse CourseLeaf courseblock HTML into structured course records.

    Handles both the stock CourseLeaf template (uses .courseblocktitle /
    .courseblockdesc class names on <p> elements) and VT's customised template
    which uses only <div class="cols noindent"> for the title and plain
    <div class="noindent"> for the body (no .courseblocktitle/.courseblockdesc).
    """
    soup = BeautifulSoup(html or "", "html.parser")
    records: list[CourseRecord] = []
    blocks = soup.select(".courseblock")
    if not blocks:
        blocks = soup.find_all(class_=lambda c: c and "courseblock" in str(c).split())

    for block in blocks:
        # --- locate the title element ---
        # VT custom template: <div class="cols noindent"> holds the course code.
        # Stock CourseLeaf: <p class="courseblocktitle"> holds the course code.
        title_tag = block.select_one("div.cols.noindent, div.noindent.cols")
        if title_tag is None:
            title_tag = block.select_one(".courseblocktitle")
        if title_tag is None:
            title_tag = block.find(["p", "div", "span"], class_=re.compile("title"))

        header = _parse_course_header(_tag_text(title_tag or block))
        if not header:
            continue

        description_parts: list[str] = []
        extra_fields: dict[str, str] = {}   # Pathway, Contact Hours, etc.
        fields: dict[str, str | list[str] | None] = {
            "prerequisites": None,
            "corequisites": None,
            "when_offered": None,
            "cross_listings": [],
        }

        description_found = False

        for child in block.find_all(["p", "div"], recursive=False):
            if child is title_tag:
                continue
            classes = set(child.get("class") or [])

            # Skip the VT "cols noindent" title div if we encounter it again
            # (shouldn't happen but guard anyway).
            if "cols" in classes and "noindent" in classes:
                continue

            text = _tag_text(child)
            if not text:
                # Ignore empty spacer divs (VT template has several).
                continue

            # --- VT custom template body divs (class="noindent", no stock classes) ---
            # VT template uses <div class="noindent"> for both labelled fields and
            # plain descriptions. Stock CourseLeaf uses <p class="courseblockextra
            # noindent"> for labelled extras — these must fall through to the stock
            # path below (they carry no <strong> wrapper, just plain "Label: value"
            # text). Detect VT-only divs by requiring: tag is <div>, class list
            # contains "noindent", no stock indicator class present.
            _stock_classes = {"courseblockextra", "courseblockdesc", "courseblocktitle"}
            is_vt_noindent = (
                child.name == "div"
                and "noindent" in classes
                and "cols" not in classes
                and _stock_classes.isdisjoint(classes)
            )
            if is_vt_noindent:
                if _has_label_strong(child):
                    # This is a labelled field: <strong>Label:</strong> value
                    # Build a plain "Label: value" string for _split_labelled_extra.
                    key, value = _split_labelled_extra(text)
                    if key == "cross_listings" and value:
                        fields[key] = _extract_cross_listings(value)
                    elif key and value:
                        fields[key] = value
                    else:
                        # Unknown labelled field (Pathway Concept Area(s),
                        # Instructional Contact Hours, Repeatability, ...).
                        # _has_label_strong guarantees the first <strong> ends
                        # in ':', so splitting on the first colon yields a clean
                        # label/value. Captured verbatim so nothing is dropped.
                        label, _, value = text.partition(":")
                        label = _clean_text(label)
                        value = _clean_text(value)
                        if label and value:
                            extra_fields[label] = value
                else:
                    # Plain description paragraph — take the first non-empty one.
                    if not description_found:
                        description_parts.append(text)
                        description_found = True
                continue

            # --- Stock CourseLeaf template (p.courseblocktitle / p.courseblockextra) ---
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
            extras=dict(extra_fields),
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


_SKIP_NAV_RE = re.compile(r"skip to (?:main )?content", re.I)
_BRANDING_RE = re.compile(r"academic catalog", re.I)
_REQ_KEYWORD_RE = re.compile(r"requirement|curriculum|plan of study|credit", re.I)
_FOOTNOTE_RE = re.compile(r"^(?:\d+\s|\d+\.|\*|\u2020|Note:)", re.I)
_TABLE_CONTEXT_RE = re.compile(r"curriculum|requirement|plan|course|credit|degree", re.I)
_REQUIREMENT_PANE_ID_RE = re.compile(
    r"(?:requirement|curriculum|planofstudy).*textcontainer$", re.I
)


def _program_title(soup: BeautifulSoup) -> str:
    """Return the real CourseLeaf program title.

    CourseLeaf renders the program name in ``h1.page-title``. The first ``<h1>``
    in the DOM is site branding (``#site-title`` = "2026-2027 Academic
    Catalog"); "Skip to Content" is a separate skip-nav anchor. Both must be
    ignored — historically the fallback grabbed the first ``<h1>`` and produced
    the 643 mis-titled ("Skip to Content" / "Academic Catalog") program cards.
    """
    titled = _tag_text(soup.select_one("h1.page-title"))
    if titled:
        return titled
    for h1 in soup.find_all("h1"):
        if (h1.get("id") or "") == "site-title":
            continue
        text = _tag_text(h1)
        if not text or _SKIP_NAV_RE.search(text) or _BRANDING_RE.search(text):
            continue
        return text
    return _tag_text(soup.title)


def program_requirement_containers(container: Tag) -> list[Tag]:
    """Return CourseLeaf requirement/curriculum panes in DOM order.

    VT's live 2026-2027 template does not use one stable
    ``#requirementstextcontainer``.  Examples include
    ``#programcurriculumtextcontainer`` and
    ``#graduationrequirementstextcontainer``; some pages use plan-of-study
    variants.  Treat those panes as the source of record and intentionally do
    not include ``#roadmaptextcontainer`` to avoid embedding sample-semester
    roadmap duplicates as degree requirements.
    """
    panes: list[Tag] = []
    seen: set[int] = set()
    for tag in container.find_all(True):
        tag_id = str(tag.get("id") or "")
        if not tag_id or not _REQUIREMENT_PANE_ID_RE.search(tag_id):
            continue
        marker = id(tag)
        if marker in seen:
            continue
        seen.add(marker)
        panes.append(tag)
    return panes


def parse_program_detail(
    html: str, source_url: str = CATALOG_BASE_URL, *, is_fallback: bool = False
) -> ProgramRecord | None:
    """Parse a single CourseLeaf program-detail page into one ProgramRecord.

    Targets the real title (``h1.page-title``) and content container
    (``#contentarea``): the description comes from the Overview tab pane
    (``#textcontainer``) and the plan-of-study from the Requirements tab pane
    (``#requirementstextcontainer``). Falls back to generic selectors
    (``#content``/``main``) for synthetic/non-CourseLeaf pages.

    Returns ``None`` when no real title is found or the record fails program
    validation — so empty JS shells and non-program pages are dropped rather
    than stored as stubs.

    ``is_fallback`` is forwarded to validate_program_record. It defaults to
    False for direct callers (harvest of a KNOWN program-detail URL): those
    pages are validated structurally only, so a real program whose
    ``#requirementstextcontainer`` failed to render (JS hiccup) and whose
    Overview prose lacks the requirement/curriculum/credit keywords is retained
    instead of being silently dropped (BUG-D). Only the parse_programs guessing
    fallback (a no-links page we're merely treating AS a detail page) passes
    is_fallback=True to keep the keyword gate that kills stub cards.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    title = _program_title(soup)
    if not title:
        return None

    content = (
        soup.select_one("#contentarea")
        or soup.select_one("#content, main, #textcontainer")
        or soup
    )

    # Description: prefer the Overview pane; drop short nav labels ("Our
    # Website", "Disciplinary Areas:") so the description carries real prose.
    desc_source = content.select_one("#textcontainer") or content
    paragraphs = [
        text
        for p in desc_source.find_all("p")
        if (text := _tag_text(p)) and len(text) > 25 and not _LABEL_RE.match(text)
    ]

    # Preserve actual curriculum tables AND surrounding qualifiers/footnotes.
    # A heading/list-only scan silently dropped required courses and alternatives.
    # The live VT template uses several requirement pane IDs, not just the stock
    # #requirementstextcontainer, so collect all requirement/curriculum panes.
    req_panes = program_requirement_containers(content)
    requirements: list[str] = []
    for req_pane in (req_panes or [content]):
        requirements.extend(requirements_blocks(req_pane))
    if not paragraphs and not requirements:
        return None  # Empty browser/challenge shell is not a replacement source.
    if re.search(r"page not found|access denied|just a moment|^loading(?:\.*)?$", title, re.I):
        return None

    record = ProgramRecord(
        title=title,
        url=source_url,
        degree=_infer_degree(title),
        description="\n\n".join(paragraphs[:3]) or None,
        requirements=requirements,
        requirements_markdown=True,
        level=_infer_program_level(source_url),
        key=program_key(source_url or title),
    )
    result = validate_program_record(record, is_fallback=is_fallback)
    if not result.valid:
        # WARNING (not DEBUG) so dropped programs are visible in run logs — a
        # render flake that costs a whole program should not hide silently.
        logger.warning(
            "Rejected program detail record (%s): %s", result.reason, source_url
        )
        return None
    return record


def _is_program_detail_path(path: str) -> bool:
    """True for a program-DETAIL path — i.e. a leaf program page, not a hub.

    Real program details live at depth >= 4 (``/<level>/<college>/<dept>/
    <program>/`` or ``/<level>/<section>/<program>/``); every structural hub is
    shallower — the bare level index (depth 2), college hubs and
    ``/graduate/degree-programs/`` / ``/undergraduate/minors/`` (depth 3). The
    depth-4 floor therefore excludes sidebar-nav hubs that would otherwise be
    stored as bogus program cards.
    """
    return (
        ("/undergraduate/" in path or "/graduate/" in path)
        and path.count("/") >= 4
    )


def parse_program_links(html: str, source_url: str = CATALOG_BASE_URL) -> list[str]:
    """Discover program-detail URLs from a CourseLeaf listing/seed page.

    The ``/program-explorer/`` app renders every degree program (undergrad +
    grad, ~622) as a plain ``<a>`` inside ``ul.isotope li.item`` — so prefer
    that structure, which yields the full flat list in one render (the bare
    ``/undergraduate/`` and ``/graduate/`` indexes only expose depth-2 college
    hubs and would need recursion). Fall back to a generic scan for
    program-detail-shaped hrefs on non-explorer pages (minors/certificate hubs,
    synthetic listings). URL fragments (e.g. ``#contentarea``) are stripped and
    duplicates removed — the explorer lists each program twice (grid + list).
    """
    soup = BeautifulSoup(html or "", "html.parser")
    anchors = soup.select("ul.isotope li.item a[href]")
    if not anchors:
        anchors = soup.select("a[href]")

    urls: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        raw = str(anchor.get("href") or "")
        if not raw:
            continue
        url = urljoin(source_url, raw).partition("#")[0]
        if not _is_program_detail_path(urlparse(url).path.lower()):
            continue
        key = program_key(url)
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls


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

    # No program links on the page — treat it as a program-detail page and
    # extract one record via the CourseLeaf-aware detail parser (which targets
    # h1.page-title / #contentarea and validates real program content). This is
    # the GUESSING path, so keep the keyword gate (is_fallback=True) that drops
    # non-program pages misidentified as details (the 643 stub-card source).
    detail = parse_program_detail(html, source_url, is_fallback=True)
    return [detail] if detail is not None else []


def _frontmatter(data: dict) -> str:
    clean = {k: v for k, v in data.items() if v not in (None, [], "")}
    dumped = yaml.safe_dump(clean, sort_keys=False, allow_unicode=False).strip()
    return f"---\n{dumped}\n---\n\n"


def _content_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _program_total_credits(data: dict) -> str | None:
    """Extract an explicit total-credit requirement from rendered source text.

    This is a source-grounded retrieval aid, not an inference: only emit a
    summary when the requirements text itself says "Total Credits ... N" or
    "Complete N credit hours".  Putting the total near the top keeps credit
    questions from retrieving long elective tables ahead of the answer.
    """
    source = "\n".join(str(item) for item in (data.get("requirements") or []))
    for pattern in (
        r"\|\s*Total Credits\s*\|[^\n]*?\|\s*(\d{2,3})\s*\|",
        r"\bTotal Credits\b[^\n]{0,120}\b(\d{2,3})\b",
        r"\bComplete\s+(\d{2,3})\s+credit hours\b",
    ):
        match = re.search(pattern, source, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def course_card_markdown(record: CourseRecord | dict, catalog_edition: str | None = None) -> str:
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
    # Every other labelled field the catalog publishes (Pathway Concept Area(s),
    # Instructional Contact Hours, Repeatability, ...) — render verbatim so it is
    # part of the embedded text, not just metadata.
    extras = data.get("extras") or {}
    for label, value in extras.items():
        body_lines.append(f"**{label}:** {value}")
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
        "extra_fields": extras,
        "content_hash": _content_hash(body),
    }
    if catalog_edition:
        frontmatter["catalog_edition"] = catalog_edition
    return _frontmatter(frontmatter) + body


def program_card_markdown(record: ProgramRecord | dict, catalog_edition: str | None = None) -> str:
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
    if total_credits := _program_total_credits(data):
        body_lines.append(
            f"**Total Credits:** {total_credits}. Students must complete "
            f"{total_credits} credit hours for this program to graduate."
        )
    if data.get("description"):
        body_lines += ["", data["description"]]
    if data.get("requirements"):
        body_lines += ["", "## Requirements", ""]
        if data.get("requirements_markdown"):
            body_lines.append("\n\n".join(data["requirements"]))
        else:
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
        # Do not duplicate full requirements/tables in YAML frontmatter. The
        # compatible deployed ingestor parses the frontmatter delimiter with a
        # simple string search, so markdown table separators inside YAML lists
        # can truncate metadata and collapse doc_id to the empty-URL fallback.
        # Requirements remain losslessly in the markdown body for retrieval.
        "content_hash": _content_hash(body),
    }
    if catalog_edition:
        frontmatter["catalog_edition"] = catalog_edition
    return _frontmatter(frontmatter) + body


async def _render_html(urls: list[str], threshold: float = 0.5) -> dict[str, str]:
    """Browser-render CourseLeaf URLs and return {url: html}.

    Buffered helper for the small index/seed fetches. Delegates to
    ``render_in_chunks`` so even these short fetches get the chunk watchdog;
    the large subject/program-detail phases stream through ``render_in_chunks``
    directly so cards persist per chunk (see ``harvest_courseleaf_catalog``).
    """
    rendered: dict[str, str] = {}
    async for url, html in render_in_chunks(list(urls), js_run_config(threshold)):
        rendered[url] = html
    return rendered


async def _stream_from_fetch(fetch_html, urls: list[str]):
    """Adapt a dict-returning ``fetch_html`` test seam into a (url, html)
    stream so the harvest's per-chunk write path is exercised identically by
    tests and production."""
    rendered = await fetch_html(list(urls))
    for url, html in rendered.items():
        yield url, html


def detect_catalog_edition(html: str) -> str | None:
    """Extract the catalog edition (e.g. ``"2026-2027"``) from rendered
    CourseLeaf HTML.

    CourseLeaf renders the edition in the site title ("2026-2027 Academic
    Catalog"); fall back to ``<title>`` then a whole-document scan. Returns
    ``None`` when no ``YYYY-YYYY`` token is present. Used to warn when the live
    catalog year drifts from the configured ``CATALOG_EDITION``.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for selector in ("#site-title", "title"):
        match = _EDITION_RE.search(_tag_text(soup.select_one(selector)))
        if match:
            return match.group(1)
    match = _EDITION_RE.search(soup.get_text(" ", strip=True))
    return match.group(1) if match else None


async def harvest_courseleaf_catalog(
    storage,
    target_bucket: str,
    *,
    catalog_edition: str | None = None,
    fetch_html=None,
    render_stream=None,
    threshold: float = 0.5,
    chunk_size: int = DEFAULT_RENDER_CHUNK_SIZE,
    idle_timeout: float = DEFAULT_RENDER_IDLE_TIMEOUT,
    max_retries: int = 0,
    now: datetime | None = None,
) -> dict:
    """Render and store CourseLeaf course/program cards into the catalog bucket.

    The two large phases — subject pages (~140) and program-detail pages
    (~640) — stream through ``render_in_chunks``: each page's cards are written
    to MinIO as soon as that page renders, so a killed run keeps every chunk it
    already finished (no all-or-nothing buffer), per-chunk logging gives live
    progress, and a single stale/mismatched/hung URL stalls at most its own
    chunk (the render watchdog), never the whole harvest. This job re-renders
    everything on a clean run by design — it runs once per catalog edition and
    must re-capture changed courses — so there is no skip-existing step.

    Test seams: ``fetch_html`` accepts a URL list and returns {url: html} (used
    for the small index/seed fetches, and adapted into a stream for the big
    phases when ``render_stream`` is not given). ``render_stream`` overrides the
    streaming renderer directly with an ``(urls) -> AsyncIterator[(url, html)]``
    factory. Production uses crawl4ai's browser tier because CourseLeaf plain
    HTTP responses are bot-gated.

    `catalog_edition` (e.g. "2026-2027") is injected into each card's frontmatter
    so downstream consumers can filter by semester. When omitted the field is absent.
    """
    test_mode = fetch_html is not None
    fetch_html = fetch_html or (lambda urls: _render_html(urls, threshold))
    now = now or datetime.now(timezone.utc)

    def stream(urls):
        if render_stream is not None:
            return render_stream(urls)
        if test_mode:
            return _stream_from_fetch(fetch_html, urls)
        return render_in_chunks(
            list(urls), js_run_config(threshold),
            chunk_size=chunk_size, idle_timeout=idle_timeout,
            max_retries=max_retries,
        )

    stats = {
        "courses_harvested": 0,
        "programs_harvested": 0,
        "pages_rendered": 0,
        "pages_failed": 0,
    }

    # 1. Course-descriptions index (small, buffered) -> subject page URLs.
    index_html = await fetch_html([COURSE_INDEX_URL])
    stats["pages_rendered"] += len(index_html)
    subject_urls = parse_subject_urls(index_html.get(COURSE_INDEX_URL, ""))

    # Edition sanity check: cards are always tagged with the configured
    # CATALOG_EDITION, but warn loudly if the LIVE catalog year differs — that
    # means VT rolled the edition and CATALOG_EDITION was not bumped, so cards
    # would be mis-tagged.
    detected_edition = detect_catalog_edition(index_html.get(COURSE_INDEX_URL, ""))
    if detected_edition and catalog_edition and detected_edition != catalog_edition:
        logger.warning(
            "Catalog edition mismatch: live catalog is %s but CATALOG_EDITION "
            "is %s — cards tagged %s. Bump CATALOG_EDITION if VT rolled the year.",
            detected_edition, catalog_edition, catalog_edition,
        )
    elif detected_edition:
        logger.info("Detected live catalog edition: %s", detected_edition)

    # 2. Subject pages: render and write course cards per chunk.
    rendered_subjects = 0
    async for url, html in stream(subject_urls):
        rendered_subjects += 1
        for record in parse_courseblocks(html, source_url=url):
            storage.upload_document(
                course_object_key(record),
                course_card_markdown(record, catalog_edition=detect_catalog_edition(html) or catalog_edition),
                bucket=target_bucket,
            )
            stats["courses_harvested"] += 1
    stats["pages_rendered"] += rendered_subjects
    stats["pages_failed"] += max(0, len(subject_urls) - rendered_subjects)

    # 3. Program seed/listing pages (small, buffered) -> detail URLs.
    program_html = await fetch_html(list(PROGRAM_SEED_URLS))
    stats["pages_rendered"] += len(program_html)
    stats["pages_failed"] += max(0, len(PROGRAM_SEED_URLS) - len(program_html))
    seen_programs: set[str] = set()
    program_detail_urls: list[str] = []
    for url, html in program_html.items():
        for detail_url in parse_program_links(html, source_url=url):
            if detail_url not in seen_programs:
                seen_programs.add(detail_url)
                program_detail_urls.append(detail_url)

    # 4. Program detail pages: render and write program cards per chunk. Uses
    # the detail parser directly (NOT parse_programs) so a detail page's own
    # sidebar program links can't be mistaken for a listing.
    rendered_details = 0
    async for url, html in stream(program_detail_urls):
        rendered_details += 1
        record = parse_program_detail(html, source_url=url)
        if record is None:
            continue
        storage.upload_document(
            program_object_key(record),
            program_card_markdown(record, catalog_edition=detect_catalog_edition(html) or catalog_edition),
            bucket=target_bucket,
        )
        stats["programs_harvested"] += 1
    stats["pages_rendered"] += rendered_details
    stats["pages_failed"] += max(0, len(program_detail_urls) - rendered_details)

    return stats
