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
    heading_texts = [s.text for s in sections]
    assert any("## Faculty" in t for t in heading_texts)
    assert any("## Staff" in t for t in heading_texts)
    assert any("### Admin Staff" in t for t in heading_texts)


def test_split_sections_headings_path_h1():
    sections = split_sections(MULTI_HEADING_DOC)
    people_section = next(s for s in sections if "# People" in s.text and "##" not in s.text)
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


# --- is_stale_time_sensitive_page ---

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


# --- richer frontmatter fields ---

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


def test_parse_frontmatter_reads_structured_course_fields():
    md = (
        "---\n"
        "doc_id: cs3114\n"
        "url: https://catalog.vt.edu/course-descriptions/cs/\n"
        "title: 'CS 3114: Data Structures and Algorithms'\n"
        "page_type: course\n"
        "subject: CS\n"
        "course_number: '3114'\n"
        "course_level: ug\n"
        "credits: '3'\n"
        "prerequisites: CS 2114 or CS 2505.\n"
        "corequisites: MATH 2534.\n"
        "when_offered: Fall and Spring.\n"
        "cross_listings:\n"
        "  - CMDA 3114\n"
        "---\n\n# CS 3114\nbody"
    )
    fm, _ = parse_frontmatter(md)
    assert fm.page_type == "course"
    assert fm.subject == "CS"
    assert fm.course_number == "3114"
    assert fm.course_level == "ug"
    assert fm.credits == "3"
    assert fm.prerequisites == "CS 2114 or CS 2505."
    assert fm.corequisites == "MATH 2534."
    assert fm.when_offered == "Fall and Spring."
    assert fm.cross_listings == ["CMDA 3114"]


def test_parse_frontmatter_reads_structured_program_fields():
    md = (
        "---\n"
        "doc_id: cs-bs\n"
        "url: https://catalog.vt.edu/undergraduate/computer-science-bs/\n"
        "title: Computer Science, B.S.\n"
        "page_type: program\n"
        "college: College of Engineering\n"
        "program_name: Computer Science\n"
        "degree_type: BS\n"
        "---\n\n# Computer Science, B.S.\nbody"
    )
    fm, _ = parse_frontmatter(md)
    assert fm.page_type == "program"
    assert fm.college == "College of Engineering"
    assert fm.program_name == "Computer Science"
    assert fm.degree_type == "BS"
