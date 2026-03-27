"""Clean boilerplate from crawled VT CS department markdown pages."""

from __future__ import annotations

import re


# --- Single-line patterns to remove ---

_SKIP_LINK_RE = re.compile(r"^\s*\*\s*\[Skip to (main content|search)\]")
_VT_HEADER_RE = re.compile(
    r"(Virginia Tech\u00ae? home|Universal Access|accessibility_icon|"
    r"Close Universal Access dialog|Report a barrier|Accessibility portal|"
    r"Pause all background videos|Underline all links|toggle_slider)"
)
_TRACKING_PIXEL_RE = re.compile(r"!\[.*?\]\(https?://bat\.bing\.com/")
_CMS_ARTIFACT_RE = re.compile(
    r"^\s*\*\s*(Bio|General|Article|Redirect) Item \1 Item\s*$"
)
_COPYRIGHT_RE = re.compile(r"^\u00a9 \d{4} Virginia Polytechnic")
_VT_LOGO_RE = re.compile(r"!\[Virginia Tech logo\]")
_MAP_IMAGE_RE = re.compile(r"!\[Map of Virginia")
_SEARCH_HELP_RE = re.compile(
    r"^##\s*(Search Tips|Search Help|More search options|People Results|VT News Results)"
)
_BREADCRUMB_RE = re.compile(
    r"^\s*\d+\.\s*\[?\s*(College of Engineering|Department of Computer Science)\s*/?\s*\]?"
)
_HEADER_LINK_RE = re.compile(
    r"^\s*\*?\s*\[?\s*(Apply|Visit|Give|Shop|Future Students|Current Students|"
    r"Parents and Families|Faculty and Staff|Alumni|Industry and Partners|"
    r"Hokie Sports Shop|Hokie Shop|Hokie Gear|Hokie License Plates|"
    r"Hokie Spa|Canvas|Jobs|Academic Calendar|Tuition|"
    r"University Photo Library|University Libraries|Undergraduate Course Catalog|"
    r"TLOS Course Catalog|Ensemble support|IT Support|VT Policy Library|"
    r"University Status|Principles of Community|Privacy Statement|Acceptable Use|"
    r"We Remember|Accessibility|Consumer Information|Cost & Aid|SAFE at VT|"
    r"Policies|Equal Opportunity|WVTF|University Bookstore|Jobs at Virginia Tech)\s*\]?"
)
_SOCIAL_ICON_RE = re.compile(
    r"^\s*\*\s*\[(Instagram|Facebook|Linked-?In|Threads|Youtube|X|Blue Sky)\]"
    r"\(https?://"
)

# Lines containing these exact strings are nav/search chrome, not content
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
    "See more people results",
    "See more news results",
]

_VT_FOOTER_LINKS = [
    "Get Directions",
    "See All Locations",
    "Contact Virginia Tech",
]

# --- Block markers ---
_DEPT_FOOTER_MARKER = "### Follow Computer Science"
_EXPLORE_MARKER_RE = re.compile(r"^Explore\s*$")


_CMS_ERROR_RE = re.compile(
    r"^#\s+Resource at '.*?' not found",
    re.MULTILINE,
)


def is_error_page(markdown: str) -> bool:
    """Return True if the markdown is a CMS 'Resource not found' error page.

    The VT CMS returns HTTP 200 with a body containing a specific error heading
    when a page has been deleted or moved. These pages have no useful content.
    """
    body = markdown
    if markdown.startswith("---"):
        end = markdown.find("---", 3)
        if end != -1:
            body = markdown[end + 3:]
    return bool(_CMS_ERROR_RE.search(body[:500]))


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

        # --- Block state machines ---

        if in_explore_block:
            # Explore sidebar ends when we hit a real content heading
            if re.match(r"^#{1,2}\s+\S", stripped):
                in_explore_block = False
                # fall through to process this line normally
            elif (
                stripped.startswith("* [")
                or stripped.startswith("* [ ")
                or stripped.startswith("*  ")
                or stripped.startswith("* Current page:")
                or not stripped
            ):
                continue
            else:
                in_explore_block = False
                # fall through

        if in_search_block:
            if stripped.startswith("#") and not _SEARCH_HELP_RE.match(stripped):
                in_search_block = False
                # fall through
            else:
                continue

        if in_dept_footer or in_vt_footer:
            continue

        # --- Detect block starts ---

        if _EXPLORE_MARKER_RE.match(stripped):
            in_explore_block = True
            continue

        if stripped == "Search":
            in_search_block = True
            continue

        if stripped.startswith(_DEPT_FOOTER_MARKER):
            in_dept_footer = True
            continue

        if _VT_LOGO_RE.search(line) or _MAP_IMAGE_RE.search(line):
            in_vt_footer = True
            continue

        # --- Single-line removals ---

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
        if re.match(
            r"^\s*\[?\s*Computer Science\s*\]?\s*\(https://website\.cs\.vt\.edu/?\)\s*$",
            line,
        ):
            continue
        if stripped == "Menu":
            continue
        if stripped.startswith("![](data:image/svg+xml"):
            continue
        if "[Intranet]" in line and "(Internal)" in line:
            continue
        # "search" alone on a line (search tab label)
        if stripped == "search":
            continue
        # Clear search box link
        if "Clear search box" in line:
            continue
        # Apparel/merchandise description lines from Shop dropdown
        if stripped in (
            "Apparel, clothing, gear and merchandise",
            "University Bookstore, merchandise and gifts",
            "Everything you need to know about Hokie gear",
            "Part of every Virginia Tech plate purchase funds scholarships",
        ):
            continue

        cleaned.append(line)

    # Collapse excessive blank lines (3+ consecutive → 2)
    result_body = "\n".join(cleaned)
    result_body = re.sub(r"\n{4,}", "\n\n\n", result_body)

    # Extract and format structured bio contact info if present
    result_body = _format_bio_contact(result_body)

    return frontmatter + result_body


def _format_bio_contact(body: str) -> str:
    """Extract scattered contact details and consolidate into a Contact section.

    Bio pages rendered from raw_markdown have email, phone, and office info
    scattered among image markup and SVG icons. This function finds those details
    and inserts a clean **Contact Information** block after the name/title heading.
    """
    # Extract email addresses (mailto: links or bare @vt.edu addresses)
    emails = re.findall(
        r"\[([^\]]*@[^\]]+)\]\(mailto:[^)]+\)", body
    )
    if not emails:
        emails = re.findall(r"[\w.+-]+@(?:cs\.)?vt\.edu", body)

    # Extract phone numbers
    phones = re.findall(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}", body)

    # Extract office/room info (e.g., "2202 Kraft Drive" or "Torgersen 2160")
    office_patterns = [
        re.findall(r"(?:Office|Room|Rm\.?)\s*:?\s*([^\n,]+)", body, re.IGNORECASE),
        re.findall(r"(\d+\s+[\w\s]+(?:Hall|Drive|Building|Center|Ave|Blvd)(?:\s+\w+)?)", body),
    ]
    offices = []
    for matches in office_patterns:
        offices.extend(m.strip() for m in matches if m.strip())

    if not emails and not phones:
        return body

    # Build a clean contact block
    contact_lines = ["\n\n**Contact Information:**"]
    seen = set()
    for email in emails:
        if email not in seen:
            contact_lines.append(f"- Email: {email}")
            seen.add(email)
    for phone in phones:
        if phone not in seen:
            contact_lines.append(f"- Phone: {phone}")
            seen.add(phone)
    for office in offices[:2]:  # Limit to avoid false positives
        if office not in seen:
            contact_lines.append(f"- Office: {office}")
            seen.add(office)

    contact_block = "\n".join(contact_lines) + "\n"

    # Insert after the first heading (the person's name)
    # Look for the end of the title/position block (first blank line after headings)
    heading_match = re.search(r"^(#\s+.+\n(?:.*\S.*\n)*)", body, re.MULTILINE)
    if heading_match:
        insert_pos = heading_match.end()
        return body[:insert_pos] + contact_block + body[insert_pos:]

    # Fallback: prepend to body
    return contact_block + body


def build_department_info_doc() -> str:
    """Generate a reference document with CS department institutional info.

    This is stored once in the cleaned bucket so RAG can find campus addresses,
    phone numbers, and social links even though they're stripped from individual pages.
    """
    return """\
---
url: '_department-info'
title: 'CS Department Contact Information and Locations'
crawl_depth: 0
crawl_timestamp: 'generated'
---

# Computer Science Department - Contact Information and Locations

## Social Media
- Facebook: https://facebook.com/VT.ComputerScience
- X (Twitter): https://x.com/vt_cs
- Instagram: https://www.instagram.com/vt_cs/
- LinkedIn: https://www.linkedin.com/company/vt-cs/
- YouTube: https://www.youtube.com/@VTEngineering/featured

## Blacksburg Campus
Torgersen Hall RM 3160
620 Drillfield Dr.
Blacksburg, VA 24061
- Undergraduate: (540) 231-6931
- Graduate: (540) 231-0746

## Institute for Advanced Computing
3625 Potomac Ave.
Alexandria, VA 22305
- M.S. and Ph.D.: (703) 538-8370
- M.Eng: (703) 538-3768

## Virginia Tech Research Center
900 N. Glebe Rd.
Arlington, VA 22203
- Phone: (571) 858-3000

## Part of College of Engineering
Virginia Tech College of Engineering: https://eng.vt.edu/
"""
