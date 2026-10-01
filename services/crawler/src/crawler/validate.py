"""Content validation layer — pre-storage gates for all content types.

Called after crawl4ai returns, before MinIO storage. Every check that can
reject content lives here so the storage path in engine.py / documents.py
stays clean.

Two rejection outcomes:
  - BLOCKED: external block (bot challenge, login wall) — skip storage and
    mark the frontier row 'skipped' (reason "blocked:..."). The frontier is
    ephemeral, so the next scheduled run (fresh DB) re-crawls it.
  - REJECTED: permanent issue (soft 404, HTTP 404, empty content, boilerplate,
    magic byte mismatch, etc.) — mark_skipped so it is never re-crawled.

The caller (engine.py) distinguishes the two by checking the `blocked` flag
on ValidationResult.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from crawl4ai.antibot_detector import is_blocked

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_CONTENT_WORDS = 25

# Default HTML size cap — can be overridden via config.
DEFAULT_MAX_HTML_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Login wall detection ─────────────────────────────────────────────────────
# Hosts that indicate an auth redirect when they appear in redirected_url.
LOGIN_WALL_HOSTS = frozenset({
    "idp.vt.edu",
    "login.vt.edu",
    "login.microsoftonline.com",
    "login.live.com",
    "accounts.google.com",
    "shibboleth.vt.edu",
})

# HTML patterns scanned in the first 4KB of the page (speed: no full parse).
# STRONG — definitive SSO/Shibboleth/SAML markers. A match is a login wall
# regardless of how much other content the page has.
_LOGIN_STRONG_PATTERNS = [
    re.compile(r'action="[^"]*shib', re.I),
    re.compile(r'name="SAMLRequest"', re.I),
    re.compile(r'name="SAMLResponse"', re.I),
    re.compile(r"Sign in with your Virginia Tech", re.I),
    re.compile(r"Virginia Tech PID", re.I),
]
# WEAK — a bare password form / 2FA copy. These are a login WALL only when the
# form DOMINATES the page (a login stub), NOT when they're a header or intranet
# login widget on an otherwise content-rich department page (BUG-C).
_LOGIN_WEAK_PATTERNS = [
    re.compile(r'id="password"[\s>]', re.I),
    re.compile(r'<form[^>]*password', re.I),
    re.compile(r"two-factor authentication", re.I),
]
_LOGIN_SCAN_BYTES = 4096
# A page whose stripped text exceeds this many words is content-rich, so a weak
# password-form marker is treated as a widget rather than a wall (BUG-C).
_LOGIN_WALL_MAX_CONTENT_WORDS = 50

# ── Soft 404 detection ───────────────────────────────────────────────────────
# Matched against title + h1 text. Two tiers so a legitimate page whose title
# merely CONTAINS a phrase isn't dropped (BUG-B):
#
# STRONG — unambiguous "not found" markers; a match is a soft 404 regardless of
# the surrounding words.
_SOFT_404_STRONG_PATTERNS = [
    re.compile(r"page not found", re.I),
    re.compile(r"404\s+not\s+found", re.I),
    re.compile(r"\berror\s+404\b", re.I),
    re.compile(r"\bhttp\s+404\b", re.I),
]
# GONE — "moved / removed / no longer exists" markers only count when they
# describe the PAGE or its content itself, not an unrelated subject. So
# "The Department Has Moved to Goodwin Hall" (a real page) and an h1 like
# "How to appeal an Access Denied decision" are NOT rejected, while
# "This page has moved" / "The content no longer exists" still are. The
# over-broad bare "access denied" and "content moved" title matches are dropped.
_SOFT_404_GONE_PATTERN = re.compile(
    r"\b(page|pages|content|url|link|document|file|resource)\b"
    r"[\w\s,'’-]{0,30}?"
    r"\b(no longer exists|no longer available|has been removed|has moved|"
    r"could not be found|cannot be found|was not found|does not exist)\b",
    re.I,
)
# Quick regex to extract <title> and <h1> without a full HTML parse.
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.DOTALL)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")

# ── Boilerplate leak detection ───────────────────────────────────────────────
# Markers that indicate PruningContentFilter failed and the extracted text is
# pure nav/chrome. Checked against the EXTRACTED text (after fit_markdown),
# not the raw HTML — the filter should have stripped these.
_BOILERPLATE_MARKERS = (
    "Skip to main content",
    "Universal Access",
    "Skip to Content",
)

# ── Document magic byte validation ───────────────────────────────────────────
# Maps file extension to the set of puremagic extensions that are acceptable.
# If the URL ends with .pdf but puremagic detects .html, it's a mismatch.
_DOCUMENT_EXT_MAP = {
    ".pdf": {".pdf"},
    ".docx": {".docx"},
    ".doc": {".doc", ".docx"},  # some servers serve docx for .doc URLs
}


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a validation check.

    Attributes:
        valid: True if content passes validation.
        reason: Human-readable rejection reason (empty if valid).
        layer: Which validation layer produced the rejection.
        blocked: True if this is a temporary block (bot challenge, login wall)
            that should be re-crawled next run. The caller marks the frontier
            row 'skipped' (reason "blocked:..."); the ephemeral frontier means
            the next scheduled run re-crawls it.
    """
    valid: bool
    reason: str = ""
    layer: str = ""
    blocked: bool = False

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True)

    @classmethod
    def reject(cls, reason: str, layer: str, *, blocked: bool = False) -> "ValidationResult":
        return cls(valid=False, reason=reason, layer=layer, blocked=blocked)


# ── Layer 1: HTTP Response Validation ────────────────────────────────────────

def validate_http_response(
    result,
    *,
    max_html_size_bytes: int = DEFAULT_MAX_HTML_SIZE_BYTES,
) -> ValidationResult:
    """Validate HTTP response metadata: status code and HTML size.

    Rejects:
      - HTTP 4xx/5xx status codes (429 is left to is_blocked in layer 2)
      - HTML body exceeding max_html_size_bytes
    """
    status_code = getattr(result, "status_code", None)

    # Status code gate: reject 4xx/5xx except 429 (handled by is_blocked).
    if status_code is not None and 400 <= status_code < 600 and status_code != 429:
        return ValidationResult.reject(
            f"http_error_{status_code}", "http_response"
        )

    # HTML size cap.
    html = getattr(result, "html", "") or ""
    if len(html.encode("utf-8", errors="replace")) > max_html_size_bytes:
        return ValidationResult.reject("html_oversized", "http_response")

    return ValidationResult.ok()


# ── Layer 2: HTML Content Validation ─────────────────────────────────────────

def validate_html_content(result) -> ValidationResult:
    """Validate HTML page content: bot challenges, login walls, soft 404s.

    This runs BEFORE select_content (PruningContentFilter) so we reject
    obviously-bad pages without paying the extraction cost.

    Check order is intentional:
      1. Login-wall redirect — a redirect to idp.vt.edu is definitive,
         regardless of HTML size. Must run before is_blocked, which would
         otherwise catch small login pages as "near-empty content".
      2. is_blocked — bot challenges, 429, 403/503 shells, structural issues.
      3. Login-wall HTML patterns — SAML/Shibboleth form markers without a
         redirect (catches inline login forms on large pages).
      4. Soft 404 — 200 status with "not found" content.

    Rejects (blocked=True, re-crawl next run):
      - Bot challenges via crawl4ai is_blocked()
      - Login walls via redirect URL + HTML pattern scan

    Rejects (blocked=False, terminal):
      - Soft 404s: 200 response with "page not found" / "access denied" content
    """
    # 1. Login-wall redirect check (fast hostname comparison).
    login_result = _check_login_wall(result)
    if not login_result.valid:
        return login_result

    # 2. Bot challenge detection (crawl4ai v0.9.0 is_blocked).
    html = getattr(result, "html", "") or ""
    status_code = getattr(result, "status_code", None)
    error_message = getattr(result, "error_message", None)
    blocked, reason = is_blocked(status_code, html, error_message)
    if blocked:
        return ValidationResult.reject(
            f"bot_challenge:{reason}", "html_content", blocked=True
        )

    # 3. Soft 404 detection.
    soft_result = _check_soft_404(html)
    if not soft_result.valid:
        return soft_result

    return ValidationResult.ok()


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.DOTALL)


def _visible_word_count(html: str) -> int:
    """Rough count of the page's human-visible words (script/style stripped).

    Used only to decide whether a bare password form dominates the page — an
    approximation, not a precise content measure, so a full parse isn't worth
    the cost on every page.
    """
    no_code = _SCRIPT_STYLE_RE.sub(" ", html)
    return len(_TAG_STRIP_RE.sub(" ", no_code).split())


def _check_login_wall(result) -> ValidationResult:
    """Detect login walls via redirect URL and HTML patterns.

    Primary signal: result.redirected_url pointing to a known SSO host.
    Secondary signal: Shibboleth/SAML markers in the first 4KB of HTML
    (definitive). A bare password form is a wall only when it DOMINATES the
    page — a content-rich page with a header/intranet login widget is not a
    login wall (BUG-C).
    """
    redirected_url = getattr(result, "redirected_url", None) or ""
    if redirected_url:
        redirected_host = (urlparse(redirected_url).hostname or "").lower()
        if redirected_host and redirected_host in LOGIN_WALL_HOSTS:
            return ValidationResult.reject(
                f"login_wall:redirect:{redirected_host}", "html_content",
                blocked=True,
            )

    html = getattr(result, "html", "") or ""
    head = html[:_LOGIN_SCAN_BYTES]

    # Definitive SSO/SAML markers — a wall regardless of page content.
    for pattern in _LOGIN_STRONG_PATTERNS:
        if pattern.search(head):
            return ValidationResult.reject(
                "login_wall:html_pattern", "html_content", blocked=True
            )

    # Bare password form — only a wall when it dominates a content-sparse page.
    for pattern in _LOGIN_WEAK_PATTERNS:
        if pattern.search(head):
            if _visible_word_count(html) <= _LOGIN_WALL_MAX_CONTENT_WORDS:
                return ValidationResult.reject(
                    "login_wall:html_pattern", "html_content", blocked=True
                )
            # Content-rich page: it's a login widget, not a wall.
            break

    return ValidationResult.ok()


def _check_soft_404(html: str) -> ValidationResult:
    """Detect soft 404s: 200 response with 'not found' content.

    Checks title + h1 text against known soft-404 patterns. Uses regex
    extraction instead of BeautifulSoup for speed — this runs on every page.
    """
    if not html:
        return ValidationResult.ok()

    title_text = ""
    h1_text = ""

    title_match = _TITLE_RE.search(html)
    if title_match:
        title_text = _TAG_STRIP_RE.sub("", title_match.group(1)).strip()

    h1_match = _H1_RE.search(html)
    if h1_match:
        h1_text = _TAG_STRIP_RE.sub("", h1_match.group(1)).strip()

    check_text = f"{title_text} {h1_text}".strip()
    if not check_text:
        return ValidationResult.ok()

    for pattern in _SOFT_404_STRONG_PATTERNS:
        if pattern.search(check_text):
            return ValidationResult.reject("soft_404", "html_content")

    if _SOFT_404_GONE_PATTERN.search(check_text):
        return ValidationResult.reject("soft_404", "html_content")

    return ValidationResult.ok()


def has_boilerplate_leak(text: str) -> bool:
    """Check if extracted text contains boilerplate markers.

    Called AFTER select_content (PruningContentFilter) — if these markers
    survive the filter, the extraction failed and the content is pure
    nav/chrome. Returns True if the text should be rejected.
    """
    if not text:
        return False
    for marker in _BOILERPLATE_MARKERS:
        if marker in text:
            return True
    return False


# ── Layer 3: Document Validation ─────────────────────────────────────────────

def validate_document(data: bytes, url: str) -> ValidationResult:
    """Validate downloaded document bytes via magic bytes.

    Catches the case where a .pdf URL serves an HTML 404 page, or a .docx URL
    serves a ZIP bomb. Uses puremagic to deep-scan the file content.

    Returns ok() if puremagic is not installed or can't identify the file —
    the caller's extraction pass will handle those cases.
    """
    try:
        import puremagic
    except ImportError:
        logger.debug("puremagic not installed — skipping magic byte validation")
        return ValidationResult.ok()

    path = urlparse(url).path.lower()

    # Determine expected extension from URL.
    expected_ext = None
    for ext in _DOCUMENT_EXT_MAP:
        if path.endswith(ext):
            expected_ext = ext
            break

    if expected_ext is None:
        # Not a document URL we validate — let extraction handle it.
        return ValidationResult.ok()

    if not data:
        return ValidationResult.reject("empty_document_bytes", "document")

    try:
        matches = puremagic.magic_string(data, filename=path)
    except Exception as exc:
        logger.debug("puremagic could not identify %s: %s", url, exc)
        return ValidationResult.ok()

    if not matches:
        return ValidationResult.ok()

    # Check if any detected extension matches the expected one.
    acceptable_exts = _DOCUMENT_EXT_MAP[expected_ext]
    detected_exts = {getattr(m, "extension", "") for m in matches}

    if not (detected_exts & acceptable_exts):
        detected = matches[0].extension or matches[0].mime_type or "unknown"
        return ValidationResult.reject(
            f"magic_byte_mismatch:expected_{expected_ext},got_{detected}",
            "document",
        )

    return ValidationResult.ok()


# ── Layer 4: Structural Validation ───────────────────────────────────────────

def validate_catalog_card(card: dict) -> ValidationResult:
    """Validate a catalog card has minimum required metadata.

    Rejects:
      - Cards where both title AND abstract are None (no metadata found)
      - Cards where the title is the URL itself (fallback extraction failed)
    """
    title = card.get("title")
    abstract = card.get("abstract")

    if not title and not abstract:
        return ValidationResult.reject("empty_catalog_card", "structural")

    url = card.get("url", "")
    if title and url and title.strip() == url.strip():
        return ValidationResult.reject("empty_catalog_card:title_is_url", "structural")

    return ValidationResult.ok()


def validate_program_record(record, *, is_fallback: bool = False) -> ValidationResult:
    """Validate a program record has real program content.

    Rejects:
      - Records with no title or title == URL (always — structural check).
      - ONLY when ``is_fallback``: records that lack program-related keywords in
        title/description/requirements. The keyword gate is reserved for the
        parse_programs GUESSING fallback (a no-links page merely treated as a
        detail page) — the source of the 643 broken stub cards. A KNOWN
        program-detail page (is_fallback=False) is NOT keyword-gated, so a real
        program whose requirements pane failed to render is retained (BUG-D).
    """
    title = getattr(record, "title", None) or ""
    requirements = getattr(record, "requirements", None) or []
    description = getattr(record, "description", None) or ""
    url = getattr(record, "url", "") or ""

    if not title or title.strip() == url.strip():
        return ValidationResult.reject("broken_program_record:no_title", "structural")

    if is_fallback:
        body_text = f"{title} {description} {' '.join(requirements)}".lower()
        if not re.search(r"requirement|curriculum|plan of study|credit", body_text, re.I):
            return ValidationResult.reject(
                "broken_program_record:no_program_keywords", "structural"
            )

    return ValidationResult.ok()
