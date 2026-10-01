# services/crawler/src/crawler/urls.py
"""Single source of truth for URL hygiene across ingestion.

Two responsibilities, deliberately split:

  is_ingestible_url(url)  — REJECT-side gate. Returns False for anything that
      should never enter the frontier (malformed hrefs, embedded schemes,
      control chars, non-http(s), no host, over-length). Used by classify_url
      so both the registry seed path and BFS expansion share one gate.

  safe_object_key(url)    — SANITIZE-side net. Turns any URL into a MinIO-legal
      object key even if a weird-but-accepted URL slips through: collapses
      consecutive slashes, strips control chars, and replaces characters MinIO
      rejects (notably ':' from an embedded 'https://'). Identity-preserving for
      already-clean VT URLs so existing object keys never churn.

The crash that motivated this module: a page linked
  https://www.pubs.ext.vt.edu/.../426-041.htmlhttps://www.pubs.ext.vt.edu/426-841.html
i.e. two URLs concatenated with no separator. urlparse saw scheme=https,
host=vt, and an embedded 'https://' sitting in the PATH. classify_url accepted
it, it was rendered, and url_to_object_key produced a key containing '://'
(consecutive slashes) which MinIO rejects as XMinioInvalidObjectName —
unhandled, it killed the whole run.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

# A normal absolute URL is well under this; anything longer is almost certainly
# a concatenation/loop artifact, not a real page.
MAX_URL_LENGTH = 2048

ALLOWED_SCHEMES = ("http", "https")

# Non-textual media/binary extensions: the RAG corpus is text only, so these are
# rejected outright at the ingestion gate. They are never rendered, downloaded,
# or stored. Documents the pipeline CAN extract text from (.pdf/.doc/.docx) are
# deliberately NOT here — those are handled by the documents pass.
REJECTED_EXTENSIONS = (
    # video — explicitly rejected: never processed
    ".mp4", ".mov", ".avi", ".wmv", ".flv", ".mkv", ".webm", ".m4v",
    ".mpeg", ".mpg", ".3gp", ".ogv", ".m2v", ".m2ts", ".ts", ".vob",
    # audio
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".oga", ".wma", ".m4a", ".aiff", ".opus",
    # images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".tif", ".tiff", ".heic", ".heif", ".avif", ".psd", ".eps",
    # archives / disk images / executables / installers
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".dmg", ".iso", ".exe", ".msi", ".bin", ".pkg", ".deb", ".rpm", ".apk",
    # binary office formats the pipeline does NOT extract (not text-usable)
    ".ppt", ".pptx", ".xls", ".xlsx", ".odp", ".ods",
    # fonts / other binaries
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".swf", ".class", ".jar", ".dll",
)

# Hostname sanity: ASCII letters/digits/dot/hyphen only (VT is all-ASCII; this
# also rejects a host field that captured junk from a malformed URL).
_HOST_RE = re.compile(r"^[a-z0-9.-]+$")

# Control characters (C0 + DEL) — never valid in a URL we want to ingest.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Object-key character allow-list. Anything outside this set is replaced with
# '_'. Existing clean VT keys use only letters/digits/'._-/' so they are
# untouched; '%' and '~' are kept so percent-encoded paths survive intact.
_KEY_ALLOWED_RE = re.compile(r"[^A-Za-z0-9._\-/%~]")
_MULTISLASH_RE = re.compile(r"/{2,}")


def _embeds_open_redirect(query: str) -> bool:
    """True if a query param VALUE is a full http(s) URL to a bare non-FQDN host.

    Catches open-redirect/proxy artifacts like ``?url=http://evil`` (host has no
    dot — a placeholder/garbage target). A param pointing at a real dotted host
    (``?next=https://login.vt.edu/sso``) is deliberately NOT rejected here; scope
    decides whether that target is in-scope.
    """
    for _key, value in parse_qsl(query, keep_blank_values=True):
        candidate = value.strip()
        if candidate.lower().startswith(("http://", "https://")):
            host = (urlsplit(candidate).hostname or "").lower()
            if not host or "." not in host or not _HOST_RE.match(host):
                return True
    return False


def is_ingestible_url(url: str) -> bool:
    """Return True only if `url` is safe to enqueue and later store.

    Rejects: non-str/empty, over-length, control chars or whitespace anywhere,
    non-http(s) schemes, missing/illegal hostname, and an embedded scheme in the
    path (the concatenated-href crash case).
    """
    if not url or not isinstance(url, str):
        return False
    if len(url) > MAX_URL_LENGTH:
        return False
    if _CONTROL_RE.search(url) or any(c.isspace() for c in url):
        return False

    try:
        parts = urlsplit(url)
    except ValueError:
        return False

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    host = (parts.hostname or "").lower()
    if not host or not _HOST_RE.match(host):
        return False

    # An embedded '://' in the PATH means two URLs were concatenated (a broken
    # href like ".../426-041.htmlhttps://www...") — the '://' would land in the
    # object key and crash storage. Only the definitive '://' is rejected: a
    # bare 'http:'/'https:' inside a path slug (/tags/http-basics,
    # /wiki/https:hypertext) is a legitimate page, not a concatenation (BUG-G).
    path_lower = parts.path.lower()
    if "://" in path_lower:
        return False

    # An open-redirect / proxy param carries a full URL as a query-param VALUE
    # (?url=http://evil). Reject one whose embedded host is a bare non-FQDN
    # (garbage/placeholder); a param aimed at a real dotted host
    # (?next=https://login.vt.edu/sso) is left for classify_url/scope to judge
    # so legitimate intra-VT redirect links are not dropped (BUG-G).
    if _embeds_open_redirect(parts.query):
        return False

    # Reject non-textual media/binaries outright: video, audio, images, archives,
    # and binary office formats the pipeline cannot turn into LLM-usable text.
    # Extractable documents (.pdf/.doc/.docx) are intentionally NOT rejected here.
    if path_lower.endswith(REJECTED_EXTENSIONS):
        return False

    return True


def _sanitize_segment(segment: str) -> str:
    """Make one path/host segment MinIO-safe; '.'/'..' become '_'."""
    cleaned = _KEY_ALLOWED_RE.sub("_", segment)
    if cleaned in (".", ".."):
        return "_"
    return cleaned


def safe_object_key(url: str, suffix: str = ".md") -> str:
    """Convert a URL to a MinIO-legal object key, sanitizing unsafe characters.

    Identity-preserving for already-clean URLs:
      https://cs.vt.edu/academics/courses.html -> cs.vt.edu/academics/courses.html.md
    Defensive for hostile ones — consecutive slashes collapse and ':' / other
    illegal chars become '_', so a key can never trip XMinioInvalidObjectName.
    """
    parts = urlsplit(url)
    host = (parts.hostname or parts.netloc or "host").lower()
    host = _sanitize_segment(host) or "host"

    path = parts.path.replace("\\", "/")
    path = _MULTISLASH_RE.sub("/", path).strip("/")
    if not path:
        path = "index"

    segments = [_sanitize_segment(s) for s in path.split("/") if s]
    path = "/".join(s for s in segments if s) or "index"

    if not path.endswith(suffix):
        path = path + suffix
    return f"{host}/{path}"
