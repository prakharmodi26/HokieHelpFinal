"""Tests for crawler.urls — the URL hygiene gate and object-key sanitizer."""

from crawler.urls import MAX_URL_LENGTH, is_ingestible_url, safe_object_key

# The exact concatenated-href URL that killed the 16.5h run.
CRASH_URL = (
    "https://www.pubs.ext.vt.edu/content/pubs_ext_vt_edu/en/426/426-041/"
    "426-041.htmlhttps://www.pubs.ext.vt.edu/426/426-841/426-841.html"
)


# --- is_ingestible_url: reject side ---

def test_rejects_the_exact_crash_url():
    """ISC-2: embedded 'https://' in the path is rejected."""
    assert is_ingestible_url(CRASH_URL) is False


def test_rejects_embedded_scheme_variants():
    assert is_ingestible_url("https://x.vt.edu/a.htmlhttp://y.vt.edu/b") is False
    assert is_ingestible_url("https://x.vt.edu/path/https://evil") is False


def test_rejects_non_http_schemes():
    """ISC-3."""
    for url in (
        "javascript:alert(1)",
        "mailto:dean@vt.edu",
        "data:text/html,<h1>x</h1>",
        "ftp://files.vt.edu/x",
        "tel:+15405551234",
    ):
        assert is_ingestible_url(url) is False, url


def test_rejects_missing_hostname():
    """ISC-4."""
    assert is_ingestible_url("https:///path-only") is False
    assert is_ingestible_url("http://") is False


def test_rejects_video_and_nontext_media():
    """RAG corpus is text-only: video/audio/image/archive/binary URLs are rejected."""
    for url in (
        "https://video.vt.edu/media/lecture.mp4",
        "https://x.vt.edu/clip.MOV",          # case-insensitive
        "https://x.vt.edu/talk.webm",
        "https://x.vt.edu/audio/podcast.mp3",
        "https://x.vt.edu/img/logo.png",
        "https://x.vt.edu/photo.jpeg",
        "https://x.vt.edu/data/archive.zip",
        "https://x.vt.edu/slides/deck.pptx",
        "https://x.vt.edu/sheet.xlsx",
        "https://x.vt.edu/install.dmg",
    ):
        assert is_ingestible_url(url) is False, url


def test_allows_html_and_extractable_documents():
    """HTML pages and text-extractable documents (pdf/doc/docx) still pass."""
    for url in (
        "https://cs.vt.edu/academics/index.html",
        "https://cs.vt.edu/about",            # no extension
        "https://x.vt.edu/handbook.pdf",
        "https://x.vt.edu/form.docx",
        "https://x.vt.edu/legacy.doc",
    ):
        assert is_ingestible_url(url) is True, url


def test_rejects_control_chars_and_whitespace():
    """ISC-5."""
    assert is_ingestible_url("https://x.vt.edu/a\x00b") is False
    assert is_ingestible_url("https://x.vt.edu/a\nb") is False
    assert is_ingestible_url("https://x.vt.edu/a b") is False
    assert is_ingestible_url("  https://x.vt.edu/a") is False


def test_rejects_overlength():
    """ISC-6."""
    long_url = "https://x.vt.edu/" + ("a" * (MAX_URL_LENGTH + 10))
    assert len(long_url) > MAX_URL_LENGTH
    assert is_ingestible_url(long_url) is False


def test_rejects_empty_and_non_str():
    assert is_ingestible_url("") is False
    assert is_ingestible_url(None) is False  # type: ignore[arg-type]


# --- is_ingestible_url: accept side (no false positives) ---

def test_accepts_normal_vt_urls():
    """ISC-7."""
    for url in (
        "https://cs.vt.edu/academics/courses.html",
        "http://www.vt.edu",
        "https://dining.vt.edu/menus/",
        "https://cs.vt.edu/page?next=https://login.vt.edu/sso",  # '://' only in query
        "https://lib.vt.edu/research/guides_2024.html",
        "https://x.vt.edu/a%20b/file.pdf",  # percent-encoded space, legal
    ):
        assert is_ingestible_url(url) is True, url


# --- safe_object_key: sanitize side ---

def test_object_key_basic_identity():
    """ISC-10: already-clean URLs produce the historical key unchanged."""
    assert (
        safe_object_key("https://cs.vt.edu/academics/courses.html")
        == "cs.vt.edu/academics/courses.html.md"
    )


def test_object_key_root_and_trailing_slash():
    assert safe_object_key("https://cs.vt.edu") == "cs.vt.edu/index.md"
    assert safe_object_key("https://cs.vt.edu/academics/") == "cs.vt.edu/academics.md"


def test_object_key_collapses_slashes_and_strips_colon():
    """ISC-8 + ISC-9: the crash URL yields a MinIO-legal key."""
    key = safe_object_key(CRASH_URL)
    assert "://" not in key
    assert "//" not in key
    assert ":" not in key
    assert all(ord(c) >= 0x20 for c in key)
    assert key.endswith(".md")


def test_object_key_strips_query_and_fragment():
    assert (
        safe_object_key("https://cs.vt.edu/page?foo=bar#sec")
        == "cs.vt.edu/page.md"
    )


def test_object_key_neutralizes_dot_segments():
    key = safe_object_key("https://x.vt.edu/a/../b")
    assert ".." not in key.split("/")
