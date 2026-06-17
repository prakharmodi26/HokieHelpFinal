# services/crawler/tests/test_scope.py
from crawler.scope import (
    Scope,
    registered_domain,
    subdomain_of,
    is_vt_host,
    is_document_url,
    classify_url,
)


def test_registered_domain_and_subdomain():
    assert registered_domain("https://dining.vt.edu/menus") == "vt.edu"
    assert subdomain_of("https://dining.vt.edu/menus") == "dining"
    assert subdomain_of("https://website.cs.vt.edu/About.html") == "website.cs"
    assert subdomain_of("https://vt.edu/") == ""


def test_is_vt_host():
    assert is_vt_host("https://registrar.vt.edu/x") is True
    assert is_vt_host("https://vt.edu/") is True
    assert is_vt_host("https://news.google.com/vt") is False


def test_is_document_url():
    assert is_document_url("https://x.vt.edu/a/checksheet.pdf") is True
    assert is_document_url("https://files.example.com/form.docx") is True
    assert is_document_url("https://x.vt.edu/page.html") is False


def test_classify_vt_page():
    scope, hops = classify_url(
        "https://dining.vt.edu/menus",
        parent_hops=["https://vt.edu/"],
        external_hops=1,
    )
    assert scope is Scope.VT_PAGE
    assert hops == ["https://vt.edu/", "https://dining.vt.edu/menus"]


def test_classify_document_anywhere():
    scope, _ = classify_url(
        "https://cdn.thirdparty.com/vt/form.pdf",
        parent_hops=["https://hr.vt.edu/forms"],
        external_hops=1,
    )
    assert scope is Scope.DOCUMENT


def test_classify_external_within_one_hop():
    # parent is a VT page (0 external hops so far) -> external child allowed, tagged EXTERNAL
    scope, hops = classify_url(
        "https://www.linkedin.com/company/vt-cs",
        parent_hops=["https://vt.edu/", "https://cs.vt.edu/people"],
        external_hops=1,
    )
    assert scope is Scope.EXTERNAL_PAGE
    assert hops[-1] == "https://www.linkedin.com/company/vt-cs"


def test_classify_external_beyond_hop_is_rejected():
    # parent chain already contains one external host -> a further external is OUT_OF_SCOPE
    scope, _ = classify_url(
        "https://medium.com/@someone",
        parent_hops=["https://cs.vt.edu/x", "https://www.linkedin.com/y"],
        external_hops=1,
    )
    assert scope is Scope.OUT_OF_SCOPE


class _Blocker:
    def is_blocked(self, host):
        return host == "login.vt.edu"

    def is_blocked_url(self, url):
        from urllib.parse import urlparse
        p = urlparse(url)
        host = (p.hostname or "").lower()
        return self.is_blocked(host) or "/playlists/track/" in p.path


def test_classify_blocklisted_host_is_out_of_scope():
    scope, _ = classify_url(
        "https://login.vt.edu/auth", parent_hops=["https://vt.edu/"],
        external_hops=1, host_filter=_Blocker(),
    )
    assert scope is Scope.OUT_OF_SCOPE


def test_classify_blocklisted_path_is_out_of_scope():
    """Path-trap URL (e.g. wuvt playlist archive) is OUT_OF_SCOPE even when the
    host itself is allowed — keeps the front page, drops the firehose."""
    scope, _ = classify_url(
        "https://www.wuvt.vt.edu/playlists/track/449476",
        parent_hops=["https://vt.edu/"],
        external_hops=1, host_filter=_Blocker(),
    )
    assert scope is Scope.OUT_OF_SCOPE


def test_classify_rejects_concatenated_href_crash_url():
    """ISC-12: the exact crash URL (embedded scheme in path) is OUT_OF_SCOPE,
    so it can never enter the frontier from seed OR BFS expansion."""
    crash = (
        "https://www.pubs.ext.vt.edu/content/en/426/426-041/426-041.html"
        "https://www.pubs.ext.vt.edu/426/426-841/426-841.html"
    )
    scope, _ = classify_url(crash, parent_hops=[], external_hops=1)
    assert scope is Scope.OUT_OF_SCOPE


def test_classify_rejects_malformed_urls():
    for bad in (
        "javascript:alert(1)",
        "https://x.vt.edu/a\nb",
        "https:///no-host",
    ):
        scope, _ = classify_url(bad, parent_hops=[], external_hops=1)
        assert scope is Scope.OUT_OF_SCOPE, bad
