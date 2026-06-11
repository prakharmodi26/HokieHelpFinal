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
