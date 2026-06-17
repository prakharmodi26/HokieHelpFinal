# services/crawler/src/crawler/scope.py
"""URL scope classification and hop-history computation.

Three storage scopes:
  VT_PAGE       — host is vt.edu or a subdomain; full ingest to the main corpus.
  EXTERNAL_PAGE — non-VT HTML reached within `external_hops` of a VT page;
                  stored separately for later review, never auto-fed to RAG.
  DOCUMENT      — a PDF/DOC/DOCX URL on ANY host (VT content often lives on CDNs);
                  ingested to the main corpus.
  OUT_OF_SCOPE  — anything else (e.g. a second external hop); never fetched.
"""
from __future__ import annotations

import enum
from urllib.parse import urlparse

import tldextract

from crawler.urls import is_ingestible_url

# Use a no-network extractor (no live public-suffix fetch in containers/tests).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

DOCUMENT_EXTENSIONS = (".pdf", ".doc", ".docx")
_VT_REGISTERED = "vt.edu"


class Scope(enum.Enum):
    VT_PAGE = "vt_page"
    EXTERNAL_PAGE = "external_page"
    DOCUMENT = "document"
    OUT_OF_SCOPE = "out_of_scope"


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def registered_domain(url: str) -> str:
    ext = _EXTRACT(_host(url))
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}"


def subdomain_of(url: str) -> str:
    """Return the subdomain label(s) below the registered domain ('' for apex)."""
    return _EXTRACT(_host(url)).subdomain


def is_vt_host(url: str) -> bool:
    return registered_domain(url) == _VT_REGISTERED


def is_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOCUMENT_EXTENSIONS)


def _external_hops_in_chain(hops: list[str]) -> int:
    """Count how many hops in the chain are on non-VT, non-document hosts."""
    return sum(
        1 for u in hops if not is_vt_host(u) and not is_document_url(u)
    )


def classify_url(
    url: str,
    parent_hops: list[str],
    external_hops: int,
    host_filter=None,
) -> tuple[Scope, list[str]]:
    """Classify a URL given its parent's hop chain; return (scope, new_hop_chain).

    new_hop_chain = parent_hops + [url], i.e. the full referrer path to `url`.
    """
    hops = list(parent_hops) + [url]
    # Single hygiene gate for both the registry seed path and BFS expansion:
    # reject malformed/over-length/embedded-scheme URLs before any scope work.
    # Subsumes the old bare http(s) scheme check.
    if not is_ingestible_url(url):
        return Scope.OUT_OF_SCOPE, hops

    if host_filter is not None and host_filter.is_blocked_url(url):
        return Scope.OUT_OF_SCOPE, hops

    if is_document_url(url):
        return Scope.DOCUMENT, hops
    if is_vt_host(url):
        return Scope.VT_PAGE, hops

    # External HTML: allowed only if it does not exceed the external-hop budget.
    # parent_hops already includes any external hosts traversed to get here.
    already_external = _external_hops_in_chain(parent_hops)
    if already_external < external_hops:
        return Scope.EXTERNAL_PAGE, hops
    return Scope.OUT_OF_SCOPE, hops
