# services/crawler/src/crawler/scope.py
"""URL scope classification and hop-history computation.

Three active storage scopes:
  VT_PAGE       — host is vt.edu (or a subdomain) OR is in the VT-affiliated
                  allowlist; full ingest to the main corpus.
  DOCUMENT      — a PDF/DOC/DOCX URL on a VT-affiliated host; ingested to the
                  main corpus.  Non-affiliated doc hosts → OUT_OF_SCOPE.
  OUT_OF_SCOPE  — anything else (non-VT HTML, non-affiliated PDFs, blocked
                  hosts/paths); never fetched or stored.

EXTERNAL_PAGE is retained as an enum member to avoid breaking any frontier
rows from earlier runs that stored scope="external_page", but it is UNREACHABLE
from classify_url — no code path returns it. External, non-VT HTML is
unconditionally OUT_OF_SCOPE; the external bucket no longer exists.
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
    # EXTERNAL_PAGE is retained only as a sentinel for old frontier rows that
    # stored scope="external_page". classify_url never returns it; any attempt
    # to route it to storage falls through to the main bucket (bucket_for_scope
    # no longer has an external-bucket branch).
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


def is_vt_affiliated(url: str, matcher=None) -> bool:
    """Return True if *url*'s host is VT-affiliated.

    A host is VT-affiliated when it is under the ``vt.edu`` registered domain
    OR it matches the optional *matcher* (an ``AffiliatedHostsMatcher``).
    """
    if is_vt_host(url):
        return True
    if matcher is not None and matcher.is_affiliated(url):
        return True
    return False


def is_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOCUMENT_EXTENSIONS)


def classify_url(
    url: str,
    parent_hops: list[str],
    host_filter=None,
    affiliated_matcher=None,
) -> tuple[Scope, list[str]]:
    """Classify a URL given its parent's hop chain; return (scope, new_hop_chain).

    new_hop_chain = parent_hops + [url], i.e. the full referrer path to `url`.

    ``affiliated_matcher`` is an optional ``AffiliatedHostsMatcher`` that
    extends VT affiliation beyond the ``vt.edu`` registered domain.  When
    provided, allowlisted hosts (e.g. hokiesports.com, *.vt.domains) are
    treated as VT pages and their documents are ingested to the main corpus.

    Non-VT, non-affiliated HTML unconditionally returns OUT_OF_SCOPE — there
    is no external-hops budget and no EXTERNAL_PAGE pathway.
    """
    hops = list(parent_hops) + [url]
    # Single hygiene gate for both the registry seed path and BFS expansion:
    # reject malformed/over-length/embedded-scheme URLs before any scope work.
    # Subsumes the old bare http(s) scheme check.
    if not is_ingestible_url(url):
        return Scope.OUT_OF_SCOPE, hops

    if host_filter is not None and host_filter.is_blocked_url(url):
        return Scope.OUT_OF_SCOPE, hops

    affiliated = is_vt_affiliated(url, affiliated_matcher)

    if is_document_url(url):
        # Only ingest documents from VT-affiliated hosts; drop all others to
        # prevent non-VT CDN/arxiv/SEC docs leaking into the main corpus.
        if affiliated:
            return Scope.DOCUMENT, hops
        return Scope.OUT_OF_SCOPE, hops

    if affiliated:
        return Scope.VT_PAGE, hops

    # Non-affiliated HTML: OUT_OF_SCOPE unconditionally.
    # The previous EXTERNAL_PAGE / external-hops budget path is removed.
    # A stale CRAWL_EXTERNAL_HOPS=1 env var in the cluster is now a no-op.
    return Scope.OUT_OF_SCOPE, hops
