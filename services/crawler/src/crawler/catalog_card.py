# services/crawler/src/crawler/catalog_card.py
"""Metadata-only catalog cards for catalog-mode hosts (Task 8.2).

Catalog hosts (repositories, archives, project apps) are never full-text
ingested. Instead each page/record becomes a compact card — title, authors,
year, type, abstract, link — extracted from head metadata with a strict
per-field source precedence:

    citation_*  >  Dublin Core (DC.*/DCTERMS.*)  >  JSON-LD  >  OpenGraph
                >  <title> + meta description

Per-record coverage for repositories comes from the OAI-PMH / Dataverse
harvesters below (one card per record), not from page crawling.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

# JSON-LD @type values that describe a catalogable work (not e.g. Organization).
_JSONLD_WORK_TYPES = {
    "Article", "ScholarlyArticle", "NewsArticle", "Dataset", "CreativeWork",
    "Book", "Thesis", "Report", "SoftwareApplication", "SoftwareSourceCode",
}


def _year_from(value: str | None) -> int | None:
    """First plausible 4-digit year in a free-form date string."""
    if not value:
        return None
    m = _YEAR_RE.search(value)
    return int(m.group(0)) if m else None


def _meta_by_name(soup: BeautifulSoup) -> dict[str, list[str]]:
    """{lowercased meta name: [content, ...]} preserving document order."""
    out: dict[str, list[str]] = {}
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or "").strip().lower()
        content = (tag.get("content") or "").strip()
        if name and content:
            out.setdefault(name, []).append(content)
    return out


def _first(meta: dict[str, list[str]], *names: str) -> str | None:
    for n in names:
        if meta.get(n):
            return meta[n][0]
    return None


def _all(meta: dict[str, list[str]], *names: str) -> list[str]:
    for n in names:
        if meta.get(n):
            return list(meta[n])
    return []


def _citation_fields(soup, meta) -> dict:
    return {
        "title": _first(meta, "citation_title"),
        "authors": _all(meta, "citation_author"),
        "year": _year_from(_first(meta, "citation_publication_date",
                                  "citation_date", "citation_online_date")),
        "abstract": _first(meta, "citation_abstract"),
        "type": None,  # Highwire citation_* tags carry no work type
    }


def _dublin_core_fields(soup, meta) -> dict:
    return {
        "title": _first(meta, "dc.title", "dcterms.title"),
        "authors": _all(meta, "dc.creator", "dcterms.creator", "dc.contributor.author"),
        "year": _year_from(_first(meta, "dcterms.issued", "dc.date",
                                  "dcterms.date", "dc.date.issued")),
        "abstract": _first(meta, "dc.description", "dcterms.abstract",
                           "dc.description.abstract"),
        "type": _first(meta, "dc.type", "dcterms.type"),
    }


def _jsonld_author_names(author) -> list[str]:
    items = author if isinstance(author, list) else [author]
    names = []
    for a in items:
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, dict) and a.get("name"):
            names.append(a["name"])
    return names


def _jsonld_fields(soup, meta) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = data if isinstance(data, list) else data.get("@graph", [data]) \
            if isinstance(data, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            raw_type = node.get("@type") or []
            types = [raw_type] if isinstance(raw_type, str) else list(raw_type)
            matched = [t for t in types if t in _JSONLD_WORK_TYPES]
            if not matched:
                continue
            return {
                "title": node.get("headline") or node.get("name"),
                "authors": _jsonld_author_names(node.get("author") or []),
                "year": _year_from(node.get("datePublished")
                                   or node.get("dateCreated")),
                "abstract": node.get("description") or node.get("abstract"),
                "type": matched[0],
            }
    return {}


def _opengraph_fields(soup, meta) -> dict:
    og: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        prop = (tag.get("property") or "").strip().lower()
        content = (tag.get("content") or "").strip()
        if prop.startswith(("og:", "article:")) and content and prop not in og:
            og[prop] = content
    og_type = og.get("og:type")
    return {
        "title": og.get("og:title"),
        "authors": [],
        "year": _year_from(og.get("article:published_time")),
        "abstract": og.get("og:description"),
        # og:type "website" says nothing useful about the work
        "type": og_type if og_type and og_type != "website" else None,
    }


def _fallback_fields(soup, meta) -> dict:
    title_tag = soup.find("title")
    return {
        "title": title_tag.get_text(strip=True) if title_tag else None,
        "authors": [],
        "year": None,
        "abstract": _first(meta, "description"),
        "type": None,
    }


def extract_card(html: str, url: str) -> dict:
    """Extract a compact catalog card from page head metadata.

    Per-field precedence: citation_* > Dublin Core > JSON-LD > OpenGraph >
    <title>/meta description. Returns
    {title, authors[], year, type, url, abstract, source_host}.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    meta = _meta_by_name(soup)
    sources = [
        _citation_fields(soup, meta),
        _dublin_core_fields(soup, meta),
        _jsonld_fields(soup, meta),
        _opengraph_fields(soup, meta),
        _fallback_fields(soup, meta),
    ]
    card: dict = {"title": None, "authors": [], "year": None,
                  "type": None, "abstract": None}
    for field in card:
        for src in sources:
            if src.get(field):
                card[field] = src[field]
                break
    card["url"] = url
    card["source_host"] = (urlparse(url).hostname or "").lower()
    return card


def card_markdown(card: dict) -> str:
    """Render a card dict as the small markdown body stored for the chunker."""
    lines = [f"# {card.get('title') or card['url']}", ""]
    if card.get("authors"):
        lines.append(f"**Authors:** {', '.join(card['authors'])}")
    if card.get("year"):
        lines.append(f"**Year:** {card['year']}")
    if card.get("type"):
        lines.append(f"**Type:** {card['type']}")
    lines.append(f"**Link:** {card['url']}")
    if card.get("abstract"):
        lines += ["", card["abstract"]]
    return "\n".join(lines) + "\n"
