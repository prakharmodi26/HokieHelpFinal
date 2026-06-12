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

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Callable, Iterator
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup

from crawler.markdown_doc import build_markdown_document
from crawler.registry import load_catalog_entries
from crawler.scope import subdomain_of

logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------
# Record harvesting (repositories): one card per record via OAI-PMH or the
# Dataverse search API instead of page crawling. http_get is injected so
# tests run entirely on recorded fixtures — no live network calls.
# --------------------------------------------------------------------------

HttpGet = Callable[[str], str]

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_OAI_DC = "{http://www.openarchives.org/OAI/2.0/oai_dc/}"
_DC = "{http://purl.org/dc/elements/1.1/}"


def harvest_oai(base_url: str, oai_path: str, http_get: HttpGet) -> Iterator[dict]:
    """Yield one card dict per oai_dc record from an OAI-PMH ListRecords
    endpoint, following resumptionToken paging. Deleted records are skipped."""
    endpoint = base_url.rstrip("/") + oai_path
    host = (urlparse(base_url).hostname or "").lower()
    url = f"{endpoint}?verb=ListRecords&metadataPrefix=oai_dc"
    while url:
        root = ET.fromstring(http_get(url))
        list_records = root.find(f"{_OAI}ListRecords")
        if list_records is None:
            return
        for rec in list_records.findall(f"{_OAI}record"):
            header = rec.find(f"{_OAI}header")
            if header is None or header.get("status") == "deleted":
                continue
            dc_el = rec.find(f"{_OAI}metadata/{_OAI_DC}dc")
            if dc_el is None:
                continue

            def texts(tag: str) -> list[str]:
                return [e.text.strip() for e in dc_el.findall(f"{_DC}{tag}")
                        if e.text and e.text.strip()]

            identifiers = texts("identifier")
            rec_url = next((i for i in identifiers if i.startswith("http")), base_url)
            titles, dates = texts("title"), texts("date")
            types_, descriptions = texts("type"), texts("description")
            yield {
                "record_id": header.findtext(f"{_OAI}identifier") or rec_url,
                "title": titles[0] if titles else None,
                "authors": texts("creator"),
                "year": _year_from(dates[0]) if dates else None,
                "type": types_[0] if types_ else None,
                "abstract": descriptions[0] if descriptions else None,
                "url": rec_url,
                "source_host": host,
            }
        token_el = list_records.find(f"{_OAI}resumptionToken")
        token = (token_el.text or "").strip() if token_el is not None else ""
        url = (f"{endpoint}?verb=ListRecords&resumptionToken={quote(token)}"
               if token else None)


def harvest_dataverse(base_url: str, api_path: str, http_get: HttpGet,
                      per_page: int = 100) -> Iterator[dict]:
    """Yield one card dict per dataset from a Dataverse search API,
    paging with start/per_page until total_count is exhausted."""
    endpoint = base_url.rstrip("/") + api_path
    host = (urlparse(base_url).hostname or "").lower()
    start = 0
    while True:
        url = f"{endpoint}/search?q=*&type=dataset&per_page={per_page}&start={start}"
        data = json.loads(http_get(url)).get("data") or {}
        items = data.get("items") or []
        for it in items:
            yield {
                "record_id": it.get("global_id") or it.get("url") or it.get("name"),
                "title": it.get("name"),
                "authors": list(it.get("authors") or []),
                "year": _year_from(it.get("published_at")),
                "type": it.get("type") or "dataset",
                "abstract": it.get("description"),
                "url": it.get("url") or f"https://{host}/",
                "source_host": host,
            }
        start += len(items)
        if not items or start >= int(data.get("total_count") or 0):
            return


def _default_http_get(url: str) -> str:
    import httpx  # already a project dependency (documents pass)

    resp = httpx.get(url, timeout=30.0, follow_redirects=True,
                     headers={"User-Agent": "HokieHelpCrawler/1.0"})
    resp.raise_for_status()
    return resp.text


def _sanitize_record_id(record_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id).strip("_.")
    return safe or "record"


def _record_doc(card: dict, host: str, now: datetime) -> str:
    body = card_markdown(card)
    # doc_id is derived from host+record_id, NOT doc_id_for_url(card["url"]):
    # Dataverse record URLs differ only in the query string, which the
    # url-normalising doc id strips — every record would collide otherwise.
    doc_id = hashlib.sha256(
        f"{host}:{card['record_id']}".encode()
    ).hexdigest()[:16]
    return build_markdown_document(
        url=card["url"], title=card.get("title"), markdown_content=body,
        crawl_depth=0, crawl_timestamp=now, doc_id=doc_id,
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        source_host=host, subdomain=subdomain_of(f"https://{host}/"),
        page_type="catalog_card", is_external=False,
    )


def harvest_catalog_records(
    registry_path: str,
    storage,
    cleaned_bucket: str,
    http_get: HttpGet | None = None,
) -> dict:
    """Harvest every registry catalog entry that has an OAI/Dataverse endpoint
    and write one card doc per record DIRECTLY to the cleaned bucket (the
    chunker's input) under <host>/_catalog/<record-id>.md.

    Entries without harvest config (project apps like opendsa/codeworkout)
    are covered by their single homepage card from the catalog-mode crawl.
    A failing host is logged and skipped — it never kills the run.
    """
    http_get = http_get or _default_http_get
    stats = {"records_harvested": 0, "hosts_harvested": 0, "hosts_failed": 0}
    for entry in load_catalog_entries(registry_path):
        host = entry["host"]
        if entry.get("oai"):
            cards = harvest_oai(f"https://{host}", entry["oai"], http_get)
        elif entry.get("api") and (entry.get("type") or "").startswith("dataverse"):
            cards = harvest_dataverse(f"https://{host}", entry["api"], http_get)
        else:
            continue
        try:
            count = 0
            now = datetime.now(timezone.utc)
            for card in cards:
                key = (f"{host}/_catalog/"
                       f"{_sanitize_record_id(str(card['record_id']))}.md")
                storage.upload_document(key, _record_doc(card, host, now),
                                        bucket=cleaned_bucket)
                count += 1
            stats["records_harvested"] += count
            stats["hosts_harvested"] += 1
            logger.info("Harvested %d catalog records from %s", count, host)
        except Exception as exc:
            stats["hosts_failed"] += 1
            logger.warning("Catalog harvest failed for %s: %s", host, exc)
    return stats
