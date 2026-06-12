# services/crawler/tests/test_catalog_card.py
"""Catalog-card extraction: head-metadata precedence chain + card markdown.

Catalog-mode hosts get a compact "know it exists" card — never full-text body
ingest. extract_card walks a strict source precedence per field:
citation_* > Dublin Core > JSON-LD > OpenGraph > <title>/meta description.
"""
from unittest.mock import MagicMock

from crawler.catalog_card import (
    card_markdown,
    extract_card,
    harvest_catalog_records,
    harvest_dataverse,
    harvest_oai,
)

URL = "https://vtechworks.lib.vt.edu/items/abc-123"


def _page(head: str) -> str:
    return f"<html><head>{head}</head><body><p>ignored body</p></body></html>"


# ---------------------------------------------------------------- precedence

FULL_STACK_HEAD = """
<title>Fallback Title - VTechWorks</title>
<meta name="description" content="Fallback description.">
<meta property="og:title" content="OG Title">
<meta property="og:description" content="OG description.">
<meta property="og:type" content="website">
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Article",
 "headline": "JSON-LD Title", "description": "JSON-LD abstract.",
 "author": {"@type": "Person", "name": "Ld Author"},
 "datePublished": "2021-03-01"}
</script>
<meta name="DC.title" content="DC Title">
<meta name="DC.creator" content="Dc Author">
<meta name="DCTERMS.issued" content="2022-05-01">
<meta name="DC.description" content="DC abstract.">
<meta name="DC.type" content="Thesis">
<meta name="citation_title" content="Citation Title">
<meta name="citation_author" content="Doe, Jane">
<meta name="citation_author" content="Smith, Bob">
<meta name="citation_publication_date" content="2023/06/15">
<meta name="citation_abstract" content="Citation abstract.">
"""


def test_citation_meta_beats_everything():
    card = extract_card(_page(FULL_STACK_HEAD), URL)
    assert card["title"] == "Citation Title"
    assert card["authors"] == ["Doe, Jane", "Smith, Bob"]
    assert card["year"] == 2023
    assert card["abstract"] == "Citation abstract."
    # citation_* carries no type; DC is the next source that does
    assert card["type"] == "Thesis"
    assert card["url"] == URL
    assert card["source_host"] == "vtechworks.lib.vt.edu"


def test_dublin_core_beats_jsonld_og_and_title():
    head = FULL_STACK_HEAD
    for tag in ("citation_title", "citation_author",
                "citation_publication_date", "citation_abstract"):
        head = "\n".join(l for l in head.splitlines() if tag not in l)
    card = extract_card(_page(head), URL)
    assert card["title"] == "DC Title"
    assert card["authors"] == ["Dc Author"]
    assert card["year"] == 2022
    assert card["abstract"] == "DC abstract."


def test_jsonld_beats_og_and_title():
    head = """
    <title>Fallback Title</title>
    <meta property="og:title" content="OG Title">
    <script type="application/ld+json">
    {"@type": ["ScholarlyArticle"], "name": "JSON-LD Title",
     "author": [{"name": "A One"}, "B Two"],
     "datePublished": "2020-01-15", "description": "JSON-LD abstract."}
    </script>
    """
    card = extract_card(_page(head), URL)
    assert card["title"] == "JSON-LD Title"
    assert card["authors"] == ["A One", "B Two"]
    assert card["year"] == 2020
    assert card["abstract"] == "JSON-LD abstract."
    assert card["type"] == "ScholarlyArticle"


def test_jsonld_ignores_non_creative_types():
    head = """
    <title>Fallback Title</title>
    <script type="application/ld+json">
    {"@type": "Organization", "name": "Some Org"}
    </script>
    """
    card = extract_card(_page(head), URL)
    assert card["title"] == "Fallback Title"


def test_og_beats_title_and_meta_description():
    head = """
    <title>Fallback Title</title>
    <meta name="description" content="Fallback description.">
    <meta property="og:title" content="OG Title">
    <meta property="og:description" content="OG description.">
    """
    card = extract_card(_page(head), URL)
    assert card["title"] == "OG Title"
    assert card["abstract"] == "OG description."


def test_title_and_meta_description_fallback():
    head = """
    <title>  OpenDSA - Interactive Data Structures  </title>
    <meta name="description" content="eTextbook system for DSA courses.">
    """
    card = extract_card(_page(head), "https://opendsa.cs.vt.edu/")
    assert card["title"] == "OpenDSA - Interactive Data Structures"
    assert card["abstract"] == "eTextbook system for DSA courses."
    assert card["authors"] == []
    assert card["year"] is None
    assert card["source_host"] == "opendsa.cs.vt.edu"


def test_empty_html_yields_minimal_card():
    card = extract_card("", URL)
    assert card["title"] is None
    assert card["authors"] == []
    assert card["url"] == URL
    assert card["source_host"] == "vtechworks.lib.vt.edu"


def test_year_extracted_from_messy_date_strings():
    head = '<meta name="citation_title" content="T">' \
           '<meta name="citation_publication_date" content="Published 15 June 1999.">'
    card = extract_card(_page(head), URL)
    assert card["year"] == 1999


# ------------------------------------------------------------- card markdown


def test_card_markdown_contains_all_fields():
    card = {
        "title": "Citation Title", "authors": ["Doe, Jane", "Smith, Bob"],
        "year": 2023, "type": "Thesis", "url": URL,
        "abstract": "Citation abstract.", "source_host": "vtechworks.lib.vt.edu",
    }
    md = card_markdown(card)
    assert "# Citation Title" in md
    assert "Doe, Jane, Smith, Bob" in md
    assert "2023" in md
    assert "Thesis" in md
    assert URL in md
    assert "Citation abstract." in md


def test_card_markdown_handles_sparse_card():
    md = card_markdown({
        "title": None, "authors": [], "year": None, "type": None,
        "url": "https://geovine.cgit.vt.edu/", "abstract": None,
        "source_host": "geovine.cgit.vt.edu",
    })
    assert "https://geovine.cgit.vt.edu/" in md  # link always present
    assert "Authors" not in md
    assert "Year" not in md


# ----------------------------------------------------- OAI-PMH harvest (fixtures)

_OAI_NS = (
    'xmlns="http://www.openarchives.org/OAI/2.0/" '
    'xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/"'
)


def _oai_record(num, title, creators, date, desc):
    creators_xml = "".join(f"<dc:creator>{c}</dc:creator>" for c in creators)
    return f"""
    <record>
      <header><identifier>oai:vtechworks.lib.vt.edu:10919/{num}</identifier>
        <datestamp>{date}</datestamp></header>
      <metadata>
        <oai_dc:dc>
          <dc:title>{title}</dc:title>
          {creators_xml}
          <dc:date>{date}</dc:date>
          <dc:type>Thesis</dc:type>
          <dc:description>{desc}</dc:description>
          <dc:identifier>http://hdl.handle.net/10919/{num}</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>"""


OAI_PAGE_1 = f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH {_OAI_NS}>
  <responseDate>2026-06-11T00:00:00Z</responseDate>
  <ListRecords>
    {_oai_record(1, "First Thesis", ["Doe, Jane", "Smith, Bob"], "2019-08-12", "Abstract one.")}
    {_oai_record(2, "Second Thesis", ["Roe, Rich"], "2021-01-30", "Abstract two.")}
    <record>
      <header status="deleted">
        <identifier>oai:vtechworks.lib.vt.edu:10919/99</identifier>
      </header>
    </record>
    <resumptionToken completeListSize="3" cursor="0">tok-page-2</resumptionToken>
  </ListRecords>
</OAI-PMH>"""

OAI_PAGE_2 = f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH {_OAI_NS}>
  <responseDate>2026-06-11T00:00:01Z</responseDate>
  <ListRecords>
    {_oai_record(3, "Third Thesis", ["Poe, Edgar"], "2023-11-02", "Abstract three.")}
    <resumptionToken completeListSize="3" cursor="2"/>
  </ListRecords>
</OAI-PMH>"""


def _oai_http_get(requested):
    def http_get(url):
        requested.append(url)
        return OAI_PAGE_2 if "resumptionToken=tok-page-2" in url else OAI_PAGE_1
    return http_get


def test_harvest_oai_parses_records_and_follows_resumption_token():
    requested: list[str] = []
    cards = list(harvest_oai(
        "https://vtechworks.lib.vt.edu", "/server/oai/request",
        _oai_http_get(requested),
    ))
    # 3 live records across 2 pages; the deleted record is skipped
    assert [c["title"] for c in cards] == ["First Thesis", "Second Thesis", "Third Thesis"]
    assert cards[0]["authors"] == ["Doe, Jane", "Smith, Bob"]
    assert cards[0]["year"] == 2019
    assert cards[0]["type"] == "Thesis"
    assert cards[0]["abstract"] == "Abstract one."
    assert cards[0]["url"] == "http://hdl.handle.net/10919/1"
    assert cards[0]["record_id"] == "oai:vtechworks.lib.vt.edu:10919/1"
    assert cards[0]["source_host"] == "vtechworks.lib.vt.edu"
    # exactly one resumption-token page hop, against the same endpoint
    assert len(requested) == 2
    assert requested[0].startswith(
        "https://vtechworks.lib.vt.edu/server/oai/request?verb=ListRecords"
    )
    assert "metadataPrefix=oai_dc" in requested[0]
    assert "resumptionToken=tok-page-2" in requested[1]


# --------------------------------------------------- Dataverse harvest (fixtures)

DATAVERSE_PAGE_1 = """{
  "status": "OK",
  "data": {
    "q": "*", "total_count": 3, "start": 0, "count_in_response": 2,
    "items": [
      {"name": "Driving Study A", "type": "dataset",
       "url": "https://dataverse.vtti.vt.edu/citation?persistentId=doi:10.15787/VTT1/AAAA",
       "global_id": "doi:10.15787/VTT1/AAAA",
       "description": "Naturalistic driving data A.",
       "published_at": "2020-03-15T10:00:00Z",
       "authors": ["Driver, Dana"]},
      {"name": "Driving Study B", "type": "dataset",
       "url": "https://dataverse.vtti.vt.edu/citation?persistentId=doi:10.15787/VTT1/BBBB",
       "global_id": "doi:10.15787/VTT1/BBBB",
       "description": "Naturalistic driving data B.",
       "published_at": "2022-07-01T10:00:00Z",
       "authors": ["Rider, Rory", "Driver, Dana"]}
    ]
  }
}"""

DATAVERSE_PAGE_2 = """{
  "status": "OK",
  "data": {
    "q": "*", "total_count": 3, "start": 2, "count_in_response": 1,
    "items": [
      {"name": "Driving Study C", "type": "dataset",
       "url": "https://dataverse.vtti.vt.edu/citation?persistentId=doi:10.15787/VTT1/CCCC",
       "global_id": "doi:10.15787/VTT1/CCCC",
       "description": "Naturalistic driving data C.",
       "published_at": "2024-01-20T10:00:00Z",
       "authors": []}
    ]
  }
}"""


def test_harvest_dataverse_parses_search_pages():
    requested: list[str] = []

    def http_get(url):
        requested.append(url)
        return DATAVERSE_PAGE_2 if "start=2" in url else DATAVERSE_PAGE_1

    cards = list(harvest_dataverse(
        "https://dataverse.vtti.vt.edu", "/api", http_get, per_page=2,
    ))
    assert [c["title"] for c in cards] == [
        "Driving Study A", "Driving Study B", "Driving Study C",
    ]
    assert cards[0]["authors"] == ["Driver, Dana"]
    assert cards[0]["year"] == 2020
    assert cards[0]["type"] == "dataset"
    assert cards[0]["abstract"] == "Naturalistic driving data A."
    assert cards[0]["record_id"] == "doi:10.15787/VTT1/AAAA"
    assert cards[0]["source_host"] == "dataverse.vtti.vt.edu"
    assert len(requested) == 2
    assert requested[0].startswith("https://dataverse.vtti.vt.edu/api/search?")
    assert "start=0" in requested[0] and "start=2" in requested[1]


# ----------------------------------------------------- harvest entrypoint

_HARVEST_REG = """
full:
  - registrar.vt.edu
catalog:
  - host: vtechworks.lib.vt.edu
    type: dspace
    rest: /server/api
    oai: /server/oai/request
  - host: dataverse.vtti.vt.edu
    type: dataverse
    api: /api
  - host: opendsa.cs.vt.edu
special:
  - host: dining.vt.edu
    handler: dining_tool
"""


def test_harvest_catalog_records_writes_cards_to_cleaned_bucket(tmp_path):
    reg = tmp_path / "reg.yaml"
    reg.write_text(_HARVEST_REG)
    storage = MagicMock()

    def http_get(url):
        if "oai" in url:
            return OAI_PAGE_2 if "resumptionToken" in url else OAI_PAGE_1
        return DATAVERSE_PAGE_1 if "start=0" in url else DATAVERSE_PAGE_2

    stats = harvest_catalog_records(str(reg), storage, "crawled-pages-cleaned",
                                    http_get=http_get)
    assert stats["records_harvested"] == 6  # 3 OAI + 3 Dataverse
    assert stats["hosts_harvested"] == 2  # opendsa has no oai/api config
    assert stats["hosts_failed"] == 0

    calls = storage.upload_document.call_args_list
    keys = [c[0][0] for c in calls]
    docs = {c[0][0]: c[0][1] for c in calls}
    # one card per record under <host>/_catalog/<sanitized-record-id>.md
    assert "vtechworks.lib.vt.edu/_catalog/oai_vtechworks.lib.vt.edu_10919_1.md" in keys
    assert "dataverse.vtti.vt.edu/_catalog/doi_10.15787_VTT1_AAAA.md" in keys
    assert all(c.kwargs["bucket"] == "crawled-pages-cleaned" for c in calls)
    # chunker-compatible frontmatter on every card
    card = docs["vtechworks.lib.vt.edu/_catalog/oai_vtechworks.lib.vt.edu_10919_1.md"]
    assert "page_type: 'catalog_card'" in card
    assert "source_host: 'vtechworks.lib.vt.edu'" in card
    assert "subdomain: 'vtechworks.lib'" in card
    assert "is_external: false" in card
    assert "First Thesis" in card
    # doc_ids must be unique even though dataverse record URLs differ only in
    # the query string (which url-based doc ids strip)
    import re
    doc_ids = {re.search(r"doc_id: '([^']+)'", d).group(1) for d in docs.values()}
    assert len(doc_ids) == 6


def test_harvest_catalog_records_continues_past_failing_host(tmp_path):
    reg = tmp_path / "reg.yaml"
    reg.write_text(_HARVEST_REG)
    storage = MagicMock()

    def http_get(url):
        if "oai" in url:
            raise RuntimeError("endpoint down")
        return DATAVERSE_PAGE_1 if "start=0" in url else DATAVERSE_PAGE_2

    stats = harvest_catalog_records(str(reg), storage, "crawled-pages-cleaned",
                                    http_get=http_get)
    # vtechworks OAI fails, dataverse still harvested
    assert stats["hosts_failed"] == 1
    assert stats["hosts_harvested"] == 1
    assert stats["records_harvested"] == 3
