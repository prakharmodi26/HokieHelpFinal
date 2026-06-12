# services/crawler/tests/test_catalog_card.py
"""Catalog-card extraction: head-metadata precedence chain + card markdown.

Catalog-mode hosts get a compact "know it exists" card — never full-text body
ingest. extract_card walks a strict source precedence per field:
citation_* > Dublin Core > JSON-LD > OpenGraph > <title>/meta description.
"""
from crawler.catalog_card import card_markdown, extract_card

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
