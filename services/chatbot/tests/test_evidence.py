from copy import deepcopy

from chatbot.evidence import NO_EVIDENCE_ANSWER, finalize_answer, select_evidence, source_url


def hit(text="Some evidence", url="https://vt.edu/page", title="Page", **extra):
    return {"text": text, "url": url, "title": title, **extra}


def test_context_budgets_and_no_mutation():
    chunks = [hit("word " * 2000, url=f"https://vt.edu/{i}") for i in range(30)]
    original = deepcopy(chunks)
    selected = select_evidence(chunks, max_chunks=12, max_chars=2400, max_chunk_chars=500)
    assert 0 < len(selected) <= 12
    assert all(len(c["text"]) <= 500 for c in selected)
    rendered = "\n\n---\n\n".join(f"[{c['source_id']}] {c['title']} — {c['url']}\n{c['text']}" for c in selected)
    assert len(rendered) <= 2400
    assert chunks == original


def test_cap_final_chunks():
    assert len(select_evidence([hit(url=f"https://vt.edu/{i}") for i in range(30)])) == 12


def test_page_dedup_and_stable_ids():
    selected = select_evidence([hit(), hit("Another passage", url="https://vt.edu/page#section"), hit(), hit(url="https://vt.edu/other")])
    assert [c["source_id"] for c in selected] == [1, 1, 2]
    answer, sources = finalize_answer("Second page [2], first page [1].", selected)
    assert answer == "Second page [2], first page [1]."
    assert [s["id"] for s in sources] == [2, 1]


def test_url_variants_share_source_ids_and_duplicate_passages_are_removed():
    chunks = [hit(url="https://www.vt.edu/page/?utm_source=x"),
              hit(url="http://vt.edu/page#section"),
              hit("Another passage", url="https://vt.edu/page")]
    selected = select_evidence(chunks)
    assert len(selected) == 2
    assert [c["source_id"] for c in selected] == [1, 1]
    assert len(finalize_answer("Fact [1]", selected)[1]) == 1


def test_only_cited_sources():
    selected = select_evidence([hit(), hit(url="https://vt.edu/unused")])
    _, sources = finalize_answer("Fact [1]. More [1].", selected)
    assert sources == [{"id": 1, "title": "Page", "url": "https://vt.edu/page"}]


def test_bad_references_and_missing_citations_fail_closed():
    selected = select_evidence([hit()])
    for answer in ["", "No citations", "Fact [2]", "Good [1] but bad [90]", "Fact [" + "9" * 5000 + "]"]:
        assert finalize_answer(answer, selected) == (NO_EVIDENCE_ANSWER, [])


def test_grouped_citations_normalized():
    selected = select_evidence([hit(), hit(url="https://vt.edu/other")])
    answer, sources = finalize_answer("Fact [1, 2].", selected)
    assert answer == "Fact [1][2]."
    assert len(sources) == 2


def test_bad_urls_and_empty_text_not_context():
    assert select_evidence([hit(url="javascript:alert(1)"), hit(" "), hit(url=""), hit(url="https://user:pass@vt.edu/page")]) == []
    assert source_url("https://VT.edu/page#section") == "https://vt.edu/page"
    assert source_url("https://[invalid") == ""


def test_budget_includes_metadata():
    assert select_evidence([hit(url="https://vt.edu/" + "a" * 500)], max_chars=100) == []


def test_catalog_identity_and_edition_are_visible_within_budget():
    chunk = hit("Complete 123 credit hours", program_name="Computer Science Major",
                program_level="undergraduate", catalog_edition="2026-2027")
    original = deepcopy(chunk)
    selected = select_evidence([chunk], max_chars=300)
    text = selected[0]["text"]
    assert "Program: Computer Science Major" in text
    assert "Degree level: undergraduate" in text
    assert "Catalog edition: 2026-2027" in text
    assert "Complete 123 credit hours" in text
    assert chunk == original
    rendered = f"[1] {selected[0]['title']} — {selected[0]['url']}\n{text}"
    assert len(rendered) <= 300


def test_catalog_metadata_is_bounded_and_single_line():
    chunk = hit(program_name="name\n" * 1000, program_level=None, catalog_edition=2026)
    text = select_evidence([chunk])[0]["text"]
    assert len(text.splitlines()[0]) <= len("Program: ") + 200
    assert len(text.splitlines()) == 2
    assert "Degree level:" not in text
    assert "Catalog edition:" not in text


def test_images_are_not_evidence():
    assert select_evidence([hit("![Image](https://vt.edu/image.png)")]) == []


def test_heading_context_and_readable_titles():
    selected = select_evidence([hit(title="Computer%20Science", headings_path=["Programs", "Computer Science"])])
    assert selected[0]["title"] == "Computer Science"
    assert selected[0]["text"].startswith("Section: Programs > Computer Science\n")
