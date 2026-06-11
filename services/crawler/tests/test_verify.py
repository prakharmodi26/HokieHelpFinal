from crawler.verify import assess_document, summarize


def test_assess_document_flags_empty_and_boilerplate():
    good = "---\nurl: 'https://x.vt.edu/a'\nsource_host: 'x.vt.edu'\n---\n\n# Real\n" + "word " * 60
    a = assess_document("x.vt.edu/a.md", good)
    assert a.ok is True
    assert a.word_count >= 50

    empty = "---\nurl: 'https://x.vt.edu/b'\n---\n\n   "
    b = assess_document("x.vt.edu/b.md", empty)
    assert b.ok is False
    assert "empty" in b.reasons[0]

    no_fm = "# Orphan\nbody words " * 5
    c = assess_document("x.vt.edu/c.md", no_fm)
    assert c.ok is False
    assert any("frontmatter" in r for r in c.reasons)


def test_summarize_counts():
    from crawler.verify import DocAssessment
    rows = [
        DocAssessment("a", True, 100, []),
        DocAssessment("b", False, 0, ["empty body"]),
        DocAssessment("c", False, 5, ["too short"]),
    ]
    s = summarize(rows)
    assert s["total"] == 3
    assert s["ok"] == 1
    assert s["failed"] == 2
    assert "empty body" in s["reason_histogram"]
