from crawler.extract import (
    make_markdown_generator,
    select_content,
    looks_empty,
    word_count,
)


def test_make_markdown_generator_uses_pruning_filter():
    gen = make_markdown_generator(threshold=0.5)
    # the generator carries a content filter; just assert it constructs
    assert gen is not None


def test_select_content_prefers_fit_over_raw():
    class FakeMd:
        fit_markdown = "# Real content\nlots of words here"
        raw_markdown = "# Real content\nlots of words here\n[nav][nav][nav]"
    text, used = select_content(FakeMd())
    assert used == "fit"
    assert text.startswith("# Real content")


def test_select_content_falls_back_to_raw_when_fit_empty():
    class FakeMd:
        fit_markdown = "   "
        raw_markdown = "# Something\nbody"
    text, used = select_content(FakeMd())
    assert used == "raw"
    assert "Something" in text


def test_looks_empty_and_word_count():
    assert looks_empty("") is True
    assert looks_empty("   \n  ") is True
    assert looks_empty("# Title\nApp loading...") is True  # below min words
    assert looks_empty("# Title\n" + "word " * 40) is False
    assert word_count("one two three") == 3
