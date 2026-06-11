"""Generic, template-agnostic content extraction.

Uses Crawl4AI's PruningContentFilter to strip nav/footer/boilerplate by scoring
text density / link density / tag importance — no per-site regex. We store both
raw_markdown and fit_markdown; fit is the cleaned text fed to the chunker.
`looks_empty` flags pages that came back without real content (typical of SPAs
that need JS rendering) so the render engine can retry them in the JS tier.
"""
from __future__ import annotations

import re

from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

MIN_CONTENT_WORDS = 25


def make_markdown_generator(threshold: float = 0.5) -> DefaultMarkdownGenerator:
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=threshold, threshold_type="dynamic"
        ),
    )


def select_content(markdown_obj) -> tuple[str, str]:
    """Return (text, used) where used is 'fit' or 'raw'. Prefer fit_markdown."""
    fit = (getattr(markdown_obj, "fit_markdown", None) or "").strip()
    if fit:
        return fit, "fit"
    raw = (getattr(markdown_obj, "raw_markdown", None) or "").strip()
    return raw, "raw"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def looks_empty(text: str) -> bool:
    """True if the page has no usable content (empty or below the word floor)."""
    if not text or not text.strip():
        return True
    return word_count(text) < MIN_CONTENT_WORDS
