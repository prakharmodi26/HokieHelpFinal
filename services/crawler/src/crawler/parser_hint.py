# services/crawler/src/crawler/parser_hint.py
"""Coarse page-type inference from a URL path.

Mirrors chunker.parser.infer_page_type EXACTLY (services/chunker/src/chunker/
parser.py) — the crawler stamps page_type into frontmatter and the chunker
re-derives it, so the two heuristics must stay in lockstep. If you change one,
change both.
"""
from __future__ import annotations


def infer_page_type(url: str) -> str:
    """Infer a coarse page type from the URL path."""
    path = url.lower()
    if any(k in path for k in ["/people/", "/faculty/", "faculty"]):
        return "faculty"
    if any(k in path for k in ["/courses/", "/classes/", "/course"]):
        return "course"
    if any(k in path for k in ["/research/", "/labs/"]):
        return "research"
    if any(k in path for k in ["/news/", "/events/"]):
        return "news"
    if any(k in path for k in ["/about", "/info"]):
        return "about"
    return "general"
