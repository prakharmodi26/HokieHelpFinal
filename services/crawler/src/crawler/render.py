"""Two-tier render configs + scope→bucket routing for the crawl engine.

Fast tier: quick render with generic fit_markdown extraction, robots respected.
JS tier:   for pages that came back empty (SPAs, lazy-loaded dining menus) —
           scans the full page, waits for late JS, longer timeout.

Routing:   VT pages and documents -> main bucket (chunker input);
           external HTML -> separate review bucket (never auto-chunked).
"""
from __future__ import annotations

from crawl4ai import CrawlerRunConfig

from crawler.extract import make_markdown_generator
from crawler.scope import Scope


def fast_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        page_timeout=30000,
        delay_before_return_html=0.3,
        stream=True,
        verbose=False,
    )


def js_run_config(threshold: float = 0.5) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        markdown_generator=make_markdown_generator(threshold),
        check_robots_txt=True,
        excluded_tags=["nav", "script", "style", "noscript", "iframe"],
        scan_full_page=True,
        scroll_delay=0.5,
        wait_for_images=True,
        delay_before_return_html=1.0,
        page_timeout=90000,
        stream=True,
        verbose=False,
    )


def bucket_for_scope(scope: Scope, main_bucket: str, external_bucket: str) -> str:
    if scope is Scope.EXTERNAL_PAGE:
        return external_bucket
    return main_bucket
