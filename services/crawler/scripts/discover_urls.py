#!/usr/bin/env python3
"""URL discovery script — maps the full website.cs.vt.edu domain.

Run from the services/crawler directory:
    python scripts/discover_urls.py [--depth N] [--max-pages N]

Outputs all discovered URLs grouped by top-level path prefix.
Use this report to decide which paths to block in CRAWL_BLOCKED_PATHS.
"""
import argparse
import asyncio
import sys
from collections import defaultdict
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain


SEED_URL = "https://website.cs.vt.edu"
ALLOWED_DOMAIN = "website.cs.vt.edu"


async def discover(max_depth: int, max_pages: int) -> list[str]:
    """Run BFS and return all successfully reached URLs."""
    filter_chain = FilterChain([
        DomainFilter(allowed_domains=[ALLOWED_DOMAIN]),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        include_external=False,
        max_pages=max_pages,
        filter_chain=filter_chain,
    )

    # Minimal config — we only care about URLs, not content
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=0.5,
    )

    urls: list[str] = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=SEED_URL, config=run_config):
            if result.success:
                urls.append(result.url)
                print(f"  [{len(urls)}] {result.url}", file=sys.stderr)

    return urls


def report(urls: list[str]) -> None:
    """Print a grouped report of all discovered URLs."""
    by_prefix: dict[str, list[str]] = defaultdict(list)

    for url in sorted(urls):
        path = urlparse(url).path.lstrip("/")
        prefix = path.split("/")[0] if "/" in path else path
        by_prefix[prefix].append(url)

    print(f"\n{'='*60}")
    print(f"DISCOVERY REPORT — {len(urls)} pages found")
    print(f"{'='*60}\n")

    for prefix, group in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        print(f"[{len(group):4d}]  /{prefix}/")
        for url in group[:5]:
            print(f"          {url}")
        if len(group) > 5:
            print(f"          ... and {len(group) - 5} more")
        print()

    print(f"\nFull URL list:")
    for url in sorted(urls):
        print(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover all URLs on website.cs.vt.edu")
    parser.add_argument("--depth", type=int, default=5, help="BFS max depth (default: 5)")
    parser.add_argument("--max-pages", type=int, default=5000, help="Max pages to crawl (default: 5000)")
    args = parser.parse_args()

    print(f"Starting discovery: depth={args.depth}, max_pages={args.max_pages}", file=sys.stderr)
    urls = asyncio.run(discover(args.depth, args.max_pages))
    report(urls)


if __name__ == "__main__":
    main()
