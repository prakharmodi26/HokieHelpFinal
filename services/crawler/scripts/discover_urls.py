#!/usr/bin/env python3
"""URL discovery script — maps the full website.cs.vt.edu domain.

Run from the services/crawler directory:
    python scripts/discover_urls.py [--depth N] [--max-pages N]

Combines two strategies for maximum coverage:
  1. Common Crawl index (instant bulk discovery, no crawling)
  2. BFS deep crawl with Crawl4AI (catches pages CC missed)

The site has no sitemap, so we need both approaches.
"""
import argparse
import asyncio
import sys
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

from crawl4ai import AsyncWebCrawler, AsyncUrlSeeder, CrawlerRunConfig, SeedingConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain


SEED_URL = "https://website.cs.vt.edu"
ALLOWED_DOMAIN = "website.cs.vt.edu"
BLOCKED_DOMAINS = {
    "git.cs.vt.edu", "gitlab.cs.vt.edu", "mail.cs.vt.edu",
    "webmail.cs.vt.edu", "portal.cs.vt.edu", "api.cs.vt.edu",
    "forum.cs.vt.edu", "login.cs.vt.edu", "students.cs.vt.edu",
    "wordpress.cs.vt.edu", "wiki.cs.vt.edu",
}


def _normalize(url: str) -> str:
    """Strip fragments and trailing slashes for dedup."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(parsed._replace(fragment="", path=path, query=""))


async def discover_cc() -> set[str]:
    """Phase 1: Instant discovery via Common Crawl index."""
    print("Phase 1: Querying Common Crawl index...", file=sys.stderr)
    config = SeedingConfig(
        source="cc",
        extract_head=False,
        live_check=False,
        verbose=True,
        filter_nonsense_urls=True,
    )
    async with AsyncUrlSeeder() as seeder:
        results = await seeder.urls(ALLOWED_DOMAIN, config)

    urls = set()
    for entry in results:
        url = entry["url"] if isinstance(entry, dict) else str(entry)
        host = urlparse(url).hostname or ""
        if host == ALLOWED_DOMAIN:
            urls.add(_normalize(url))

    print(f"  Common Crawl: {len(urls)} URLs", file=sys.stderr)
    return urls


async def discover_bfs(max_depth: int, max_pages: int) -> set[str]:
    """Phase 2: BFS crawl to catch pages CC missed."""
    print(f"\nPhase 2: BFS crawl (depth={max_depth}, max_pages={max_pages})...", file=sys.stderr)

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=[ALLOWED_DOMAIN, "cs.vt.edu"],
            blocked_domains=list(BLOCKED_DOMAINS),
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        include_external=False,
        max_pages=max_pages,
        filter_chain=filter_chain,
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        verbose=False,
        delay_before_return_html=0.5,
    )

    urls: set[str] = set()
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=SEED_URL, config=run_config):
            if result.success:
                host = urlparse(result.url).hostname
                if host == ALLOWED_DOMAIN:
                    urls.add(_normalize(result.url))
                    print(f"  [BFS {len(urls)}] {result.url}", file=sys.stderr)

    print(f"  BFS crawl: {len(urls)} URLs", file=sys.stderr)
    return urls


async def discover(max_depth: int, max_pages: int) -> list[str]:
    """Combine CC + BFS for maximum URL coverage."""
    cc_urls = await discover_cc()
    bfs_urls = await discover_bfs(max_depth, max_pages)

    combined = cc_urls | bfs_urls
    only_cc = cc_urls - bfs_urls
    only_bfs = bfs_urls - cc_urls

    print(f"\n--- Coverage Summary ---", file=sys.stderr)
    print(f"  Common Crawl only: {len(only_cc)}", file=sys.stderr)
    print(f"  BFS only:          {len(only_bfs)}", file=sys.stderr)
    print(f"  Both:              {len(cc_urls & bfs_urls)}", file=sys.stderr)
    print(f"  Combined total:    {len(combined)}", file=sys.stderr)

    return sorted(combined)


def report(urls: list[str]) -> None:
    """Print a grouped report of all discovered URLs."""
    by_prefix: dict[str, list[str]] = defaultdict(list)

    for url in urls:
        path = urlparse(url).path.lstrip("/")
        prefix = path.split("/")[0] if "/" in path else path
        by_prefix[prefix].append(url)

    print(f"\n{'='*60}")
    print(f"DISCOVERY REPORT — {len(urls)} unique pages found")
    print(f"{'='*60}\n")

    for prefix, group in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        print(f"[{len(group):4d}]  /{prefix}/")
        for url in group[:5]:
            print(f"          {url}")
        if len(group) > 5:
            print(f"          ... and {len(group) - 5} more")
        print()

    print(f"\nFull URL list:")
    for url in urls:
        print(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover all URLs on website.cs.vt.edu")
    parser.add_argument("--depth", type=int, default=5, help="BFS max depth (default: 5)")
    parser.add_argument("--max-pages", type=int, default=5000, help="Max pages to crawl (default: 5000)")
    args = parser.parse_args()

    urls = asyncio.run(discover(args.depth, args.max_pages))
    report(urls)


if __name__ == "__main__":
    main()
