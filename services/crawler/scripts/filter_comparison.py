"""
Compare crawl4ai filter configurations across representative VT CS pages.

Tests 5 configs on 4 page types, scores each on:
  - Boilerplate removed (nav/header/footer phrases absent)
  - Content preserved (key phrases present)
  - Size (chars in fit_markdown)
"""

import asyncio
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# ---------------------------------------------------------------------------
# Test pages — one of each type
# ---------------------------------------------------------------------------
PAGES = [
    {
        "name": "Homepage (thin, depth 0)",
        "url": "https://website.cs.vt.edu",
        "must_contain": ["Department of Computer Science", "undergraduate", "graduate"],
        "must_not_contain": ["Skip to main content", "Submenu Toggle", "Follow Computer Science"],
    },
    {
        "name": "About (medium, depth 1)",
        "url": "https://website.cs.vt.edu/About.html",
        "must_contain": ["incubator", "research", "undergraduate", "graduate"],
        "must_not_contain": ["Skip to main content", "Submenu Toggle", "Follow Computer Science"],
    },
    {
        "name": "News article (rich, depth 2)",
        "url": "https://website.cs.vt.edu/About/News/cs-2024-deans-award-winners.html",
        "must_contain": ["Dean", "Excellence in Teaching", "Chris North"],
        "must_not_contain": ["Skip to main content", "Submenu Toggle", "Follow Computer Science", "Hokie Spa"],
    },
    {
        "name": "Staff bio (thin, depth 2)",
        "url": "https://website.cs.vt.edu/people/administration/andrea-sirles.html",
        "must_contain": ["Andrea Sirles", "Graduate program coordinator"],
        "must_not_contain": ["Skip to main content", "Submenu Toggle", "Follow Computer Science", "Hokie Spa", "Search query"],
    },
]

# ---------------------------------------------------------------------------
# Configs to compare
# ---------------------------------------------------------------------------
def make_configs():
    return {
        "A: Current (Pruning fixed 0.45)": CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.45, threshold_type="fixed")
            ),
            verbose=False,
        ),
        "B: excluded_tags only": CrawlerRunConfig(
            excluded_tags=["nav", "header", "footer", "script", "style"],
            word_count_threshold=10,
            verbose=False,
        ),
        "C: excluded_tags + Pruning fixed 0.45": CrawlerRunConfig(
            excluded_tags=["nav", "header", "footer", "script", "style"],
            word_count_threshold=10,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.45, threshold_type="fixed")
            ),
            verbose=False,
        ),
        "D: excluded_tags + Pruning dynamic 0.45": CrawlerRunConfig(
            excluded_tags=["nav", "header", "footer", "script", "style"],
            word_count_threshold=10,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.45,
                    threshold_type="dynamic",
                    min_word_threshold=10,
                )
            ),
            verbose=False,
        ),
        "E: css_selector #vt_main": CrawlerRunConfig(
            css_selector="#vt_main",
            verbose=False,
        ),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(markdown: str, must_contain: list[str], must_not_contain: list[str]) -> dict:
    present = sum(1 for p in must_contain if p.lower() in markdown.lower())
    absent = sum(1 for p in must_not_contain if p.lower() not in markdown.lower())
    return {
        "content_score": f"{present}/{len(must_contain)}",
        "boilerplate_score": f"{absent}/{len(must_not_contain)}",
        "chars": len(markdown),
        "lines": markdown.count("\n"),
        "content_ok": present == len(must_contain),
        "boilerplate_ok": absent == len(must_not_contain),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
async def run():
    configs = make_configs()

    async with AsyncWebCrawler() as crawler:
        for page in PAGES:
            print(f"\n{'='*70}")
            print(f"PAGE: {page['name']}")
            print(f"URL:  {page['url']}")
            print(f"{'='*70}")
            print(f"{'Config':<42} {'Content':>9} {'Boilerplate':>12} {'Chars':>7} {'Lines':>6}  Status")
            print(f"{'-'*90}")

            for name, config in configs.items():
                result = await crawler.arun(url=page["url"], config=config)

                if not result.success:
                    print(f"{name:<42}  FAILED: {result.error_message}")
                    continue

                md = result.markdown.fit_markdown or result.markdown.raw_markdown
                s = score(md, page["must_contain"], page["must_not_contain"])

                status = ""
                if not s["content_ok"]:
                    # Find which phrases are missing
                    missing = [p for p in page["must_contain"] if p.lower() not in md.lower()]
                    status = f"⚠ MISSING: {missing}"
                if not s["boilerplate_ok"]:
                    remaining = [p for p in page["must_not_contain"] if p.lower() in md.lower()]
                    status += f"  ✗ BOILERPLATE: {remaining}"
                if s["content_ok"] and s["boilerplate_ok"]:
                    status = "✓ PASS"

                print(
                    f"{name:<42} {s['content_score']:>9} {s['boilerplate_score']:>12} "
                    f"{s['chars']:>7,} {s['lines']:>6}  {status}"
                )

    print(f"\n{'='*70}")
    print("LEGEND")
    print("  Content score:     phrases_found / phrases_expected")
    print("  Boilerplate score: boilerplate_phrases_absent / total_checked")
    print("  ✓ PASS = all content present AND all boilerplate absent")


if __name__ == "__main__":
    asyncio.run(run())
