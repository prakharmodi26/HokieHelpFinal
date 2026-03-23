"""Pipeline configuration — merge env vars with CLI overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    namespace: str
    crawl_seed_url: str
    crawl_max_depth: int
    crawl_max_pages: int
    poll_interval: int  # seconds between job status checks

    # Docker images for each stage
    crawler_image: str
    chunker_image: str
    embedder_image: str

    @staticmethod
    def build(
        *,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        seed_url: Optional[str] = None,
    ) -> PipelineConfig:
        """Build config: CLI args > env vars > hardcoded defaults."""
        return PipelineConfig(
            namespace=os.getenv("PIPELINE_NAMESPACE", "test"),
            crawl_seed_url=(
                seed_url
                or os.getenv("CRAWL_SEED_URL", "https://website.cs.vt.edu")
            ),
            crawl_max_depth=(
                max_depth
                if max_depth is not None
                else int(os.getenv("CRAWL_MAX_DEPTH", "2"))
            ),
            crawl_max_pages=(
                max_pages
                if max_pages is not None
                else int(os.getenv("CRAWL_MAX_PAGES", "500"))
            ),
            poll_interval=int(os.getenv("PIPELINE_POLL_INTERVAL", "30")),
            crawler_image=os.getenv(
                "CRAWLER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-crawler:latest",
            ),
            chunker_image=os.getenv(
                "CHUNKER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-chunker:latest",
            ),
            embedder_image=os.getenv(
                "EMBEDDER_IMAGE",
                "ghcr.io/prakharmodi26/hokiehelp-embedder:latest",
            ),
        )
