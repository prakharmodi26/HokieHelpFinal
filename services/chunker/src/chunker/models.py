"""ChunkRecord dataclass — the output unit of the chunking service."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ChunkRecord:
    """A single chunk produced from one crawled page."""

    chunk_id: str          # "{document_id}_{chunk_index:04d}"
    document_id: str       # 16-hex doc ID from crawler
    chunk_index: int       # 0-based position within the document
    text: str              # Chunk content (includes heading)
    url: str               # Source page URL
    title: str             # Page title
    page_type: str         # "faculty" | "course" | "research" | "news" | "about" | "general"
    headings_path: List[str]  # e.g. ["People", "Faculty", "Professor Jane Doe"]
    content_hash: str      # SHA-256[:16] of this chunk's text
    crawl_timestamp: str   # ISO timestamp from frontmatter
    token_count: int       # Approximate token count (len(text) // 4); used for embedding batching

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "page_type": self.page_type,
            "headings_path": self.headings_path,
            "content_hash": self.content_hash,
            "crawl_timestamp": self.crawl_timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChunkRecord:
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            url=data["url"],
            title=data["title"],
            page_type=data["page_type"],
            headings_path=data["headings_path"],
            content_hash=data["content_hash"],
            crawl_timestamp=data["crawl_timestamp"],
            token_count=data.get("token_count", len(data["text"]) // 4),
        )
