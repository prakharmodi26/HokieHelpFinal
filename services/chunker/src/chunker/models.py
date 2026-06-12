"""ChunkRecord dataclass — the output unit of the chunking service."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List


STRUCTURED_METADATA_FIELDS = (
    "subject",
    "course_number",
    "course_level",
    "credits",
    "prerequisites",
    "corequisites",
    "when_offered",
    "cross_listings",
    "college",
    "program_name",
    "degree_type",
)


def _has_value(value: Any) -> bool:
    return value not in (None, "", [])


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
    source_host: str = ""  # e.g. "dining.vt.edu"
    subdomain: str = ""    # e.g. "dining"
    subject: Any = None
    course_number: Any = None
    course_level: Any = None
    credits: Any = None
    prerequisites: Any = None
    corequisites: Any = None
    when_offered: Any = None
    cross_listings: Any = None
    college: Any = None
    program_name: Any = None
    degree_type: Any = None

    def to_dict(self) -> dict:
        data = {
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
            "source_host": self.source_host,
            "subdomain": self.subdomain,
        }
        for field_name in STRUCTURED_METADATA_FIELDS:
            value = getattr(self, field_name)
            if _has_value(value):
                data[field_name] = value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ChunkRecord:
        structured_metadata = {
            field_name: data.get(field_name)
            for field_name in STRUCTURED_METADATA_FIELDS
        }
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
            source_host=data.get("source_host", ""),
            subdomain=data.get("subdomain", ""),
            **structured_metadata,
        )
