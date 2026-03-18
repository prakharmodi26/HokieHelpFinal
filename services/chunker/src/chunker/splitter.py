"""Token-aware chunking: merge small sections, split large ones with overlap."""
from __future__ import annotations

import hashlib
from typing import Any, List

from chunker.models import ChunkRecord
from chunker.parser import FrontmatterData, Section, infer_page_type


def estimate_tokens(text: str) -> int:
    """Approximate token count as character count // 4 (4 chars ≈ 1 token)."""
    return len(text) // 4


def split_large_section(section: Section, preferred_tokens: int, overlap_tokens: int) -> List[str]:
    """Split an oversized section into windows at paragraph boundaries with overlap.

    Strategy:
    1. Split on blank lines (paragraphs).
    2. Accumulate paragraphs until the window would exceed preferred_tokens.
    3. Flush the window, then rewind by overlap_tokens chars for the next window.
    """
    preferred_chars = preferred_tokens * 4
    overlap_chars = overlap_tokens * 4

    if estimate_tokens(section.text) <= preferred_tokens:
        return [section.text.strip()]

    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]

    windows: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 > preferred_chars and current_parts:
            window_text = "\n\n".join(current_parts)
            windows.append(window_text)
            # Rewind: keep tail of current window for overlap
            overlap_text = window_text[-overlap_chars:]
            current_parts = [overlap_text] if overlap_text.strip() else []
            current_len = len(overlap_text)
        current_parts.append(para)
        current_len += para_len + 2  # +2 for the '\n\n' separator

    if current_parts:
        windows.append("\n\n".join(current_parts))

    return windows if windows else [section.text.strip()]


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_chunks(
    sections: List[Section],
    frontmatter: Any,  # FrontmatterData or duck-typed
    config: Any,       # ChunkerConfig or duck-typed
) -> List[ChunkRecord]:
    """Convert a list of sections into ChunkRecord objects.

    Rules:
    - section < chunk_min_tokens  → accumulate into a pending batch for forward merge
    - section in [min, preferred] → flush pending (if any), emit section as one chunk
    - section > preferred         → flush pending, split into windows with overlap
    """
    preferred = config.chunk_preferred_tokens
    overlap = config.chunk_overlap_tokens
    minimum = config.chunk_min_tokens
    page_type = infer_page_type(frontmatter.url)

    chunk_texts: List[tuple[str, List[str]]] = []  # (text, headings_path)
    pending_parts: List[Section] = []

    def _flush_pending() -> None:
        if not pending_parts:
            return
        merged_text = "\n\n".join(s.text for s in pending_parts)
        merged_path = pending_parts[0].headings_path
        chunk_texts.append((merged_text, merged_path))
        pending_parts.clear()

    for section in sections:
        tokens = estimate_tokens(section.text)

        if tokens > preferred:
            _flush_pending()
            windows = split_large_section(section, preferred, overlap)
            for window in windows:
                chunk_texts.append((window, section.headings_path))
        elif tokens < minimum:
            pending_parts.append(section)
            # If pending accumulated enough, flush
            pending_total = sum(estimate_tokens(s.text) for s in pending_parts)
            if pending_total >= minimum:
                _flush_pending()
        else:
            if pending_parts:
                combined = sum(estimate_tokens(s.text) for s in pending_parts) + tokens
                if combined <= preferred:
                    pending_parts.append(section)
                    _flush_pending()
                else:
                    _flush_pending()
                    chunk_texts.append((section.text, section.headings_path))
            else:
                chunk_texts.append((section.text, section.headings_path))

    _flush_pending()

    records: List[ChunkRecord] = []
    for i, (text, headings_path) in enumerate(chunk_texts):
        records.append(ChunkRecord(
            chunk_id=f"{frontmatter.doc_id}_{i:04d}",
            document_id=frontmatter.doc_id,
            chunk_index=i,
            text=text,
            url=frontmatter.url,
            title=frontmatter.title,
            page_type=page_type,
            headings_path=headings_path,
            content_hash=_chunk_hash(text),
            crawl_timestamp=frontmatter.crawl_timestamp,
        ))
    return records
