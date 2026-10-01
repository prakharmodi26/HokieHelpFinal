"""Bound evidence sent upstream and bind citations to that exact evidence set.

Citation validation checks reference integrity, not whether a claim is entailed.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import unquote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)
NO_EVIDENCE_ANSWER = (
    "I couldn't find enough reliable information in the available Virginia Tech "
    "sources to answer that. Please clarify the program, topic, or year if relevant."
)
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


def source_url(value: str) -> str:
    """Accept only navigable web sources; remove fragments for page-level IDs."""
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        if parts.username or parts.password:
            return ""
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    except (ValueError, AttributeError):
        return ""


def select_evidence(
    chunks: list[dict], *, max_chunks: int = 12,
    max_chars: int = 24000, max_chunk_chars: int = 4000,
) -> list[dict]:
    """Select ranked hits, no neighbors; apply a hard rendered-context char budget.

    Character limits are conservative operational bounds, not token counts. The
    total LLM prompt additionally contains separately bounded history/instructions.
    Titles/URLs are included in the budget. Never mutate retrieval results.
    """
    from chatbot.llm import _clean_chunk_text
    from chatbot.retriever import canonical_url

    selected: list[dict] = []
    source_ids: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    remaining = max_chars
    for chunk in chunks:
        if len(selected) >= max_chunks or remaining <= 0:
            break
        url = source_url(chunk.get("url") or "")
        text = _clean_chunk_text(chunk.get("text") or "")
        if not url or not text:
            continue
        headings = chunk.get("headings_path") or []
        if isinstance(headings, list):
            section = " > ".join(str(h) for h in headings)[:300]
            if section:
                text = f"Section: {section}\n{text}"
        identity = []
        for key, label, limit in (("program_name", "Program", 200),
                                  ("program_level", "Degree level", 40),
                                  ("catalog_edition", "Catalog edition", 40)):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                value = re.sub(r"\s+", " ", value).strip()[:limit]
                identity.append(f"{label}: {value}")
        if identity:
            text = "\n".join(identity) + "\n" + text
        page_key = canonical_url(url)
        if (page_key, text) in seen:
            continue
        seen.add((page_key, text))
        sid = source_ids.get(page_key, len(source_ids) + 1)
        title = unquote((chunk.get("title") or url).strip())[:300]
        overhead = len(f"[{sid}] {title} — {url}\n") + len("\n\n---\n\n")
        allowance = min(max_chunk_chars, remaining - overhead)
        if allowance < 100 and len(text) > allowance:
            continue
        if allowance <= 0:
            continue
        truncated = len(text) > allowance
        body = text[:allowance]
        if truncated:
            # Prefer a word boundary without violating the hard limit.
            cut = body.rfind(" ")
            if cut > allowance // 2:
                body = body[:cut]
        selected.append({**chunk, "text": body, "title": title, "url": url,
                         "source_id": sid, "context_truncated": truncated})
        source_ids[page_key] = sid
        remaining -= len(body) + overhead
    logger.info("EVIDENCE candidates=%d selected=%d pages=%d context_chars=%d limit=%d",
                len(chunks), len(selected), len(source_ids), max_chars - remaining, max_chars)
    return selected


def finalize_answer(answer: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Return only cited sources, rejecting empty/missing/unknown references.

    This intentionally does not label a valid numeric reference as proof of
    factual correctness. Grounding still needs model/corpus evaluation.
    """
    sources: dict[int, dict] = {}
    for chunk in chunks:
        sid = chunk["source_id"]
        sources.setdefault(sid, {"id": sid, "title": chunk["title"], "url": chunk["url"]})
    cited: list[int] = []
    for match in _CITATION.finditer(answer):
        for value in match.group(1).split(","):
            if len(value.strip()) > 10:
                return NO_EVIDENCE_ANSWER, []
            sid = int(value.strip())
            if sid not in sources:
                logger.warning("CITATION rejected unknown_source_id=%d", sid)
                return NO_EVIDENCE_ANSWER, []
            if sid not in cited:
                cited.append(sid)
    if not answer.strip() or not cited:
        logger.info("CITATION no usable citations; returning evidence fallback")
        return NO_EVIDENCE_ANSWER, []
    # Normalize grouped citations for the frontend, preserving assigned IDs.
    normalized = _CITATION.sub(
        lambda m: "".join(f"[{int(v.strip())}]" for v in m.group(1).split(",")), answer.strip()
    )
    return normalized, [sources[sid] for sid in cited]
