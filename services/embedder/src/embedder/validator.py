"""Validate chunk records before embedding."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a batch of chunks."""
    valid: List[dict] = field(default_factory=list)
    invalid: List[Tuple[dict, str]] = field(default_factory=list)


def validate_chunks(chunks: List[dict]) -> ValidationResult:
    """Validate chunk records. Invalid chunks are skipped with a reason."""
    result = ValidationResult()

    for chunk in chunks:
        reason = _check(chunk)
        if reason:
            result.invalid.append((chunk, reason))
            logger.debug("Invalid chunk %s: %s", chunk.get("chunk_id", "?"), reason)
        else:
            result.valid.append(chunk)

    return result


def _check(chunk: dict) -> str | None:
    """Return a failure reason string, or None if valid."""
    if not chunk.get("chunk_id"):
        return "missing chunk_id"
    if not chunk.get("document_id"):
        return "missing document_id"
    if not (chunk.get("text") or "").strip():
        return "empty text"
    if not chunk.get("url"):
        return "missing url"

    # Title fallback — not a rejection, just a fix-up
    if not chunk.get("title"):
        chunk["title"] = "Untitled"

    return None
