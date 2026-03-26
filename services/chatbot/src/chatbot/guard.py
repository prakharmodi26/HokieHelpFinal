"""Prompt injection and abuse pattern detection.

Raises PromptRejected with a user-safe message for any suspicious input.
All checks are fast string operations — no LLM calls.
"""
from __future__ import annotations

import re

MAX_LENGTH = 2000

# Patterns that indicate prompt injection attempts.
# Lower-cased before matching. Ordered roughly by severity/frequency.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bignore\s+(previous|prior|all|your)\s+(instructions?|prompts?|rules?|context)\b'),
    re.compile(r'\bforget\s+(previous|prior|all|your)\s+(instructions?|prompts?|rules?|context)\b'),
    re.compile(r'\bdisregard\b.{0,40}\b(instructions?|prompts?|rules?|context)\b'),
    re.compile(r'\boverride\b.{0,40}\b(instructions?|prompts?|rules?|context)\b'),
    re.compile(r'\byou\s+are\s+now\b'),
    re.compile(r'\bact\s+as\s+(an?\s+)?(ai|bot|gpt|llm|assistant)\b'),
    re.compile(r'\bjailbreak\b'),
    re.compile(r'\bnew\s+instructions?\b'),
    re.compile(r'\breveal\s+(your\s+)?(system\s+)?prompt\b'),
    re.compile(r'\bprint\s+(your\s+)?(instructions?|system\s+prompt)\b'),
    re.compile(r'\brepeat\s+(everything|all)\s+(above|before)\b'),
    re.compile(r'\bpretend\s+(you\s+are|to\s+be)\b'),
    re.compile(r'\bdo\s+anything\s+now\b'),   # DAN ("Do Anything Now") jailbreak variant
]


class PromptRejected(ValueError):
    """Raised when a prompt fails the injection / abuse guard."""


def check_prompt(text: str) -> None:
    """Validate a user prompt. Raises PromptRejected on any violation.

    This is a fast pre-flight check run before retrieval/LLM calls.
    It does NOT guarantee all injections are caught — it raises the bar
    and rejects the most common patterns.

    Args:
        text: The raw user question string.

    Raises:
        PromptRejected: If the prompt is empty, too long, or matches an
            injection pattern.
    """
    if not text or not text.strip():
        raise PromptRejected("Question must not be empty.")

    if len(text) > MAX_LENGTH:
        raise PromptRejected(
            f"Question exceeds the maximum allowed length of {MAX_LENGTH} characters."
        )

    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            raise PromptRejected(
                "Your question contains patterns that are not allowed. "
                "Please ask a question about Virginia Tech's CS department."
            )
