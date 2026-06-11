"""Post-ingestion verification: completeness + content quality + failure accounting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_WORDS = 25


@dataclass
class DocAssessment:
    key: str
    ok: bool
    word_count: int
    reasons: list[str] = field(default_factory=list)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def assess_document(key: str, content: str) -> DocAssessment:
    reasons: list[str] = []
    if not content.startswith("---"):
        reasons.append("missing frontmatter")
        body = content
    else:
        end = content.find("---", 3)
        body = content[end + 3:] if end != -1 else content
    stripped = body.strip()
    wc = _word_count(stripped)
    if not stripped:
        reasons.append("empty body")
    elif wc < MIN_WORDS:
        reasons.append(f"too short ({wc} words)")
    # crude global-nav leakage check (should have been pruned by fit_markdown)
    if "Skip to main content" in body or "Universal Access" in body:
        reasons.append("boilerplate leaked")
    ok = len(reasons) == 0
    return DocAssessment(key=key, ok=ok, word_count=wc, reasons=reasons)


def summarize(rows: list[DocAssessment]) -> dict:
    histogram: dict[str, int] = {}
    for r in rows:
        for reason in r.reasons:
            key = re.sub(r"\(\d+ words\)", "(N words)", reason)
            histogram[key] = histogram.get(key, 0) + 1
    return {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "failed": sum(1 for r in rows if not r.ok),
        "reason_histogram": histogram,
    }


def render_report(summary: dict, sample_failures: list[DocAssessment]) -> str:
    lines = ["# Ingestion Verification Report", "",
             f"- Total documents: {summary['total']}",
             f"- Clean (ok): {summary['ok']}",
             f"- Failed: {summary['failed']}", "",
             "## Failure reasons", ""]
    for reason, count in sorted(summary["reason_histogram"].items(), key=lambda x: -x[1]):
        lines.append(f"- {reason}: {count}")
    lines += ["", "## Sample failures", ""]
    for a in sample_failures[:25]:
        lines.append(f"- `{a.key}` — {', '.join(a.reasons)}")
    return "\n".join(lines) + "\n"
