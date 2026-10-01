#!/usr/bin/env python3
"""Locate where known indexed CS evidence drops out; uses no generative model.

PYTHONPATH=services/chatbot/src python services/chatbot/scripts/audit_retrieval_coverage.py --port-forward
Queries are hand-specified diagnostics, NOT planner acceptance results. Uses the
production search_many implementation with tracing overrides that only record.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from audit_corpus import TARGETS, markers
from evaluate_pipeline import forward
from chatbot.evidence import select_evidence
from chatbot.retriever import Retriever, canonical_url
from chatbot.program_metadata import program_matches

PRIMARY = "Virginia Tech undergraduate CS (Computer Science) BS degree checksheet total required credit hours"
CASES = [
    ("credits-complete", "how many credits do i need to complete bs in cs", PRIMARY),
    ("credits-graduate", "how many credits do i need to graduate BS in CS", PRIMARY),
    ("credits-undergraduate", "How many credits do i need to complete undergraduate in cs", PRIMARY),
    ("math-followup", "What math courses does it require?",
     "Virginia Tech undergraduate CS Computer Science major checksheet required mathematics courses"),
    ("math-major", "What about the math major?",
     "Virginia Tech undergraduate mathematics major degree required credit hours"),
    ("data-centric", "How many credits for Data-Centric Computing?",
     "Virginia Tech undergraduate Data-Centric Computing major total required credit hours"),
    ("cs-2023", "How many credits for the 2023-24 CS major?", PRIMARY + " for 2023-24"),
    ("admissions", "What are Virginia Tech's first-year admission requirements?",
     "Virginia Tech undergraduate first-year admissions requirements"),
    ("parking", "Where can I get a parking permit at Virginia Tech?",
     "Virginia Tech student parking permit purchase"),
    ("unknown-program", "What are the Chronoengineering degree requirements?",
     "Virginia Tech undergraduate Chronoengineering major degree requirements"),
    ("unknown-person", "What courses does Professor Quasar Noodlewick teach?",
     "Virginia Tech Professor Quasar Noodlewick courses taught"),
]
TARGET_URLS = {canonical_url(t["url"]): t["id"] for t in TARGETS}


def summary(chunk):
    row = {k: chunk[k] for k in ("chunk_id", "url", "title", "rerank_score", "query_ranks", "candidate_lane", "program_name", "program_level", "catalog_edition") if k in chunk}
    row["known_source"] = TARGET_URLS.get(canonical_url(chunk.get("url", "")))
    row["body_markers"] = markers(chunk.get("text", ""))
    return row


class TracingRetriever(Retriever):
    def _candidates(self, query):
        chunks = super()._candidates(query)
        self.trace.append({"query": query, "candidate_count": len(chunks),
                           "known_source_candidates": [dict(rank=i, **summary(c))
                                                       for i, c in enumerate(chunks, 1)
                                                       if canonical_url(c.get("url", "")) in TARGET_URLS]})
        return chunks


def run(args):
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Production Retriever.search_many, hand-specified normalized query + original; no LLM calls",
              "implementation_sha256": hashlib.sha256(b"".join(
                  (Path(__file__).parents[1] / "src" / "chatbot" / name).read_bytes()
                  for name in ("retriever.py", "program_metadata.py", "evidence.py"))).hexdigest(),
              "settings": {"collection": args.collection, "candidate_k": 30, "top_k": 12,
                           "min_score": 0.3, "neighbors": False,
                           "program_candidates_enabled": not args.disable_program_candidates}, "cases": []}
    with ExitStack() as stack:
        if args.port_forward:
            stack.enter_context(forward(args.context, args.namespace, "qdrant", f"{args.qdrant_port}:6333"))
            stack.enter_context(forward(args.context, args.namespace, "embeddings-service", f"{args.embeddings_port}:80"))
        retriever = TracingRetriever("BAAI/bge-m3", "127.0.0.1", args.qdrant_port, args.collection,
                                    embeddings_url=f"http://127.0.0.1:{args.embeddings_port}",
                                    program_candidates_enabled=not args.disable_program_candidates)
        stack.callback(retriever.close)
        for cid, question, primary in CASES:
            retriever.trace = []
            started = time.monotonic()
            selected = retriever.search_many([primary], original_query=question)
            evidence = select_evidence(selected)
            report["cases"].append({"id": cid, "original": question, "normalized": primary,
                                    "searches": retriever.trace,
                                    "selected": [dict(summary(c), program_match=program_matches(primary, c)) for c in selected],
                                    "evidence_chunk_ids": [c.get("chunk_id") for c in evidence],
                                    "seconds": round(time.monotonic() - started, 3)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="hokiehelp-test")
    parser.add_argument("--port-forward", action="store_true")
    parser.add_argument("--disable-program-candidates", action="store_true")
    parser.add_argument("--context", default="endeavour")
    parser.add_argument("--namespace", default="test")
    parser.add_argument("--qdrant-port", type=int, default=16333)
    parser.add_argument("--embeddings-port", type=int, default=18080)
    parser.add_argument("--output", default="docs/evaluations/corpus-retrieval-metadata.json")
    run(parser.parse_args())
