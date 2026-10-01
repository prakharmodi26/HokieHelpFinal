#!/usr/bin/env python3
"""Read-only evaluation of the branch's ACTUAL planner/retriever/evidence/answer code.

PYTHONPATH=services/chatbot/src python services/chatbot/scripts/evaluate_pipeline.py \
  --pi-config --port-forward
No cluster writes or deploy changes. Reports use constructed public questions only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import time

from benchmark_arc import credentials
from chatbot.evidence import NO_EVIDENCE_ANSWER, finalize_answer, select_evidence
from chatbot.llm import LLMClient, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from chatbot.retriever import Retriever

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    ("credits-complete", "how many credits do i need to complete bs in cs", []),
    ("credits-graduate", "how many credits do i need to graduate BS in CS", []),
    ("credits-undergraduate", "How many credits do i need to complete undergraduate in cs", []),
    ("math-followup", "What math courses does it require?", [
        {"role": "user", "content": "Tell me about the Computer Science bachelor's major at Virginia Tech."},
        {"role": "assistant", "content": "Which part of the Computer Science major would you like to know about?"}]),
    ("math-major-switch", "What about the math major?", [
        {"role": "user", "content": "How many credits do I need to graduate with a CS major?"},
        {"role": "assistant", "content": "Requirements depend on the program and entering cohort."}]),
    ("unknown-person", "What courses does Professor Quasar Noodlewick teach at Virginia Tech?", []),
    ("greeting", "Hi!", []),
    ("ambiguous", "When is the deadline?", []),
]


@contextmanager
def forward(context: str, namespace: str, service: str, mapping: str):
    process = subprocess.Popen(["kubectl", "--context", context, "-n", namespace,
                                "port-forward", f"svc/{service}", mapping, "--address=127.0.0.1"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        if not selector.select(timeout=15):
            raise RuntimeError("Port-forward did not become ready")
        if "Forwarding from" not in process.stdout.readline():
            raise RuntimeError("Port-forward failed (details withheld)")
        yield
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


def run(args):
    base, key = credentials(args.pi_config)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "implementation": "production LLMClient + Retriever.search_many + evidence selection/finalizer",
              "prompt_sha256": hashlib.sha256((QUERY_REWRITE_PROMPT + SYSTEM_PROMPT).encode()).hexdigest(),
              "settings": {"planner": args.planner, "answer_models": args.answer_models,
                           "search_original_alongside_plan": True,
                           "collection": args.collection, "candidate_k": 30, "top_k": 12,
                           "context_chars": 24000, "rerank_floor": .3, "neighbors": False},
              "limitations": ["Constructed public questions, not real user logs.",
                              "No golden credit total: program/cohort/source correctness needs human review.",
                              "The branch does not include the newer stack's MCP/version-aware retrieval."],
              "cases": []}
    with ExitStack() as stack:
        if args.port_forward:
            stack.enter_context(forward(args.context, args.namespace, "qdrant", f"{args.qdrant_port}:6333"))
            stack.enter_context(forward(args.context, args.namespace, "embeddings-service", f"{args.embeddings_port}:80"))
        retriever = Retriever("BAAI/bge-m3", "127.0.0.1", args.qdrant_port, args.collection,
                              embeddings_url=f"http://127.0.0.1:{args.embeddings_port}")
        stack.callback(retriever.close)
        planner = LLMClient(key, base, args.answer_models[0], args.planner)
        stack.callback(planner.close)
        answerers = {model: LLMClient(key, base, model) for model in args.answer_models}
        for llm in answerers.values():
            stack.callback(llm.close)
        stream_tested = False
        for cid, question, history in CASES:
            if args.cases and cid not in args.cases:
                continue
            record = {"id": cid, "question": question, "history": history}
            try:
                start = time.monotonic()
                plan = planner.plan_query(question, history)
                record.update(plan=plan.model_dump(), plan_seconds=round(time.monotonic() - start, 3))
                if plan.intent == "search":
                    start = time.monotonic()
                    retrieved = retriever.search_many(plan.queries, original_query=question)
                    chunks = select_evidence(retrieved)
                    record["retrieval_seconds"] = round(time.monotonic() - start, 3)
                    record["retrieved_count"] = len(retrieved)
                    record["context_body_chars"] = sum(len(c["text"]) for c in chunks)
                    record["evidence"] = [{k: c.get(k) for k in (
                        "chunk_id", "source_id", "title", "url", "rerank_score", "query_fusion_score", "context_truncated"
                    )} for c in chunks]
                    record["answers"] = {}
                    for model, llm in answerers.items():
                        start = time.monotonic()
                        raw = llm.chat(question, chunks, history) if chunks else NO_EVIDENCE_ANSWER
                        answer, sources = finalize_answer(raw, chunks) if chunks else (raw, [])
                        record["answers"][model] = {"raw": raw, "final": answer, "sources": sources,
                                                     "seconds": round(time.monotonic() - start, 3)}
                    if args.stream_probe and chunks and not stream_tested:
                        stream_tested = True
                        start = time.monotonic()
                        parts = list(next(iter(answerers.values())).chat_stream(question, chunks, history))
                        answer, sources = finalize_answer("".join(parts), chunks)
                        record["stream_probe"] = {"model": args.answer_models[0], "token_events": len(parts),
                                                  "seconds": round(time.monotonic() - start, 3),
                                                  "answer": answer, "sources": sources}
                else:
                    record["skipped_retrieval"] = True
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                record["http_status"] = getattr(exc, "status_code", None)
            report["cases"].append(record)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2).replace(key, "[REDACTED]") + "\n")
            print(f"{cid}: intent={record.get('plan', {}).get('intent')} chunks={len(record.get('evidence', []))} error={record.get('error_type')}", flush=True)
            if record.get("http_status") in {401, 403, 429}:
                print("Stopping evaluation: provider authorization/rate-limit failure.", flush=True)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-config", nargs="?", const=Path("~/.pi/agent/models.json"), type=Path)
    parser.add_argument("--port-forward", action="store_true")
    parser.add_argument("--context", default="endeavour")
    parser.add_argument("--namespace", default="test")
    parser.add_argument("--qdrant-port", type=int, default=26333)
    parser.add_argument("--embeddings-port", type=int, default=28080)
    parser.add_argument("--collection", default="hokiehelp-test")
    parser.add_argument("--planner", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--answer-models", nargs="+", default=["GLM-5.3", "DeepSeek-V4.1-Flash"])
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--stream-probe", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/evaluations/pipeline-evaluation.json")
    run(parser.parse_args())
