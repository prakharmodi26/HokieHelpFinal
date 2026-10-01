#!/usr/bin/env python3
"""Benchmark the ACTUAL planner/answer prompts on synthetic fixtures.

Run with PYTHONPATH=services/chatbot/src and the chatbot dependencies installed.
Credentials come from LLM_API_KEY, or explicitly opt in with --pi-config.
No fallback to another provider; no user logs are sent. Heuristics are NOT gold
semantic judgments. Full outputs are saved for human review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import httpx
from chatbot.evidence import finalize_answer, select_evidence
from chatbot.llm import LLMClient, QueryPlan, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT, build_rewrite_messages

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/query_planning_cases.json"


def credentials(pi_config: Path | None) -> tuple[str, str]:
    key = os.environ.get("LLM_API_KEY")
    base = os.environ.get("LLM_BASE_URL", "https://llm-api.arc.vt.edu/api/v1")
    if key:
        return base, key
    if pi_config is None:
        raise SystemExit("Set LLM_API_KEY or explicitly pass --pi-config")
    provider = json.loads(pi_config.expanduser().read_text())["providers"]["vt-arc"]
    raw = provider["apiKey"]
    if raw.startswith("!"):
        try:
            result = subprocess.run(raw[1:], shell=True, capture_output=True, text=True, timeout=10)
        except Exception:
            raise SystemExit("Credential helper failed (details withheld)") from None
        if result.returncode:
            raise SystemExit("Credential helper failed (details withheld)")
        key = result.stdout.strip()
    else:
        key = os.path.expandvars(raw)
        if "$" in key:
            raise SystemExit("Unresolved credential environment variable")
    if not key:
        raise SystemExit("Empty credential")
    return provider["baseUrl"], key


def terms_present(text: str, groups: list[list[str]]) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    return all(any(term.lower() in normalized for term in group) for group in groups)


def run(args) -> dict:
    base, key = credentials(args.pi_config)
    fixtures = json.loads(FIXTURES.read_text())
    cases = fixtures["planner_cases"]
    probes = fixtures["answer_probes"]
    if not args.all_cases:
        wanted = {"constraint_dense_single", "history_pronoun_resolution", "history_constraint_swap", "ambiguous_unresolvable_reference", "greeting_basic"}
        cases = [c for c in cases if c["id"] in wanted]
        probes = probes[:2]
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "endpoint": base,
              "implementation": "chatbot.llm production prompts + QueryPlan schema + evidence finalizer",
              "prompt_sha256": hashlib.sha256((QUERY_REWRITE_PROMPT + SYSTEM_PROMPT).encode()).hexdigest(),
              "settings": {"max_tokens": 2048, "planner_max_tokens": 1024, "timeout": args.timeout},
              "limitations": ["Synthetic fixture string checks are not semantic gold.",
                              "Citation integrity does not establish claim entailment.",
                              "One sample per case/model; endpoint load affects latency."], "models": {}}
    for model in args.models:
        llm = LLMClient(key, base, model, model, timeout=args.timeout, rewrite_timeout=args.timeout)
        results = []
        try:
            for kind, case in [("planner", c) for c in cases] + [("answer", p) for p in probes]:
                record = {"kind": kind, "id": case["id"]}
                start = time.monotonic()
                try:
                    if kind == "planner":
                        body = {"model": model, "temperature": 0, "max_tokens": 1024,
                                "response_format": {"type": "json_object"},
                                "messages": build_rewrite_messages(case["user_message"], case["history"])}
                    else:
                        chunks = select_evidence(case["context_chunks"])
                        body = llm._body(case["question"], chunks, [])
                    response = llm._client.post("chat/completions", json=body)
                    output = llm._completion(response)
                    record.update(output=output, finish_reason=response.json()["choices"][0].get("finish_reason"))
                    if kind == "planner":
                        plan = QueryPlan.model_validate_json(output)
                        expected = case["expected"]
                        text = " ".join(plan.queries).lower()
                        checks = {
                            "valid_schema": True, "bounded_queries": all(len(q) <= 500 for q in plan.queries),
                            "intent": plan.intent == expected["intent"],
                            "query_count": expected["query_count_min"] <= len(plan.queries) <= expected["query_count_max"],
                            "constraints": terms_present(text, expected["must_include_any"]),
                            "no_forbidden": not any(term.lower() in text for term in expected["must_not_include"]),
                        }
                    else:
                        final, sources = finalize_answer(output, chunks)
                        checks = {"required_terms": terms_present(output, case["must_include_any"]),
                                  "no_forbidden": not any(t.lower() in output.lower() for t in case["must_not_include_any"])}
                        if case["expect"] == "grounded":
                            checks["valid_citations"] = bool(sources)
                        else:
                            # Negative probes need human inspection; do not claim a
                            # substring-based test proves that no fact was invented.
                            record["requires_refusal_review"] = True
                        record["final_answer"] = final
                        record["cited_ids"] = [s["id"] for s in sources]
                    record["checks"] = checks
                    record["passed_heuristics"] = all(checks.values())
                except Exception as exc:
                    record.update(error_type=type(exc).__name__, passed_heuristics=False)
                    if isinstance(exc, httpx.HTTPStatusError):
                        record["http_status"] = exc.response.status_code
                record["seconds"] = round(time.monotonic() - start, 3)
                results.append(record)
                print(f"{model} {kind}/{case['id']}: heuristic_pass={record['passed_heuristics']} {record['seconds']}s", flush=True)
                report["models"][model] = results
                # Incremental sanitized checkpoint survives interrupted evaluation.
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2).replace(key, "[REDACTED]") + "\n")
                if record.get("http_status") in {401, 403, 429}:
                    print("Stopping benchmark: provider authorization/rate-limit failure.", flush=True)
                    return report
        finally:
            llm.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-config", nargs="?", const=Path("~/.pi/agent/models.json"), type=Path)
    parser.add_argument("--models", nargs="+", default=["GLM-5.3", "DeepSeek-V4.1-Flash"])
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/evaluations/arc-model-benchmark.json")
    run(parser.parse_args())
