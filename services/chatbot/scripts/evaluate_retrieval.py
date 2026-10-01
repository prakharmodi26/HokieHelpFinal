#!/usr/bin/env python3
"""Live retrieval quality probe for HokieHelp — READ-ONLY.

Runs synthetic public Virginia Tech queries through the *live* retrieval
stack (embeddings-service HTTP API + Qdrant hybrid RRF + bge-reranker-v2-m3)
and records descriptive results: cosine / fusion / rerank scores, candidate
counts at each stage, and selected chunk ids/titles/urls (NO corpus bodies).

Pipeline mirrored 1:1 from the deployed chatbot retriever (image :0716):
  1. embed raw query            POST {embeddings}/embed        (BAAI/bge-m3, no query prefix)
  2. hybrid search              Qdrant query_points            (prefetch dense+sparse, RRF fusion)
  3. content dedup              chunk_key = canonical_url|sha1(text)
  4. cross-encoder rerank       POST {embeddings}/rerank       (bge-reranker-v2-m3, passage =
                                                                title + headings + body)
  5. score floor + top_k slice  floor=0.3 (raw rerank score)

Differences vs. deployed retrieval (intentional, matching the planned
search_many worker): candidate_k=30 (prod: 50), top_k compared at 5 AND 12
(prod: 8), NO adjacent-chunk expansion, and per-stage candidate counts are
recorded. The version partition is a no-op on hokiehelp-test today (no
edition payload fields present), so it is not applied.

Usage:
  python evaluate_retrieval.py \
      --embeddings-url http://localhost:18000 \
      --qdrant-url http://localhost:16333 \
      --out ../../docs/evaluations/retrieval-baseline.json \
      --evidence-out /tmp/hokiehelp-live-evidence.json

Requires: httpx, qdrant-client (no torch — embeddings are computed by the
service, never locally).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from qdrant_client import QdrantClient, models

# --------------------------------------------------------------------------
# embed-text composition — byte-identical logic to the live
# services/ingestor/src/ingestor/embed_text.py (EMBED_VERSION=2,
# "identity-composed" vectors). Verified against live vectors: re-embedding
# the composed text reproduces stored dense vectors at cosine 1.000000.
# --------------------------------------------------------------------------
_MD_EMPHASIS = re.compile(r"[*_`]+")
_LEADING_HASHES = re.compile(r"^\s*#{1,6}\s+")
_WS = re.compile(r"\s+")
_TITLE_MAX_CHARS = 300


def _norm(s: str) -> str:
    from urllib.parse import unquote

    decoded = unquote(s)
    if "\ufffd" in decoded and "\ufffd" not in s:
        decoded = s
    decoded = _LEADING_HASHES.sub("", decoded)
    decoded = _MD_EMPHASIS.sub(" ", decoded)
    return _WS.sub(" ", decoded).strip()


def compose_embed_text(title: str, headings_path: list[str], body: str) -> str:
    norm_title = _norm(title)[:_TITLE_MAX_CHARS]
    headings = [h for h in (_norm(h) for h in headings_path) if h]
    if norm_title:
        headings = [h for h in headings if h.lower() != norm_title.lower()]
    lines: list[str] = []
    if norm_title:
        lines.append(norm_title)
    if headings:
        lines.append(" > ".join(headings))
    if not lines:
        return body
    return "\n".join(lines) + "\n\n" + body


# --------------------------------------------------------------------------
# canonical URL + chunk dedup key — logic mirrored from the live
# services/chatbot/src/chatbot/models.py (chunk_key / canonical_url).
# --------------------------------------------------------------------------
_TRACKING_KEYS = {"gclid", "fbclid", "mc_cid", "mc_eid"}
_INDEX_SUFFIXES = ("/index.html", "/index.htm", "/index.php")


def canonical_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url.lower()
    host = (parts.netloc or "").lower()
    if parts.scheme == "http":
        host = host.removesuffix(":80")
    elif parts.scheme == "https":
        host = host.removesuffix(":443")
    host = host.removeprefix("www.")
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    for suffix in _INDEX_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
            break
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS
    )
    qs = urlencode(query)
    return f"{host}{path}" + (f"?{qs}" if qs else "")


def chunk_key(url: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "empty"
    return f"{canonical_url(url)}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------
class EmbeddingsClient:
    """Sync client for the deployed embeddings-service (BGE-M3 + reranker)."""

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> tuple[list[float], dict]:
        for attempt in range(3):
            try:
                r = self._client.post(f"{self._base}/embed", json={"texts": [text]})
                r.raise_for_status()
                data = r.json()
                return data["dense"][0], data["sparse"][0]
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"  ! embed retry {attempt + 1}: {exc}", file=sys.stderr)
                time.sleep(2.0)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        for attempt in range(3):
            try:
                r = self._client.post(
                    f"{self._base}/rerank",
                    json={"query": query, "documents": documents, "top_n": top_n},
                )
                r.raise_for_status()
                data = r.json()
                pairs = [(int(res["index"]), float(res["score"])) for res in data["results"]]
                return sorted(pairs, key=lambda p: p[1], reverse=True)
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"  ! rerank retry {attempt + 1}: {exc}", file=sys.stderr)
                time.sleep(2.0)


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------
@dataclass
class QueryCase:
    qid: str
    category: str
    query: str
    notes: str = ""


# Heuristic wrong-program / wrong-level markers (descriptive flags only — NOT
# ground truth). Matched against title + url.
WRONG_PROGRAM_MARKERS = {
    "cs": [
        "cals", "crop", "soil", "animal", "statistic", "chemistry",
        "data-centric", "data centric", "secure computing", "computer engineering",
        "master of science", "masters", "minor",
    ],
    "math": ["computer science", "computer engineering", "statistic", "cals"],
}

EVIDENCE_QUERIES = {  # qids whose full chunk texts go to the (untracked) evidence file
    "github29-p1",
    "github29-normalized",
    "cs-major",
    "math-major",
    "followup-composed",
}


def build_suite() -> list[QueryCase]:
    return [
        # GitHub #29 exact regression paraphrases (credit-count question)
        QueryCase("github29-p1", "github29-paraphrase",
                  "how many credits do i need to complete bs in cs"),
        QueryCase("github29-p2", "github29-paraphrase",
                  "how many credits do i need to graduate BS in CS"),
        QueryCase("github29-p3", "github29-paraphrase",
                  "How many credits do i need to complete undergraduate in cs"),
        QueryCase("github29-normalized", "github29-paraphrase",
                  "Virginia Tech Bachelor of Science in Computer Science graduation credit requirements"),
        # CS program queries (GitHub #30 wrong-program surface)
        QueryCase("cs-major", "cs-program",
                  "What are the requirements for the Computer Science major at Virginia Tech?"),
        QueryCase("cs-major-short", "cs-program",
                  "computer science major requirements"),
        # Math program queries (cross-program confusion surface)
        QueryCase("math-major", "math-program",
                  "What are the requirements for the Mathematics major at Virginia Tech?"),
        QueryCase("math-major-short", "math-program",
                  "mathematics major requirements"),
        QueryCase("stats-major", "sibling-program",
                  "statistics major requirements Virginia Tech"),
        QueryCase("cs-sibling-dcc", "sibling-program",
                  "What is the Data-Centric Computing major at Virginia Tech?"),
        QueryCase("cs-vs-cpe", "cross-program",
                  "difference between computer science and computer engineering at Virginia Tech"),
        # GitHub #16 followup handling (history: user was asking about the CS major)
        QueryCase("followup-raw", "followup",
                  "What math courses does it require?",
                  notes="raw followup; live system relies on LLM condensation to resolve 'it'"),
        QueryCase("followup-composed", "followup",
                  "What math courses does the Computer Science major at Virginia Tech require?",
                  notes="context-composed variant of the same followup"),
        # Faculty primary + followup
        QueryCase("faculty-cs-head", "faculty",
                  "Who is the head of the Computer Science department at Virginia Tech?"),
        QueryCase("faculty-followup-raw", "followup",
                  "What courses do they teach?",
                  notes="raw followup after faculty-cs-head; pronoun-only, no program tokens"),
        # University-scope queries (scope mismatch surface)
        QueryCase("scope-financial-aid", "university-scope",
                  "How do I apply for financial aid at Virginia Tech?"),
        QueryCase("scope-parking", "university-scope",
                  "Where can I park on campus as a student?"),
        QueryCase("scope-admission", "university-scope",
                  "What are the admission requirements for Virginia Tech?"),
        # Unknown entities (threshold behaviour on out-of-corpus names)
        QueryCase("unknown-faculty", "unknown-entity",
                  "Tell me about Professor Maribel Quentin in the Virginia Tech Computer Science department",
                  notes="fictional name (synthetic)"),
        QueryCase("unknown-program", "unknown-entity",
                  "What is the Underwater Basket Weaving major at Virginia Tech?",
                  notes="fictional program (synthetic)"),
    ]


def run_query(
    qdrant: QdrantClient,
    emb: EmbeddingsClient,
    collection: str,
    case: QueryCase,
    candidates: int,
    floor: float,
    top_k_variants: list[int],
) -> tuple[dict, list[dict]]:
    """Run one query through the live pipeline; return (sanitized_result, evidence_chunks)."""
    t0 = time.perf_counter()

    # 1. embed raw query (live code embeds the raw user query — no BGE prefix)
    dense, sparse = emb.embed(case.query)
    t_embed = time.perf_counter() - t0

    # 2a. dense-only (cosine scores for the report)
    dense_res = qdrant.query_points(
        collection_name=collection,
        query=dense,
        using="dense",
        limit=candidates,
        with_payload=True,
    )
    dense_top = [
        {
            "title": ((p.payload or {}).get("title") or "")[:70],
            "url": ((p.payload or {}).get("url") or "")[:130],
            "cosine": round(p.score, 4),
        }
        for p in dense_res.points[:5]
    ]

    # 2b. sparse-only (count only — diagnostic)
    sparse_count = 0
    try:
        if sparse.get("indices"):
            sparse_res = qdrant.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=sparse["indices"], values=sparse["values"]
                ),
                using="sparse",
                limit=candidates,
                with_payload=False,
            )
            sparse_count = len(sparse_res.points)
    except Exception as exc:  # sparse failures are diagnostic-only
        print(f"  ! sparse-only query failed: {exc}", file=sys.stderr)

    # 2c. hybrid RRF (the live primary path)
    fused_res = qdrant.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=candidates),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse["indices"], values=sparse["values"]
                ),
                using="sparse",
                limit=candidates,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidates,
        with_payload=True,
    )
    fused_hits = fused_res.points
    dense_cosine_by_id = {p.payload.get("chunk_id"): p.score for p in dense_res.points}
    rrf_top = [
        {
            "title": ((p.payload or {}).get("title") or "")[:70],
            "url": ((p.payload or {}).get("url") or "")[:130],
            "rrf": round(p.score, 4),
        }
        for p in fused_hits[:5]
    ]
    t_search = time.perf_counter() - t0 - t_embed

    # 3. content dedup (URL-variant near-duplicates collapse before rerank)
    unique: dict[str, Any] = {}
    for h in fused_hits:
        pl = h.payload or {}
        unique.setdefault(chunk_key(pl.get("url") or "", pl.get("text") or ""), h)
    deduped = list(unique.values())
    dedup_dropped = len(fused_hits) - len(deduped)

    # 4. cross-encoder rerank over identity-composed passages
    passages = [
        compose_embed_text(
            (h.payload or {}).get("title") or "",
            (h.payload or {}).get("headings_path") or [],
            (h.payload or {}).get("text") or "",
        )
        for h in deduped
    ]
    ranked = emb.rerank(case.query, passages, top_n=len(passages))
    t_rerank = time.perf_counter() - t0 - t_embed - t_search

    # 5. floor + top_k slices (raw rerank score, floor applies before slicing)
    kept = [(idx, score) for idx, score in ranked if score >= floor]

    def selected(top_k: int, detailed: bool) -> list[dict]:
        out = []
        for rank, (idx, score) in enumerate(kept[:top_k], 1):
            pl = deduped[idx].payload or {}
            cid = pl.get("chunk_id", "")
            entry = {
                "rank": rank,
                "title": (pl.get("title") or "")[:80],
                "url": (pl.get("url") or "")[:130],
                "canonical_url": canonical_url(pl.get("url") or ""),
                "rerank": round(score, 4),
                "text_chars": len(pl.get("text") or ""),
                "dense_cosine": (
                    round(dense_cosine_by_id[cid], 4)
                    if cid in dense_cosine_by_id else None
                ),
            }
            if detailed:
                entry["chunk_id"] = cid
                entry["headings"] = " > ".join(pl.get("headings_path") or [])[:100]
                entry["page_type"] = pl.get("page_type", "")
                entry["source_host"] = pl.get("source_host", "")
            out.append(entry)
        return out

    base_k = min(top_k_variants)
    max_k = max(top_k_variants)
    sel_base = selected(base_k, detailed=True)
    sel_max = selected(max_k, detailed=False)
    max_k_dense_hits = sum(1 for s in sel_max if s["dense_cosine"] is not None)

    result = {
        "qid": case.qid,
        "category": case.category,
        "query": case.query,
        "notes": case.notes,
        "timings_s": {
            "embed": round(t_embed, 3),
            "qdrant_3_queries": round(t_search, 3),
            "rerank": round(t_rerank, 3),
        },
        "candidate_counts": {
            "prefetch_limit_per_branch": candidates,
            "dense_branch_returned": len(dense_res.points),
            "sparse_branch_returned": sparse_count,
            "fused_returned": len(fused_hits),
            "after_dedup": len(deduped),
            "dedup_dropped": dedup_dropped,
            "above_floor": len(kept),
        },
        "dense_top5": dense_top,
        "rrf_top5": rrf_top,
        "rerank_score_range": (
            {"max": round(ranked[0][1], 4), "min": round(ranked[-1][1], 4)}
            if ranked else None
        ),
        f"selected_top{base_k}": sel_base,
        f"selected_top{base_k}_total_chars": sum(s["text_chars"] for s in sel_base),
        f"selected_top{max_k}_beyond_top{base_k}": [
            s for s in sel_max if s["rank"] > base_k
        ],
        f"selected_top{max_k}_total_chars": sum(s["text_chars"] for s in sel_max),
        f"selected_top{max_k}_dense_branch_hits": max_k_dense_hits,
    }

    # evidence (full text — ONLY for the untracked /tmp file)
    evidence = []
    if case.qid in EVIDENCE_QUERIES:
        for rank, (idx, score) in enumerate(kept[:max_k], 1):
            pl = deduped[idx].payload or {}
            evidence.append(
                {
                    "rank": rank,
                    "chunk_id": pl.get("chunk_id", ""),
                    "title": pl.get("title") or "",
                    "url": pl.get("url") or "",
                    "headings_path": pl.get("headings_path") or [],
                    "rerank_score": round(score, 4),
                    "text": pl.get("text") or "",
                }
            )
    return result, evidence


# --------------------------------------------------------------------------
# Aggregate analysis (descriptive only — no invented ground truth)
# --------------------------------------------------------------------------
def _wrong_program_flags(selections: list[dict], profile: str) -> list[dict]:
    markers = WRONG_PROGRAM_MARKERS[profile]
    flags = []
    for s in selections:
        blob = f"{s['title']} {s['url']}".lower()
        hit = next((m for m in markers if m in blob), None)
        if hit:
            flags.append(
                {
                    "rank": s["rank"],
                    "marker": hit,
                    "title": s["title"],
                    "url": s["url"],
                    "rerank": s["rerank"],
                }
            )
    return flags


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def analyze(results: list[dict]) -> dict:
    by_id = {r["qid"]: r for r in results}
    agg: dict[str, Any] = {}

    # 1. Paraphrase consistency (GitHub #29): canonical-URL set overlap, top-5
    group = ["github29-p1", "github29-p2", "github29-p3", "github29-normalized",
             "cs-major", "cs-major-short"]
    sets = {
        qid: {s["canonical_url"] for s in by_id[qid]["selected_top5"]}
        for qid in group if qid in by_id
    }
    pairs = {}
    ids = sorted(sets)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pairs[f"{a} vs {b}"] = round(jaccard(sets[a], sets[b]), 3)
    agg["paraphrase_top5_url_jaccard"] = pairs
    common = set.intersection(*sets.values()) if sets else set()
    agg["paraphrase_shared_urls_all_variants"] = sorted(
        common
    ) if len(sets) > 1 else sorted(next(iter(sets.values()), set()))

    # 2. Wrong-program / wrong-level flags (GitHub #30) on CS-profile queries
    cs_qids = ["github29-p1", "github29-p2", "github29-p3", "github29-normalized",
               "cs-major", "cs-major-short", "followup-composed", "followup-raw"]

    def top12_of(qid: str) -> list[dict]:
        return by_id[qid]["selected_top5"] + by_id[qid].get("selected_top12_beyond_top5", [])

    agg["wrong_program_flags_cs_queries"] = {
        qid: {
            "top5": _wrong_program_flags(by_id[qid]["selected_top5"], "cs"),
            "top12": _wrong_program_flags(top12_of(qid), "cs"),
        }
        for qid in cs_qids if qid in by_id
    }
    math_qids = ["math-major", "math-major-short"]
    agg["wrong_program_flags_math_queries"] = {
        qid: {
            "top5": _wrong_program_flags(by_id[qid]["selected_top5"], "math"),
            "top12": _wrong_program_flags(top12_of(qid), "math"),
        }
        for qid in math_qids if qid in by_id
    }

    # 3. Followup handling (GitHub #16)
    if "followup-raw" in by_id and "followup-composed" in by_id:
        agg["followup_raw_vs_composed_top5_url_jaccard"] = round(
            jaccard(
                {s["canonical_url"] for s in by_id["followup-raw"]["selected_top5"]},
                {s["canonical_url"] for s in by_id["followup-composed"]["selected_top5"]},
            ), 3)
    if "faculty-followup-raw" in by_id:
        r = by_id["faculty-followup-raw"]
        agg["faculty_followup_raw"] = {
            "above_floor": r["candidate_counts"]["above_floor"],
            "max_rerank": r["rerank_score_range"]["max"] if r["rerank_score_range"] else None,
        }

    # 4. Unknown-entity threshold behaviour
    agg["unknown_entity_max_rerank_scores"] = {
        qid: (by_id[qid]["rerank_score_range"] or {}).get("max")
        for qid in ("unknown-faculty", "unknown-program") if qid in by_id
    }
    agg["unknown_entity_above_floor"] = {
        qid: by_id[qid]["candidate_counts"]["above_floor"]
        for qid in ("unknown-faculty", "unknown-program") if qid in by_id
    }

    # 5. top_k 5 vs 12 — extra documents, dense-branch reliance, character budget
    extra_docs, chars, dense_hits = {}, {}, {}
    for r in results:
        top5_urls = {s["canonical_url"] for s in r["selected_top5"]}
        beyond = r.get("selected_top12_beyond_top5", [])
        docs12 = top5_urls | {s["canonical_url"] for s in beyond}
        extra_docs[r["qid"]] = {
            "top5_distinct_docs": len(top5_urls),
            "top12_distinct_docs": len(docs12),
            "docs_added_by_top12": len(docs12 - top5_urls),
        }
        chars[r["qid"]] = {
            "top5_total_chars": r.get("selected_top5_total_chars"),
            "top12_total_chars": r.get("selected_top12_total_chars"),
        }
        dense_hits[r["qid"]] = r.get("selected_top12_dense_branch_hits")
    agg["top5_vs_top12_docs"] = extra_docs
    agg["selected_chars_per_query"] = chars
    agg["selected_top12_dense_branch_hits"] = dense_hits
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-url", default="http://localhost:18000")
    ap.add_argument("--qdrant-url", default="http://localhost:16333")
    ap.add_argument("--collection", default="hokiehelp-test")
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, nargs="+", default=[5, 12])
    ap.add_argument("--only", nargs="*", help="restrict to these qids")
    ap.add_argument("--out", default=None, help="sanitized JSON report path")
    ap.add_argument("--evidence-out", default=None,
                    help="full-text evidence JSON path (keep OUT of the repo)")
    args = ap.parse_args()

    emb = EmbeddingsClient(args.embeddings_url)
    qdrant = QdrantClient(url=args.qdrant_url, timeout=60.0)

    # sanity: service health + model identity (no secrets involved)
    health = emb._client.get(f"{args.embeddings_url}/health").json()
    col_info = qdrant.get_collection(args.collection)
    print(f"embeddings health: {health}")
    print(f"collection {args.collection}: points={col_info.points_count} "
          f"vectors={col_info.config.params.vectors}")

    suite = build_suite()
    if args.only:
        suite = [c for c in suite if c.qid in args.only]

    results, evidence = [], {}
    for case in suite:
        print(f"[{case.qid}] {case.query!r} ...", flush=True)
        try:
            r, ev = run_query(qdrant, emb, args.collection, case,
                              args.candidates, args.floor, args.top_k)
        except Exception as exc:
            print(f"  !! FAILED: {exc}", file=sys.stderr)
            results.append({"qid": case.qid, "category": case.category,
                            "query": case.query, "error": str(exc)})
            continue
        results.append(r)
        if ev:
            evidence[case.qid] = {
                "query": case.query,
                "category": case.category,
                "chunks": ev,
            }
        kept = r["candidate_counts"]["above_floor"]
        top1 = r["selected_top5"][0] if r["selected_top5"] else None
        print(f"  fused={r['candidate_counts']['fused_returned']} "
              f"deduped={r['candidate_counts']['after_dedup']} above_floor={kept} "
              f"| top1: {top1['title'][:60] if top1 else '(none)'} "
              f"(rerank={top1['rerank'] if top1 else '-'})")

    agg = analyze([r for r in results if "error" not in r])

    report = {
        "purpose": "Read-only live retrieval quality probe against the deployed "
                   "HokieHelp stack (embeddings-service + Qdrant). Descriptive only — "
                   "no invented ground truth; credit totals are NOT judged.",
        "environment": {
            "collection": args.collection,
            "points": col_info.points_count,
            "vector_config": {"dense": "1024/Cosine (named 'dense')",
                              "sparse": "lexical, idf modifier (named 'sparse')"},
            "embedding_model": "BAAI/bge-m3 (per embeddings-config configmap; "
                               "verified by re-embedding stored chunks at cosine 1.000000)",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "embed_text": "identity-composed (title + headings + body), embed_version=2",
            "query_prefix": "none (raw query text, matching live chatbot)",
        },
        "parameters": {
            "candidates": args.candidates,
            "rerank_score_floor": args.floor,
            "top_k_variants": args.top_k,
            "adjacent_chunk_expansion": "disabled (matches planned search_many worker)",
            "production_diff": "prod uses candidates=50, top_k=8; version partition is a "
                               "no-op on this collection (no edition payload fields present)",
        },
        "results": results,
        "aggregate": agg,
        "limitations": [
            "Single run per query; no repeat-variance estimate (GPU rerank scores are "
            "deterministic to ~1e-4, but service load varies).",
            "Scores are descriptive; no golden relevance judgments — wrong-program flags "
            "are title/heading keyword heuristics, not verified labels.",
            "Followup queries are tested as standalone retrievals (raw) and as "
            "hand-composed variants; the live condenser LLM was not exercised.",
            "Candidate count fixed at 30 per branch (prod: 50); only the planned "
            "search_many configuration was probed.",
        ],
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsanitized report -> {args.out}")
    if args.evidence_out and evidence:
        with open(args.evidence_out, "w") as f:
            json.dump(
                {
                    "purpose": "Full-text retrieved evidence for synthetic public VT "
                               "queries — input for LLM grounding tests. NOT for commit.",
                    "generated_from": {"collection": args.collection,
                                       "floor": args.floor, "top_k": 12},
                    "queries": evidence,
                },
                f, indent=2)
        print(f"full-text evidence -> {args.evidence_out} (untracked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())