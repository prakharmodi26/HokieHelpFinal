"""Read-only retrieval against the existing BGE-M3 dense/sparse corpus.

Embedding and cross-encoder models stay on the existing embedding service. No
index writes, local model loads, or neighboring-chunk expansion occur here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import httpx
from qdrant_client import QdrantClient, models

from chatbot.program_metadata import program_filter_names, program_matches, program_scope, source_conflicts, source_identity_matches

logger = logging.getLogger(__name__)


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().removeprefix("www.")
        port = parts.port
        if port and port not in {80, 443}:
            host += f":{port}"
        path = parts.path.rstrip("/") or "/"
        for suffix in ("/index.html", "/index.htm", "/index.php"):
            if path.endswith(suffix):
                path = path[:-len(suffix)] or "/"
                break
        query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                                 if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid", "mc_cid", "mc_eid"}))
        return host + path + ("?" + query if query else "")
    except (ValueError, AttributeError):
        return str(url)


def chunk_key(chunk: dict) -> str:
    # Content-key dedup also collapses separately indexed www/http URL variants.
    body = (chunk.get("text") or "").strip()
    return canonical_url(chunk.get("url") or "") + "|" + hashlib.sha256(body.encode()).hexdigest()


def compose_passage(chunk: dict) -> str:
    """Match identity-composed reranker input used by the existing live stack."""
    def norm(text: str) -> str:
        decoded = unquote(text)
        if "\ufffd" in decoded and "\ufffd" not in text:
            decoded = text
        decoded = re.sub(r"^\s*#{1,6}\s+", "", decoded)
        return re.sub(r"\s+", " ", re.sub(r"[*_`]+", " ", decoded)).strip()
    title = norm(chunk.get("title") or "")[:300]
    headings = [norm(h) for h in chunk.get("headings_path", []) if isinstance(h, str)]
    headings = [h for h in headings if h and h.lower() != title.lower()]
    prefix = "\n".join(p for p in (title, " > ".join(headings)) if p)
    return (prefix + "\n\n" if prefix else "") + (chunk.get("text") or "")


class Retriever:
    def __init__(self, embedding_model: str, qdrant_host: str, qdrant_port: int,
                 collection: str, top_k: int = 12, min_score: float = .3,
                 hybrid_enabled: bool = True, candidate_k: int = 30,
                 embeddings_url: str = "http://embeddings-service", timeout: float = 20,
                 program_candidates_enabled: bool = True) -> None:
        if embedding_model != "BAAI/bge-m3":
            raise ValueError("The live dense/sparse index requires BAAI/bge-m3")
        if top_k <= 0 or candidate_k < top_k:
            raise ValueError("Require 0 < top_k <= candidate_k")
        self._client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=timeout)
        self._embedding_client = httpx.Client(base_url=embeddings_url.rstrip("/") + "/", timeout=timeout)
        self._collection = collection
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._min_score = min_score
        self._hybrid_enabled = hybrid_enabled
        self._program_candidates_enabled = program_candidates_enabled
        # Reject the old unnamed layout. Dimensions alone cannot verify embedding
        # model identity; that service/index contract is checked by the live probe.
        try:
            info = self._client.get_collection(collection)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors or {}
            if not isinstance(vectors, dict) or "dense" not in vectors or vectors["dense"].size != 1024:
                raise ValueError("Expected the existing named 1024-dimensional dense vector index")
            if hybrid_enabled and "sparse" not in sparse:
                raise ValueError("Expected the existing named sparse vector index")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._client.close()
        self._embedding_client.close()

    def _candidates(self, query: str) -> list[dict]:
        response = self._embedding_client.post("embed", json={"texts": [query]})
        response.raise_for_status()
        embeddings = response.json()
        dense = embeddings["dense"][0]
        if len(dense) != 1024 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in dense):
            raise ValueError("Invalid BGE-M3 dense embedding")
        sparse = embeddings.get("sparse", [{}])[0]
        hybrid = self._hybrid_enabled and bool(sparse.get("indices"))
        score_name = "fusion_score" if hybrid else "vector_score"

        def lookup(query_filter=None, lane="general"):
            if hybrid:
                hits = self._client.query_points(
                    collection_name=self._collection,
                    prefetch=[models.Prefetch(query=dense, using="dense", limit=self._candidate_k, filter=query_filter),
                              models.Prefetch(query=models.SparseVector(**sparse), using="sparse",
                                              limit=self._candidate_k, filter=query_filter)],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self._candidate_k, with_payload=True,
                ).points
            else:
                hits = self._client.query_points(collection_name=self._collection, query=dense,
                                                 using="dense", query_filter=query_filter,
                                                 limit=self._candidate_k, with_payload=True).points
            unique: dict[str, dict] = {}
            for hit in hits:
                payload = hit.payload or {}
                if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                    continue
                result = {**payload, "chunk_id": payload.get("chunk_id") or str(hit.id),
                          score_name: hit.score, "candidate_lane": lane}
                unique.setdefault(chunk_key(result), result)
            return list(unique.values())

        general = lookup()
        reserved = []
        scope = program_scope(query) if self._program_candidates_enabled else None
        if scope is not None:
            conditions = [models.FieldCondition(key="page_type", match=models.MatchValue(value="program"))]
            if scope["level"]:
                conditions.append(models.FieldCondition(key="program_level", match=models.MatchValue(value=scope["level"])))
            exact_names = program_filter_names(query)
            if exact_names:
                name_match = (models.MatchValue(value=exact_names[0]) if len(exact_names) == 1
                              else models.MatchAny(any=exact_names))
                conditions.append(models.FieldCondition(key="program_name", match=name_match))
            scoped = lookup(models.Filter(must=conditions), "program_exact" if exact_names else "program")
            # Reuse the SAME embedding; this lane is a bounded metadata view, not
            # another generated query or neighbor fetch. Only explicitly named
            # program/level/edition matches can displace general candidates.
            per_page: dict[str, list[dict]] = {}
            for chunk in scoped:
                if program_matches(query, chunk):
                    page = canonical_url(chunk.get("url") or "")
                    per_page.setdefault(page, []).append(chunk)
            reserve_limit = min(self._candidate_k // 2, self._top_k) if exact_names else min(6, max(1, self._candidate_k // 3))
            # Round-robin pages for comparisons, without prematurely limiting a
            # single program to two chunks before the cross-encoder sees them.
            reserved = [items[i] for i in range(reserve_limit)
                        for items in per_page.values() if i < len(items)][:reserve_limit]
        unique = {}
        for chunk in reserved + general:
            unique.setdefault(chunk_key(chunk), chunk)
        selected = list(unique.values())[:self._candidate_k]
        logger.info("RETRIEVAL general=%d program_reserved=%d unique=%d mode=%s",
                    len(general), len(reserved), len(selected), score_name)
        return selected

    def _rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        response = self._embedding_client.post("rerank", json={
            "query": query, "documents": [compose_passage(c) for c in candidates], "top_n": len(candidates),
        })
        response.raise_for_status()
        ranks = response.json()["results"]
        ranked = []
        seen: set[int] = set()
        for row in ranks:
            index, score = row["index"], float(row["score"])
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(candidates):
                raise ValueError("Invalid reranker index")
            if index in seen or not math.isfinite(score):
                raise ValueError("Invalid reranker result")
            seen.add(index)
            # Floor has one meaning in BOTH hybrid and dense-only modes:
            # existing cross-encoder relevance, never RRF or cosine.
            if score >= self._min_score:
                ranked.append({**candidates[index], "rerank_score": score})
        if len(seen) != len(candidates):
            raise ValueError("Incomplete reranker results")
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        logger.info("RETRIEVAL reranked=%d above_floor=%d floor=%.3f", len(ranks), len(ranked), self._min_score)
        return ranked

    def _select_ranked(self, query: str, ranked: list[dict]) -> list[dict]:
        anchored = any(source_identity_matches(query, chunk) for chunk in ranked)
        eligible = [chunk for chunk in ranked
                    if not source_conflicts(query, chunk, identity_anchored=anchored)]
        logger.info("RETRIEVAL source_filter input=%d kept=%d anchored=%s", len(ranked), len(eligible), anchored)
        if not self._program_candidates_enabled:
            return eligible[:self._top_k]
        # Scores alone strongly favor verbose/wrong-program PDFs in live probes.
        # Reserve a small number of slots for explicit catalog identity matches,
        # but only AFTER the same reranker floor. One slot per canonical page.
        reserved = []
        pages = set()
        for chunk in eligible:
            page = canonical_url(chunk.get("url") or "")
            if (page not in pages and chunk.get("rerank_score", -1) >= self._min_score
                    and program_matches(query, chunk)):
                reserved.append(chunk)
                pages.add(page)
            if len(reserved) >= min(3, max(1, self._top_k // 4)):
                break
        keys = {chunk_key(c) for c in reserved}
        return (reserved + [c for c in eligible if chunk_key(c) not in keys])[:self._top_k]

    def search(self, query: str) -> list[dict]:
        return self.search_many([query])

    def search_many(self, queries: list[str], *, original_query: str | None = None) -> list[dict]:
        if not 1 <= len(queries) <= 3 or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("Retrieval requires one to three nonempty planned queries")
        searches = list(dict.fromkeys(queries))
        if original_query and original_query.strip().casefold() not in {q.strip().casefold() for q in searches}:
            searches.append(original_query.strip())
        # Preserve original-query recall, then rerank the merged pool ONCE against
        # the resolved primary question. Raw pronoun queries must not set relevance.
        merged: dict[str, dict] = {}
        for query_index, query in enumerate(searches):
            for rank, chunk in enumerate(self._candidates(query), 1):
                key = chunk_key(chunk)
                if key not in merged:
                    merged[key] = {**chunk, "query_fusion_score": 0.0, "query_ranks": []}
                item = merged[key]
                item["query_fusion_score"] += 1 / (60 + rank)
                item["query_ranks"].append({"query_index": query_index, "rank": rank})
        pool = sorted(merged.values(), key=lambda c: c["query_fusion_score"], reverse=True)
        results = self._select_ranked(queries[0], self._rank(queries[0], pool))
        # One atomic diagnostic record per search, never returned as cited sources.
        # Omit questions, history, and passage bodies; retain all candidate IDs/URLs.
        logger.info("RETRIEVAL_TRACE %s", json.dumps({
            "search_count": len(searches), "candidate_k": self._candidate_k,
            "candidates": [{k: c[k] for k in ("chunk_id", "url", "candidate_lane", "vector_score", "fusion_score", "query_fusion_score", "query_ranks") if k in c} for c in pool],
            "selected": [{"chunk_id": c["chunk_id"], "rerank_score": c["rerank_score"],
                          "program_match": program_matches(queries[0], c)} for c in results],
        }))
        logger.info("RETRIEVAL queries=%d merged=%d selected=%d", len(searches), len(merged), len(results))
        for rank, chunk in enumerate(results, 1):
            logger.info("RETRIEVAL selected_rank=%d chunk_id=%s rerank=%.4f query_rrf=%.5f",
                        rank, chunk["chunk_id"], chunk["rerank_score"], chunk["query_fusion_score"])
        return results
