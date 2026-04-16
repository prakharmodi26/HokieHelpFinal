"""Embed user queries and search Qdrant for relevant chunks."""
from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _payload_to_result(payload: dict, score: float) -> dict:
    """Convert a Qdrant payload to a result dict."""
    return {
        "score": score,
        "chunk_id": payload.get("chunk_id"),
        "text": payload.get("text", ""),
        "url": payload.get("url", ""),
        "title": payload.get("title", ""),
        "headings_path": payload.get("headings_path", []),
        "page_type": payload.get("page_type", ""),
    }


def rrf_fuse(
    vector_results: List[dict],
    keyword_results: List[dict],
    k: int = 60,
    top_k: int = 5,
    min_rrf_score: float = 0.0,
) -> List[dict]:
    """Reciprocal Rank Fusion: merge two ranked lists by chunk_id.

    RRF score = sum(1 / (k + rank)) across lists where the chunk appears.
    Chunks with RRF score below min_rrf_score are dropped.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, 1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        items[cid] = r

    for rank, r in enumerate(keyword_results, 1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in items:
            items[cid] = r

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused = []
    dropped = 0
    for cid in ranked[:top_k]:
        rrf_score = scores[cid]
        if rrf_score < min_rrf_score:
            dropped += 1
            logger.info(
                "RETRIEVER rrf_drop chunk_id=%s  rrf_score=%.5f  min_rrf_score=%.5f  title=%s",
                cid, rrf_score, min_rrf_score, (items[cid].get("title") or "")[:60],
            )
            continue
        result = items[cid]
        result["score"] = rrf_score
        fused.append(result)

    if dropped:
        logger.info("RETRIEVER rrf_fuse dropped %d chunks below min_rrf_score=%.5f", dropped, min_rrf_score)

    return fused


class Retriever:
    """Embeds queries with BGE prefix and searches Qdrant."""

    def __init__(
        self,
        embedding_model: str,
        qdrant_host: str,
        qdrant_port: int,
        collection: str,
        top_k: int,
        min_score: float = 0.53,
        hybrid_enabled: bool = True,
        keyword_search_limit: int = 10,
        rrf_k: int = 60,
        min_rrf_score: float = 0.0,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = "float16"
        self._model = SentenceTransformer(
            embedding_model,
            device=device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        self._client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._collection = collection
        self._top_k = top_k
        self._min_score = min_score
        self._hybrid_enabled = hybrid_enabled
        self._keyword_limit = keyword_search_limit
        self._rrf_k = rrf_k
        self._min_rrf_score = min_rrf_score
        logger.info(
            "Retriever ready — model=%s  device=%s  collection=%s  top_k=%d  min_score=%.2f  hybrid=%s  min_rrf_score=%.5f",
            embedding_model, device, collection, top_k, min_score, hybrid_enabled, min_rrf_score,
        )
        if hybrid_enabled:
            self._ensure_text_index()

    def _ensure_text_index(self) -> None:
        """Create a full-text index on the 'text' payload field if not present."""
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            logger.info("Created full-text index on 'text' field")
        except Exception as exc:
            # Index may already exist — that's fine
            if "already exists" in str(exc).lower():
                logger.info("Full-text index on 'text' field already exists")
            else:
                logger.warning("Could not create text index: %s", exc)

    def _vector_search(self, query: str) -> List[dict]:
        """Embed query and return ranked chunks via vector similarity."""
        text = BGE_QUERY_PREFIX + query
        logger.info("RETRIEVER vector query=%r  prefixed=%r", query, text[:120])
        vector = self._model.encode(text, show_progress_bar=False).tolist()

        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=self._top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant vector search failed: %s", exc)
            return []

        results = []
        for rank, hit in enumerate(response.points, 1):
            result = _payload_to_result(hit.payload, hit.score)
            results.append(result)
            text_preview = (result["text"] or "[EMPTY]")[:200].replace("\n", " ")
            logger.debug(
                "RETRIEVER vector rank=%d  score=%.4f  chunk_id=%s  title=%s  preview=%s",
                rank, hit.score, result["chunk_id"], result["title"][:60] if result["title"] else "",
                text_preview,
            )
        return results

    # Common English stop words that match almost every document — exclude from keyword search
    _STOP_WORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "dare",
        "what", "who", "whom", "which", "where", "when", "why", "how",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "he", "she", "we", "you", "i", "me", "my", "your", "our", "his", "her",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "there", "here", "about", "into", "through", "during", "each",
        "more", "most", "other", "some", "any", "such", "than", "too",
        "very", "just", "also", "as", "if", "while", "after", "before",
        "between", "out", "up", "down", "then", "than", "only", "same",
        "now", "like", "over", "all",
    })

    def _keyword_search(self, query: str) -> List[dict]:
        """Search Qdrant using full-text match on the 'text' payload field.

        Splits query into individual tokens and uses OR (should) matching
        so that any single matching token surfaces relevant results.
        This is critical for typo tolerance: "Kirk Camron" → "Kirk" matches
        even though "Camron" doesn't.

        Stop words are excluded — they match nearly every document and corrupt
        RRF ranking by making generic hub pages rank first.
        """
        # Split into meaningful tokens (2+ chars, matching the index min_token_len)
        # Strip trailing punctuation before stop-word check
        raw_tokens = [t.rstrip("?.,;:!") for t in query.split() if len(t) >= 2]
        tokens = [t for t in raw_tokens if t.lower() not in self._STOP_WORDS and len(t) >= 2]
        logger.info("RETRIEVER keyword tokens after stop-word filter: %s", tokens)
        if not tokens:
            return []

        results = []
        try:
            # Use 'should' (OR) so any matching token brings results
            conditions = [
                models.FieldCondition(
                    key="text",
                    match=models.MatchText(text=token),
                )
                for token in tokens
            ]
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(should=conditions),
                limit=self._keyword_limit,
                with_payload=True,
            )
            # Score each result by how many query tokens appear in its text.
            # Scroll returns results in insertion order (not relevance order),
            # so we must score manually to give RRF meaningful rank signals.
            lower_tokens = [t.lower() for t in tokens]
            scored = []
            for point in points:
                result = _payload_to_result(point.payload, 0.0)
                text_lower = (result.get("text") or "").lower()
                result["_kw_matches"] = sum(1 for t in lower_tokens if t in text_lower)
                scored.append(result)
            # Sort by match count descending — more specific chunks rank higher
            scored.sort(key=lambda r: r["_kw_matches"], reverse=True)
            for rank, result in enumerate(scored, 1):
                logger.info(
                    "RETRIEVER keyword rank=%d  matches=%d  chunk_id=%s  title=%s",
                    rank, result["_kw_matches"], result["chunk_id"],
                    (result["title"] or "")[:60],
                )
            results = scored
        except Exception as exc:
            logger.warning("Keyword search failed (falling back to vector only): %s", exc)
        logger.info("RETRIEVER keyword results=%d  query=%r  tokens=%s", len(results), query, tokens)
        return results

    def _fetch_adjacent_chunks(self, chunks: List[dict]) -> List[dict]:
        """For each retrieved chunk, fetch the immediately adjacent chunks (±1) from the same page.

        Chunk IDs have the format `{page_hash}_{seq:04d}`. Adjacent chunks from the same
        page often contain complementary info — e.g. contact details in chunk _0000 when
        the bio was retrieved in chunk _0001.

        Adjacent chunks are tagged from_vector=True (same page as a vector result) so they
        appear in LLM context. They are deduped by URL in sources so they don't add noise.
        """
        seen_ids = {c["chunk_id"] for c in chunks}
        to_fetch: List[str] = []

        for chunk in chunks:
            if not chunk.get("from_vector", True):
                continue  # only expand vector results
            cid = chunk["chunk_id"]
            parts = cid.rsplit("_", 1)
            if len(parts) != 2:
                continue
            page_hash, seq_str = parts
            try:
                seq = int(seq_str)
            except ValueError:
                continue
            for adj_seq in range(max(0, seq - 2), seq + 3):  # ±2 window
                if adj_seq == seq:
                    continue
                adj_id = f"{page_hash}_{adj_seq:04d}"
                if adj_id not in seen_ids:
                    to_fetch.append(adj_id)
                    seen_ids.add(adj_id)

        if not to_fetch:
            return chunks

        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=to_fetch),
                    )
                ]),
                limit=len(to_fetch) + 5,
                with_payload=True,
            )
            adjacent = []
            for point in points:
                result = _payload_to_result(point.payload, 0.0)
                result["from_vector"] = True  # same page as a vector result
                result["score"] = 0.0  # adjacent, not ranked
                adjacent.append(result)
                logger.info(
                    "RETRIEVER adjacent chunk_id=%s  title=%s  url=%s",
                    result["chunk_id"], (result.get("title") or "")[:60],
                    (result.get("url") or "")[:80],
                )
            logger.info("RETRIEVER expanded with %d adjacent chunks", len(adjacent))
            return chunks + adjacent
        except Exception as exc:
            logger.warning("Adjacent chunk fetch failed: %s", exc)
            return chunks

    def search(self, query: str) -> List[dict]:
        """Search Qdrant using vector similarity, optionally fused with keyword search.

        Each returned chunk has a `from_vector` bool indicating it appeared in the
        semantic vector results. Callers can use this to filter sources shown to the
        user — keyword-only chunks boost LLM context but should not appear as sources.
        """
        vector_results = self._vector_search(query)
        vector_chunk_ids = {r["chunk_id"] for r in vector_results}

        if self._hybrid_enabled:
            keyword_results = self._keyword_search(query)
            if keyword_results:
                fused = rrf_fuse(
                    vector_results, keyword_results,
                    k=self._rrf_k, top_k=self._top_k,
                    min_rrf_score=self._min_rrf_score,
                )
                for r in fused:
                    r["from_vector"] = r["chunk_id"] in vector_chunk_ids
                logger.info(
                    "RETRIEVER hybrid fused: vector=%d + keyword=%d -> %d results (min_rrf_score=%.5f)",
                    len(vector_results), len(keyword_results), len(fused), self._min_rrf_score,
                )
                for rank, r in enumerate(fused, 1):
                    logger.info(
                        "RETRIEVER fused rank=%d  rrf_score=%.5f  from_vector=%s  chunk_id=%s  title=%s  url=%s",
                        rank, r["score"], r["from_vector"], r["chunk_id"],
                        (r.get("title") or "")[:60], (r.get("url") or "")[:80],
                    )
                return self._fetch_adjacent_chunks(fused)

        # Pure vector path (or keyword returned nothing)
        results = [r for r in vector_results if r["score"] >= self._min_score]
        for r in results:
            r["from_vector"] = True
        dropped = len(vector_results) - len(results)
        if dropped:
            logger.info("RETRIEVER dropped %d chunks below min_score=%.2f", dropped, self._min_score)
        logger.info("RETRIEVER total_results=%d (kept=%d)  query=%r", len(vector_results), len(results), query)
        return self._fetch_adjacent_chunks(results)
