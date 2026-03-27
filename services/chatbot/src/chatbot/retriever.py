"""Embed user queries and search Qdrant for relevant chunks."""
from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def contextualize_query(query: str, history: list[dict]) -> str:
    """Enrich a follow-up question with conversation context for better retrieval.

    Prepends the last user message so pronouns like "their", "he", "she"
    resolve to the correct entity during embedding.
    """
    if not history:
        return query

    # Find the last user message
    for msg in reversed(history):
        if msg.get("role") == "user":
            return f"{msg['content']} {query}"

    return query


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
) -> List[dict]:
    """Reciprocal Rank Fusion: merge two ranked lists by chunk_id.

    RRF score = sum(1 / (k + rank)) across lists where the chunk appears.
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
    for cid in ranked[:top_k]:
        result = items[cid]
        result["score"] = scores[cid]
        fused.append(result)

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
        logger.info(
            "Retriever ready — model=%s  device=%s  collection=%s  top_k=%d  min_score=%.2f  hybrid=%s",
            embedding_model, device, collection, top_k, min_score, hybrid_enabled,
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

    def _keyword_search(self, query: str) -> List[dict]:
        """Search Qdrant using full-text match on the 'text' payload field.

        Splits query into individual tokens and uses OR (should) matching
        so that any single matching token surfaces relevant results.
        This is critical for typo tolerance: "Kirk Camron" → "Kirk" matches
        even though "Camron" doesn't.
        """
        # Split into meaningful tokens (2+ chars, matching the index min_token_len)
        tokens = [t for t in query.split() if len(t) >= 2]
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
            for rank, point in enumerate(points, 1):
                result = _payload_to_result(point.payload, 0.0)
                results.append(result)
                logger.info(
                    "RETRIEVER keyword rank=%d  chunk_id=%s  title=%s",
                    rank, result["chunk_id"], result["title"][:60] if result["title"] else "",
                )
        except Exception as exc:
            logger.warning("Keyword search failed (falling back to vector only): %s", exc)
        logger.info("RETRIEVER keyword results=%d  query=%r  tokens=%s", len(results), query, tokens)
        return results

    def search(self, query: str) -> List[dict]:
        """Search Qdrant using vector similarity, optionally fused with keyword search."""
        vector_results = self._vector_search(query)

        if self._hybrid_enabled:
            keyword_results = self._keyword_search(query)
            if keyword_results:
                fused = rrf_fuse(vector_results, keyword_results, k=self._rrf_k, top_k=self._top_k)
                logger.info(
                    "RETRIEVER hybrid fused: vector=%d + keyword=%d -> %d results",
                    len(vector_results), len(keyword_results), len(fused),
                )
                return fused

        # Pure vector path (or keyword returned nothing)
        results = [r for r in vector_results if r["score"] >= self._min_score]
        dropped = len(vector_results) - len(results)
        if dropped:
            logger.info("RETRIEVER dropped %d chunks below min_score=%.2f", dropped, self._min_score)
        logger.info("RETRIEVER total_results=%d (kept=%d)  query=%r", len(vector_results), len(results), query)
        return results

    def search_with_context(self, query: str, history: list[dict]) -> list[dict]:
        """Search with conversation-aware query enrichment."""
        enriched = contextualize_query(query, history)
        logger.info("RETRIEVER contextualized query=%r -> %r", query, enriched[:120])
        return self.search(enriched)
