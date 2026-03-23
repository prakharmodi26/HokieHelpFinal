"""Embed user queries and search Qdrant for relevant chunks."""
from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
        logger.info(
            "Retriever ready — model=%s  device=%s  collection=%s  top_k=%d  min_score=%.2f",
            embedding_model, device, collection, top_k, min_score,
        )

    def search(self, query: str) -> List[dict]:
        """Embed query and return top-k chunks from Qdrant."""
        text = BGE_QUERY_PREFIX + query
        logger.info("RETRIEVER query=%r  prefixed=%r", query, text[:120])
        vector = self._model.encode(text, show_progress_bar=False).tolist()
        logger.info("RETRIEVER embedding done — dim=%d", len(vector))

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=self._top_k,
            with_payload=True,
        )

        all_results = []
        for rank, hit in enumerate(response.points, 1):
            payload = hit.payload
            chunk_text = payload.get("text", "")
            result = {
                "score": hit.score,
                "chunk_id": payload.get("chunk_id"),
                "text": chunk_text,
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "headings_path": payload.get("headings_path", []),
                "page_type": payload.get("page_type", ""),
            }
            all_results.append(result)

            text_preview = (chunk_text or "[EMPTY]")[:200].replace("\n", " ")
            logger.info(
                "RETRIEVER rank=%d  score=%.4f  chunk_id=%s  title=%s  text_len=%d  preview=%s",
                rank, hit.score, payload.get("chunk_id"), payload.get("title", "")[:60],
                len(chunk_text), text_preview,
            )

        # Filter by minimum score threshold
        results = [r for r in all_results if r["score"] >= self._min_score]
        dropped = len(all_results) - len(results)
        if dropped:
            logger.info(
                "RETRIEVER dropped %d chunks below min_score=%.2f",
                dropped, self._min_score,
            )

        logger.info("RETRIEVER total_results=%d (kept=%d)  query=%r", len(all_results), len(results), query)
        return results
