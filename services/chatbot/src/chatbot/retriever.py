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
        logger.info(
            "Retriever ready — model=%s  device=%s  collection=%s  top_k=%d",
            embedding_model, device, collection, top_k,
        )

    def search(self, query: str) -> List[dict]:
        """Embed query and return top-k chunks from Qdrant."""
        text = BGE_QUERY_PREFIX + query
        vector = self._model.encode(text, show_progress_bar=False).tolist()

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=self._top_k,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            results.append({
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id"),
                "text": hit.payload.get("text", ""),
                "url": hit.payload.get("url", ""),
                "title": hit.payload.get("title", ""),
                "headings_path": hit.payload.get("headings_path", []),
                "page_type": hit.payload.get("page_type", ""),
            })
        return results
