"""Embedding model wrapper using SentenceTransformers."""
from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def build_context(chunk: dict) -> str:
    """Build enriched embedding input from chunk metadata + text.

    Format:
        Title: <title>
        Section: <last heading>
        Path: <headings joined>

        <chunk text>
    """
    parts = [f"Title: {chunk['title']}"]

    headings = chunk.get("headings_path") or []
    if headings:
        parts.append(f"Section: {headings[-1]}")
        parts.append(f"Path: {' > '.join(headings)}")

    parts.append("")
    parts.append(chunk["text"])
    return "\n".join(parts)


class Embedder:
    """Loads a SentenceTransformers model and generates embeddings.

    Uses CUDA with fp16 if a GPU is available, falls back to CPU fp32.
    """

    def __init__(self, model_name: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model: %s  (device=%s)", model_name, self._device)

        model_kwargs = {}
        if self._device == "cuda":
            # fp16 on GPU: ~2x faster, half the memory, negligible quality loss
            model_kwargs["torch_dtype"] = "float16"

        self._model = SentenceTransformer(
            model_name,
            device=self._device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        logger.info("Model loaded — dimension=%d  device=%s", self.dimension, self._device)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of float lists."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]
