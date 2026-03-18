"""LLM client for VT ARC's OpenAI-compatible API."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are HokieHelp, a helpful assistant for Virginia Tech's Computer Science department.
Answer questions using ONLY the provided context from the CS department website.
If the context doesn't contain enough information to answer, say so honestly.
Always cite your sources by mentioning the relevant page title and URL.
Be concise and direct."""


def build_rag_prompt(question: str, chunks: List[dict]) -> str:
    """Build the user message with retrieved context."""
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No relevant information was found in the CS department website. "
            "Please let the user know you couldn't find an answer."
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk.get('title', 'Untitled')} — {chunk.get('url', '')}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Context from the VT CS department website:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )


class LLMClient:
    """Calls VT ARC's OpenAI-compatible LLM API."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("LLM client ready — model=%s  base_url=%s", model, base_url)

    def ask(self, question: str, chunks: List[dict]) -> str:
        """Send a RAG query to the LLM and return the answer."""
        user_message = build_rag_prompt(question, chunks)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        logger.info("LLM responded — question=%r  answer_len=%d", question, len(answer))
        return answer
