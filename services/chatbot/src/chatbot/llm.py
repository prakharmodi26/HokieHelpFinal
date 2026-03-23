"""LLM client for VT ARC's OpenAI-compatible API."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are HokieHelp, a helpful assistant for Virginia Tech's Computer Science department.

Rules:
1. If RETRIEVED CONTEXT is provided, use it as your primary source. Include ALL relevant details — do not summarize or omit information.
2. If no retrieved context is provided (or it is empty), check the CHAT HISTORY. If the user is asking about something discussed earlier in the conversation, answer from the chat history.
3. Only say "I don't have enough information to answer that based on the CS department website." if NEITHER the retrieved context NOR the chat history contain the answer.
4. Do not use outside knowledge. Only use retrieved context and chat history.
5. NEVER use in-text citations like [Source 1], 【Source 2】, or (Source 3). Do not reference source numbers in the answer body.

Formatting:
- Use bullet points for all answers. Each distinct fact or detail gets its own bullet.
- Group related bullets under bold subheadings when the answer covers multiple topics (e.g., **Education**, **Research**, **Teaching**).
- Add a blank line between groups for readability.
- Only use paragraphs if the user explicitly asks for prose.

Sources section:
- At the very end, add a "Sources:" heading ONLY if you used retrieved context.
- List ONLY the pages you actually used from the retrieved context.
- Format: **[Page Title](URL)**
- Do NOT add a Sources section if you answered purely from chat history.

Example:
- Full name and title
- Department and university

**Research Interests:**
- Interest 1
- Interest 2

**Education:**
- Degree, University (Year)
- Degree, University (Year)

**Sources:**
- **[Page Title | Computer Science](https://website.cs.vt.edu/...)**"""


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


def build_chat_prompt(
    question: str,
    chunks: List[dict],
    history: List[dict],
) -> str:
    """Build the user message with chat history and optional retrieved context."""
    parts = []

    # Chat history section
    if history:
        hist_lines = []
        for msg in history[-10:]:  # last 10 messages (5 turns)
            role = msg["role"].capitalize()
            hist_lines.append(f"{role}: {msg['content']}")
        parts.append(
            "CHAT HISTORY:\n" + "\n".join(hist_lines)
        )

    # Retrieved context section
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk.get('title', 'Untitled')} — {chunk.get('url', '')}\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        parts.append(
            "RETRIEVED CONTEXT:\n" + context
        )

    parts.append(f"Question: {question}")

    return "\n\n---\n\n".join(parts)


class LLMClient:
    """Calls VT ARC's OpenAI-compatible LLM API."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("LLM client ready — model=%s  base_url=%s", model, base_url)

    def ask(self, question: str, chunks: List[dict]) -> str:
        """Send a RAG query to the LLM and return the answer."""
        user_message = build_rag_prompt(question, chunks)

        logger.info("LLM REQUEST — system_prompt_len=%d  user_message_len=%d", len(SYSTEM_PROMPT), len(user_message))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        usage = response.usage
        logger.info(
            "LLM RESPONSE — answer_len=%d  prompt_tokens=%s  completion_tokens=%s  model=%s",
            len(answer),
            getattr(usage, 'prompt_tokens', '?') if usage else '?',
            getattr(usage, 'completion_tokens', '?') if usage else '?',
            response.model if hasattr(response, 'model') else '?',
        )
        return answer

    def chat(self, question: str, chunks: List[dict], history: List[dict]) -> str:
        """Send a RAG query with conversation history to the LLM."""
        user_message = build_chat_prompt(question, chunks, history)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info("LLM CHAT REQUEST — history_turns=%d  chunks=%d  prompt_len=%d",
                     len(history), len(chunks), len(user_message))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        usage = response.usage
        logger.info(
            "LLM CHAT RESPONSE — answer_len=%d  prompt_tokens=%s  completion_tokens=%s",
            len(answer),
            getattr(usage, 'prompt_tokens', '?') if usage else '?',
            getattr(usage, 'completion_tokens', '?') if usage else '?',
        )
        return answer
