"""LLM client for VT ARC's OpenAI-compatible API."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are HokieHelp, a helpful assistant for Virginia Tech's Computer Science department.

Rules:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say "I don't have enough information to answer that based on the CS department website."
3. Include ALL relevant details from the context — do not summarize or omit information.
4. NEVER use in-text citations like [Source 1], 【Source 2】, or (Source 3). Do not reference source numbers in the answer body.

Formatting:
- Use bullet points for all answers. Each distinct fact or detail gets its own bullet.
- Group related bullets under bold subheadings when the answer covers multiple topics (e.g., **Education**, **Research**, **Teaching**).
- Add a blank line between groups for readability.
- Only use paragraphs if the user explicitly asks for prose.

Sources section:
- At the very end, add a "Sources:" heading.
- List ONLY the pages you actually used.
- Format: **[Page Title](URL)**

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


def build_chat_messages(
    question: str,
    chunks: List[dict],
    history: List[dict],
) -> List[dict]:
    """Build the full message list for the LLM with conversation history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    user_message = build_rag_prompt(question, chunks)
    messages.append({"role": "user", "content": user_message})
    return messages


REWRITE_PROMPT = """\
Given the conversation history and a follow-up question, rewrite the follow-up \
into a standalone search query that captures the full intent. \
Return ONLY the rewritten query, nothing else. \
If the question is already standalone, return it unchanged.

Conversation:
{history}

Follow-up question: {question}
Standalone query:"""


class LLMClient:
    """Calls VT ARC's OpenAI-compatible LLM API."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("LLM client ready — model=%s  base_url=%s", model, base_url)

    def rewrite_query(self, question: str, history: List[dict]) -> str:
        """Rewrite a follow-up question into a standalone search query."""
        if not history:
            return question

        hist_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:200]}" for m in history[-6:]
        )
        prompt = REWRITE_PROMPT.format(history=hist_text, question=question)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info("QUERY REWRITE: '%s' → '%s'", question, rewritten)
        return rewritten

    def ask(self, question: str, chunks: List[dict]) -> str:
        """Send a RAG query to the LLM and return the answer."""
        user_message = build_rag_prompt(question, chunks)

        logger.info("LLM REQUEST — system_prompt_len=%d  user_message_len=%d", len(SYSTEM_PROMPT), len(user_message))
        logger.info("LLM FULL PROMPT TO LLM:\n%s", user_message)

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
        logger.info("LLM ANSWER:\n%s", answer)
        return answer

    def chat(self, question: str, chunks: List[dict], history: List[dict]) -> str:
        """Send a RAG query with conversation history to the LLM."""
        messages = build_chat_messages(question, chunks, history)
        logger.info("LLM CHAT REQUEST — messages=%d  history_turns=%d", len(messages), len(history))
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
