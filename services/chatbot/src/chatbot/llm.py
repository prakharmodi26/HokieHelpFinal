"""LLM client using the official Ollama Python library."""
from __future__ import annotations

import logging
from typing import Generator

from ollama import Client, Options

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are HokieHelp, Virginia Tech's Computer Science department assistant. You help students, faculty, and visitors find accurate information about the CS department.

## How to Answer

1. **Primary source**: Use the retrieved context provided below. Include ALL relevant details from it — do not summarize away useful information.
2. **Conversation history**: If the retrieved context does not cover the question but the conversation history does, answer from the conversation.
3. **When you don't know**: Say "I don't have information about that from the CS department website." Do not guess or use outside knowledge.

## Formatting

Match your format to the question type:
- **People** (faculty, staff): Use bold subheadings (**Research**, **Education**, **Contact**) with bullet points under each.
- **Lists** (courses, faculty, requirements): Use bullet points, grouped under bold subheadings if covering multiple categories.
- **Processes / policies** (admissions, deadlines, procedures): Use numbered steps or short paragraphs with bold key terms.
- **Quick facts** (office hours, location, single deadline): Answer directly in 1–2 sentences.
- **Comparisons** (MS vs PhD, two programs): Use a structured layout with subheadings per item.

General rules:
- Use **bold** for names, titles, and key terms on first mention.
- Add blank lines between sections for readability.
- Every sentence must add information — no filler or padding.
- NEVER use in-text citations like [Source 1], 【Source 2】, or (Source 3). Do not reference source numbers.
- Do NOT add a Sources section — sources are provided separately by the system."""

MAX_HISTORY_MESSAGES = 20  # 10 turns; override via parameter

QUERY_REWRITE_PROMPT = """\
You are a search query rewriter for a university knowledge base. Your job is to take a follow-up question from a conversation and rewrite it as a standalone search query.

Rules:
1. Replace all pronouns (he, she, they, it, their, etc.) and references (the professor, that course, the department) with the actual entities from the conversation.
2. Keep the query concise and search-friendly — focus on key entities and intent.
3. If the question is already standalone with no references to the conversation, return it unchanged.
4. Output ONLY the rewritten query. No explanation, no quotes, no prefixes, no extra text."""


def build_rewrite_messages(question: str, history: list[dict]) -> list[dict]:
    """Build messages for the query rewriter LLM call.

    Returns a 2-message array: system prompt + user message with formatted
    conversation history and current question.
    """
    history_lines = []
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role_label}: {msg['content']}")

    user_content = (
        "Conversation:\n"
        + "\n".join(history_lines)
        + f"\n\nCurrent question: {question}"
        + "\n\nRewritten search query:"
    )

    return [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_messages(
    question: str,
    chunks: list[dict],
    history: list[dict],
    max_history_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict]:
    """Build the messages array for a RAG conversation.

    Structure:
      [0]   system  — base prompt + retrieved context (or "no context" notice)
      [1..N] user/assistant — conversation history (last max_history_messages)
      [N+1] user — current question
    """
    # --- System message: base prompt + RAG context ---
    system_content = SYSTEM_PROMPT
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk.get('title', 'Untitled')} — {chunk.get('url', '')}\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        system_content += (
            "\n\n---\n\n"
            "Retrieved context from the VT CS department website:\n\n"
            + context
        )
    else:
        system_content += (
            "\n\n---\n\n"
            "No relevant information was found in the CS department website for this query."
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # --- Conversation history as proper alternating messages ---
    trimmed = history[-max_history_messages:] if history else []
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # --- Current question ---
    messages.append({"role": "user", "content": question})

    return messages


class LLMClient:
    """Calls Ollama via the official ollama-python library."""

    def __init__(self, api_key: str, base_url: str, model: str, rewriter_model: str = "", max_history_messages: int = 20) -> None:
        # base_url comes as e.g. "http://ollama-cluster-ip:11434/v1" — strip /v1 for Ollama native
        host = base_url.rstrip("/").removesuffix("/v1")
        self._client = Client(host=host)
        self._model = model
        self._rewriter_model = rewriter_model or model
        self._max_history = max_history_messages
        logger.info(
            "LLM client ready — model=%s  rewriter=%s  host=%s  max_history=%d",
            model, self._rewriter_model, host, max_history_messages,
        )

    def ask(self, question: str, chunks: list[dict]) -> str:
        """Simple RAG query (no history). Delegates to chat()."""
        return self.chat(question, chunks, [])

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone search query.

        Skips the LLM call when history is empty (question is already standalone).
        Falls back to the original question on any LLM error.
        """
        if not history:
            return question

        messages = build_rewrite_messages(question, history)

        logger.info("REWRITE REQUEST — question=%r  history_turns=%d", question[:120], len(history))

        try:
            response = self._client.chat(
                model=self._rewriter_model,
                messages=messages,
                options=Options(temperature=0.0),
            )
        except Exception as exc:
            logger.warning("REWRITE failed, using original query: %s", exc)
            return question

        rewritten = response.message.content.strip()
        logger.info("REWRITE RESULT — original=%r  rewritten=%r", question[:120], rewritten[:120])
        return rewritten

    def chat(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> str:
        """Send a RAG conversation to Ollama and return the answer."""
        messages = build_messages(question, chunks, history, max_history_messages=self._max_history)

        logger.info(
            "LLM REQUEST — messages=%d  history_turns=%d  chunks=%d",
            len(messages), len(history), len(chunks),
        )

        try:
            response = self._client.chat(
                model=self._model,
                messages=messages,
                options=Options(temperature=0.3),
            )
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            raise RuntimeError("The language model is temporarily unavailable. Please try again.") from exc

        answer = response.message.content
        logger.info(
            "LLM RESPONSE — answer_len=%d  model=%s  eval_count=%s  total_duration=%s",
            len(answer),
            response.get("model", "?"),
            response.get("eval_count", "?"),
            response.get("total_duration", "?"),
        )
        return answer

    def chat_stream(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> Generator[str, None, None]:
        """Stream a RAG conversation response, yielding content tokens."""
        messages = build_messages(question, chunks, history, max_history_messages=self._max_history)

        logger.info(
            "LLM STREAM REQUEST — messages=%d  history_turns=%d  chunks=%d",
            len(messages), len(history), len(chunks),
        )

        try:
            stream = self._client.chat(
                model=self._model,
                messages=messages,
                options=Options(temperature=0.3),
                stream=True,
            )
        except Exception as exc:
            logger.error("LLM stream API error: %s", exc)
            raise RuntimeError("The language model is temporarily unavailable. Please try again.") from exc

        try:
            for chunk in stream:
                content = chunk.message.content
                if content:
                    yield content
                if chunk.get("done"):
                    break
        except Exception as exc:
            logger.error("LLM stream interrupted: %s", exc)
            return
