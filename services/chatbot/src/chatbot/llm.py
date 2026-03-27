"""LLM client using the official Ollama Python library."""
from __future__ import annotations

import logging
from typing import Generator

from ollama import Client, Options

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are HokieHelp, a helpful assistant for Virginia Tech's Computer Science department.

Rules:
1. If retrieved context is provided above, use it as your primary source. Include ALL relevant details — do not summarize or omit information.
2. If no retrieved context is provided (or it is empty), check the conversation history. If the user is asking about something discussed earlier, answer from the conversation.
3. Only say "I don't have enough information to answer that based on the CS department website." if NEITHER the retrieved context NOR the conversation history contain the answer.
4. Do not use outside knowledge. Only use retrieved context and conversation history.
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
- Do NOT add a Sources section if you answered purely from conversation history.

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

MAX_HISTORY_MESSAGES = 20  # 10 turns; override via parameter


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

    def __init__(self, api_key: str, base_url: str, model: str, max_history_messages: int = 20) -> None:
        # base_url comes as e.g. "http://ollama-cluster-ip:11434/v1" — strip /v1 for Ollama native
        host = base_url.rstrip("/").removesuffix("/v1")
        self._client = Client(host=host)
        self._model = model
        self._max_history = max_history_messages
        logger.info("LLM client ready — model=%s  host=%s  max_history=%d", model, host, max_history_messages)

    def ask(self, question: str, chunks: list[dict]) -> str:
        """Simple RAG query (no history). Delegates to chat()."""
        return self.chat(question, chunks, [])

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
