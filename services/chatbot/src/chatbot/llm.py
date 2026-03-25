"""LLM client for VT ARC's OpenAI-compatible API."""
from __future__ import annotations

import logging
from typing import Generator

from openai import OpenAI

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
    """Build the OpenAI messages array for a RAG conversation.

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
    """Calls VT ARC's OpenAI-compatible LLM API."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("LLM client ready — model=%s  base_url=%s", model, base_url)

    def ask(self, question: str, chunks: list[dict]) -> str:
        """Simple RAG query (no history). Delegates to chat()."""
        return self.chat(question, chunks, [])

    def chat(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> str:
        """Send a RAG conversation to the LLM and return the answer."""
        messages = build_messages(question, chunks, history)

        logger.info(
            "LLM REQUEST — messages=%d  history_turns=%d  chunks=%d",
            len(messages), len(history), len(chunks),
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        usage = response.usage
        logger.info(
            "LLM RESPONSE — answer_len=%d  prompt_tokens=%s  completion_tokens=%s  model=%s",
            len(answer),
            getattr(usage, "prompt_tokens", "?") if usage else "?",
            getattr(usage, "completion_tokens", "?") if usage else "?",
            response.model if hasattr(response, "model") else "?",
        )
        return answer

    def chat_stream(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> Generator[str, None, None]:
        """Stream a RAG conversation response, yielding content tokens."""
        messages = build_messages(question, chunks, history)

        logger.info(
            "LLM STREAM REQUEST — messages=%d  history_turns=%d  chunks=%d",
            len(messages), len(history), len(chunks),
        )

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
