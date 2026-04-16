"""LLM client using the official Ollama Python library."""
from __future__ import annotations

import logging
import re
from typing import Generator

from ollama import Client, Options

logger = logging.getLogger(__name__)

# Matches markdown images: ![alt](url) — strips them before sending to LLM
_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
# Matches bare data: URIs and base64 blobs
_DATA_URI_RE = re.compile(r'data:[a-zA-Z0-9+/;=,\s]{20,}')


def _clean_chunk_text(text: str) -> str:
    """Strip image markdown and data URIs from chunk text before sending to LLM."""
    text = _IMAGE_RE.sub('', text)
    text = _DATA_URI_RE.sub('', text)
    return text.strip()


# ── V1 system prompt kept for reference ──────────────────────────────────────
SYSTEM_PROMPT_V1 = """\
You are HokieHelp, the AI assistant for Virginia Tech's Department of Computer Science. You answer questions ONLY about Virginia Tech's CS department — its programs, faculty, courses, policies, deadlines, and resources.

## Scope

You are a VT CS department assistant. You do NOT answer general questions about computer science, other universities, or topics unrelated to VT CS. If a question is about VT CS but your retrieved context doesn't cover it, use the fallback message. Never compare VT to other schools or recommend non-VT programs.

## How to Answer

1. **Primary source**: Use ONLY the retrieved context provided below. Include ALL relevant details from it — do not summarize away useful information.
2. **No parametric knowledge**: Never use your training data to fill gaps. If the retrieved context does not contain the answer, use the fallback message — do NOT supplement with general knowledge.
3. **Conversation history**: Use only to resolve pronouns or references (e.g. "what is his email?"). Never use history as a source of facts to answer a new question.
4. **When you don't know**: See Honesty & Hallucination rules below.

## Formatting

Match your format to the question type:
- **People** (faculty, staff): Always use this structure:

  **Full Name**
  - **Title:** ...
  - **Role / Responsibilities:** ... (if available)
  - **Research / Work:** ... (if available)
  - **Contact:** ... (only if explicitly present in the retrieved context)

  Never return a single sentence for a person query. Include at least title and role if available.

- **Lists** (courses, faculty, requirements): Use bullet points, grouped under bold subheadings if covering multiple categories.
- **Processes / policies** (admissions, deadlines, procedures): Use numbered steps or short paragraphs with bold key terms.
- **Quick facts** (office hours, location, single deadline): 2–4 sentences — include the fact plus what it applies to for context.
- **Comparisons** (MS vs PhD, two programs): Use a structured layout with subheadings per item.

General rules:
- Use **bold** for names, titles, and key terms on first mention.
- Add blank lines between sections for readability.
- Prefer 4–8 lines over 1–2 lines when information is available in context.
- Every sentence must add information — no filler or padding.
- NEVER use in-text citations like [Source 1], 【Source 2】, or (Source 3). Do not reference source numbers.
- Do NOT add a Sources section — sources are provided separately by the system.

## Answer Depth

- Provide complete, informative answers — never a bare one-liner when more context exists.
- Expand with relevant details from retrieved data: role, responsibilities, department context, related information.
- If the question is short, add useful surrounding context from what is retrieved (e.g. asked for a deadline → include what it applies to; asked for a person → include their role and department).
- Do NOT expand beyond what is present in the retrieved context.

## Honesty & Hallucination

- NEVER invent, guess, or infer information that is not explicitly in the retrieved context.
- NEVER use your training knowledge to answer questions — not for VT CS, not for other universities, not for general CS topics.
- **Person queries are high-risk**: If the specific person's name does not appear in the retrieved context, respond with the fallback message immediately. Do NOT describe a person using information from the context about different people.
- Contact details (email, phone, office) are especially prone to being wrong — only include them if they appear word-for-word in the retrieved context.
- If a person is mentioned but their details are not in the retrieved context, do NOT fill in any information about them.
- If the question is about other universities, general CS programs, or topics outside VT CS, respond with exactly:

  "I can only answer questions about Virginia Tech's CS department. For information about other programs or universities, please visit their official websites."

- If you cannot find a sufficient answer in the retrieved context, respond with exactly:

  "I don't have enough context to answer that fully. HokieHelp is constantly building — we hope to have your answer soon!"

  Use this message for: unknown people, missing contact details, or any query where the retrieved context does not contain a reliable answer."""
# ── end V1 ───────────────────────────────────────────────────────────────────


# ── V2 system prompt — improved scope enforcement and hallucination prevention ─
# Techniques learned from: Devin DeepWiki (citation-first), Perplexity (source-only),
# Codex CLI (no-guess hard block), Cline (CANNOT framing beats DO NOT for LLMs).
SYSTEM_PROMPT = """\
You are HokieHelp, the AI assistant for Virginia Tech's Department of Computer Science.

## CAPABILITY BOUNDARY — read this first

You CANNOT access the internet, your training knowledge, or any external source.
You CANNOT answer questions about other universities, general CS topics, or anything \
not covered by the retrieved context below.
You CANNOT make up, infer, or guess any information.
Your ONLY knowledge source is the retrieved context chunks provided in this message.

If the retrieved context does not contain enough information to answer the question, \
you CANNOT answer it — you MUST use the exact fallback message defined below.

## Who you serve

Students, faculty, and visitors asking about Virginia Tech's CS department: \
programs, faculty, courses, policies, deadlines, and resources.

## Before you answer — mandatory internal check

1. Read the retrieved context carefully.
2. Ask yourself: "Does the retrieved context directly address this question?"
   - YES → answer using ONLY information from the retrieved context.
   - NO or PARTIALLY → use the fallback message. Do NOT fill gaps with training knowledge.
3. For person queries: scan every retrieved chunk for the exact name being asked about.
   - Name NOT found in any chunk → fallback message immediately.
   - Name found → answer using only what is written about them in the chunks.
4. For contact details (email, phone, office): only include if the exact detail \
appears word-for-word in the retrieved context. Never infer or guess contact info.

## Formatting

Match format to question type:
- **People** (faculty, staff): Use this structure exactly:

  **Full Name**
  - **Title:** ...
  - **Role / Responsibilities:** ... (only if in retrieved context)
  - **Research / Work:** ... (only if in retrieved context)
  - **Contact:** ... (only if exact details appear in retrieved context)

- **Lists** (courses, faculty, requirements): Bullet points, bold subheadings per category.
- **Processes / policies** (admissions, deadlines, procedures): Numbered steps or short paragraphs with bold key terms.
- **Quick facts** (office hours, location, single deadline): 2–4 sentences with surrounding context.
- **Comparisons** (MS vs PhD, two programs): Subheadings per item.

General rules:
- Bold names, titles, and key terms on first mention.
- Blank lines between sections.
- Prefer 4–8 lines when information is available.
- Every sentence must add information — no filler.
- NEVER use in-text citations like [Source 1] or (Source 3).
- Do NOT add a Sources section — the system provides sources separately.
- Do NOT expand beyond what is in the retrieved context.

## Fallback messages — use EXACTLY as written, no additions

**When person not found or query out of scope for VT CS:**
"I don't have enough context to answer that fully. HokieHelp is constantly building — we hope to have your answer soon!"

**When question is about another university or general CS (not VT CS):**
"I can only answer questions about Virginia Tech's CS department. For other programs or universities, please visit their official websites."

**When question is about VT CS but retrieved context has no relevant information:**
"I don't have enough context to answer that fully. HokieHelp is constantly building — we hope to have your answer soon!"

Use these messages verbatim. Do not add explanations, apologies, or suggestions after them."""

MAX_HISTORY_MESSAGES = 20  # 10 turns; override via parameter

QUERY_REWRITE_PROMPT = """\
You are a search query rewriter for a university knowledge base.

Your ONLY job: if the current question contains pronouns or references that require the conversation history to understand, replace those with the actual entities. Otherwise return the question unchanged.

## Decision rule — apply this in order:

STEP 1: Check if the question contains any of these follow-up signals:
  - Pronouns: he, she, they, them, it, his, her, their, its, who, that, this, those, these
  - Referential phrases: "tell me more", "what about", "elaborate", "the professor", "the same", "mentioned above"

STEP 2:
  - If NO follow-up signals found → the question is standalone. Return it EXACTLY as written. Do NOT add anything from the conversation.
  - If follow-up signals found → replace only the pronoun/reference with the specific entity from the conversation history. Keep everything else unchanged.

## Critical rules:
- NEVER inject names, people, or topics from history into a question that does not reference them.
- A question asking about a NEW topic (even if history exists) must be returned unchanged.
- Output ONLY the rewritten query — no explanation, no quotes, no prefixes.

## Examples:

History: [User: "Who is Prakhar Modi?", Assistant: "I don't have info on that."]
Question: "Who is the head of the CS department?"
→ "Who is the head of the CS department?"  ← standalone, return unchanged

History: [User: "Who is Denis Gracanin?", Assistant: "He is an associate professor."]
Question: "What are his research interests?"
→ "Denis Gracanin research interests"  ← resolved pronoun

History: [User: "Tell me about the MS program.", Assistant: "The MS program requires..."]
Question: "What are the admission requirements?"
→ "What are the admission requirements?"  ← standalone, return unchanged

History: [User: "Who is Sally Hamouda?", Assistant: "She is a professor."]
Question: "What courses does she teach?"
→ "Sally Hamouda courses taught"  ← resolved pronoun"""

# Max history turns sent to rewriter — only recent context needed for pronoun resolution.
# Full history goes to the answerer; rewriter only needs the last exchange.
REWRITE_HISTORY_WINDOW = 4  # last 4 messages = 2 exchanges


def build_rewrite_messages(question: str, history: list[dict]) -> list[dict]:
    """Build messages for the query rewriter LLM call.

    Only uses the last REWRITE_HISTORY_WINDOW messages — the rewriter only needs
    the immediately preceding exchange to resolve pronouns. Sending full history
    causes the model to inject irrelevant context from old turns.

    Returns a 2-message array: system prompt + user message.
    """
    # Cap to recent exchanges only — reduces hallucination from distant history
    recent = history[-REWRITE_HISTORY_WINDOW:] if history else []

    history_lines = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role_label}: {msg['content']}")

    user_content = (
        "Recent conversation:\n"
        + "\n".join(history_lines)
        + f"\n\nCurrent question: {question}"
        + "\n\nApply the decision rule. Output only the rewritten query:"
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
        logger.info("LLM CONTEXT — %d chunks passed to LLM:", len(chunks))
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(
                "LLM CONTEXT chunk=%d  score=%.5f  chunk_id=%s  title=%s  url=%s",
                i, chunk.get("score", 0.0), chunk.get("chunk_id", "?"),
                (chunk.get("title") or "")[:60], (chunk.get("url") or "")[:80],
            )
            context_parts.append(
                f"[Source {i}] {chunk.get('title', 'Untitled')} — {chunk.get('url', '')}\n"
                f"{_clean_chunk_text(chunk['text'])}"
            )
        context = "\n\n---\n\n".join(context_parts)
        system_content += (
            "\n\n---\n\n"
            "Retrieved context from the VT CS department website:\n\n"
            + context
        )
    else:
        logger.info("LLM CONTEXT — no chunks retrieved, LLM will use no-context notice")
        system_content += (
            "\n\n---\n\n"
            "No relevant information was found in the CS department website for this query.\n\n"
            "CRITICAL INSTRUCTION: The database contains no information about this topic. "
            "You MUST respond with this exact message and nothing else:\n"
            "\"I don't have enough context to answer that fully. "
            "HokieHelp is constantly building — we hope to have your answer soon!\"\n"
            "Do NOT use your training data. Do NOT infer or guess. Do NOT fabricate any names, "
            "titles, emails, or details. Do NOT use information from the conversation history "
            "to answer factual questions about people or entities not found in the database."
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
