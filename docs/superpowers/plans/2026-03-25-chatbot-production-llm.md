# Production-Grade Chat Completions & Retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current "stuff everything into one user message" approach with proper OpenAI chat completions message arrays, history-aware retrieval, and SSE streaming for production-quality conversational RAG.

**Architecture:** The LLM receives a proper messages array — system prompt (with RAG context injected) followed by alternating user/assistant history messages, then the current user question. Retrieval is enhanced with conversation context so follow-up questions like "what about their research?" resolve pronouns correctly. The frontend receives streamed tokens via SSE for instant feedback.

**Tech Stack:** OpenAI Python SDK (chat completions + streaming), FastAPI StreamingResponse (SSE), Qdrant vector search, BGE embeddings, vanilla JS EventSource.

---

## Current Problems

1. **History stuffed into one user message** — All chat history is concatenated as text (`"CHAT HISTORY:\nUser: ...\nAssistant: ..."`) and crammed into a single `{"role": "user"}` message. The model loses turn structure.
2. **Only 2 messages sent to the API** — `[system, user]` regardless of conversation length. The model can't distinguish who said what.
3. **Retrieval ignores conversation context** — Search only uses the current question verbatim. "What about their research?" embeds as-is, missing the entity from prior turns.
4. **No streaming** — User stares at a spinner for 5-10 seconds. No partial feedback.
5. **Redundant endpoints** — `/ask` and `/chat` are nearly identical; `/chat` subsumes `/ask`.

## Target Architecture

```
Frontend (SSE)
    │
    ├── POST /chat/stream  ← new, SSE streaming (primary)
    ├── POST /chat          ← non-streaming JSON (backward compat)
    └── POST /ask           ← thin wrapper, delegates to /chat logic
         │
         ▼
    ┌─────────────┐
    │  Retriever   │ ← history-aware: contextualizes query with last turn
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ build_messages() │ ← proper messages array
    └──────┬──────┘
           ▼
    messages = [
      {"role": "system",    "content": SYSTEM_PROMPT + "\n\n---\n\nRetrieved context:\n..."},
      {"role": "user",      "content": "Who is Sally Hamouda?"},           ← from history
      {"role": "assistant", "content": "Sally Hamouda is a professor..."}, ← from history
      {"role": "user",      "content": "What about their research?"},      ← current question
    ]
```

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `services/chatbot/src/chatbot/llm.py` | **Rewrite** | `build_messages()` replaces `build_rag_prompt`/`build_chat_prompt`; `LLMClient` gets `chat_stream()` method |
| `services/chatbot/src/chatbot/retriever.py` | **Modify** | Add `_contextualize_query()` and `search_with_context()` |
| `services/chatbot/src/chatbot/app.py` | **Modify** | Add `/chat/stream` SSE endpoint; update `/chat` and `/ask` to use new builder + context-aware retrieval |
| `services/chatbot/src/chatbot/config.py` | **Modify** | Add `max_history_messages` field |
| `services/frontend/app.js` | **Rewrite** | Switch from fetch-then-render to SSE streaming |
| `services/chatbot/tests/test_llm.py` | **Rewrite** | Tests for `build_messages()` and updated `LLMClient` |
| `services/chatbot/tests/test_app.py` | **Modify** | Add streaming endpoint tests, update existing |
| `services/chatbot/tests/test_retriever.py` | **Modify** | Add `search_with_context()` tests |

---

## Task 1: Refactor message building — proper messages array

**Files:**
- Modify: `services/chatbot/src/chatbot/llm.py`
- Modify: `services/chatbot/tests/test_llm.py`

This is the core change. Replace the two prompt-building functions with a single `build_messages()` that returns a proper OpenAI messages array.

- [ ] **Step 1: Write failing tests for `build_messages()`**

Add these tests to `services/chatbot/tests/test_llm.py`, replacing the old `build_rag_prompt`/`build_chat_prompt` tests:

```python
from chatbot.llm import build_messages, SYSTEM_PROMPT


def test_build_messages_with_chunks_no_history():
    """First message in a conversation with RAG context."""
    chunks = [
        {"text": "Sally Hamouda is a CS professor.", "url": "https://example.com/sally", "title": "Sally Hamouda"},
    ]
    messages = build_messages("Who is Sally Hamouda?", chunks, [])

    assert len(messages) == 2  # system + user question
    assert messages[0]["role"] == "system"
    assert "Sally Hamouda is a CS professor." in messages[0]["content"]
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Who is Sally Hamouda?"}


def test_build_messages_with_history_and_chunks():
    """Follow-up question with both history and RAG context."""
    history = [
        {"role": "user", "content": "Who is Sally Hamouda?"},
        {"role": "assistant", "content": "Sally Hamouda is a professor of CS."},
    ]
    chunks = [
        {"text": "Her research focuses on CS education.", "url": "https://example.com/sally", "title": "Sally Hamouda"},
    ]
    messages = build_messages("What about their research?", chunks, history)

    assert len(messages) == 4  # system + 2 history + user question
    assert messages[0]["role"] == "system"
    assert "CS education" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Who is Sally Hamouda?"}
    assert messages[2] == {"role": "assistant", "content": "Sally Hamouda is a professor of CS."}
    assert messages[3] == {"role": "user", "content": "What about their research?"}


def test_build_messages_no_chunks_no_history():
    """Question with no RAG context and no history."""
    messages = build_messages("random question", [], [])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "No relevant information was found" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "random question"}


def test_build_messages_history_only_no_chunks():
    """Follow-up that relies purely on conversation history."""
    history = [
        {"role": "user", "content": "Who is Denis Gracanin?"},
        {"role": "assistant", "content": "Denis Gracanin is an associate professor."},
    ]
    messages = build_messages("Who did I just ask about?", [], history)

    assert len(messages) == 4
    assert "No relevant information was found" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3] == {"role": "user", "content": "Who did I just ask about?"}


def test_build_messages_truncates_long_history():
    """History beyond max_history_messages is truncated (oldest dropped)."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(30)
    ]
    messages = build_messages("latest question", [], history, max_history_messages=10)

    # system + 10 history + user question = 12
    assert len(messages) == 12
    # Should keep the LAST 10 messages (msg 20 through msg 29)
    assert messages[1]["content"] == "msg 20"
    assert messages[10]["content"] == "msg 29"


def test_build_messages_multiple_chunks_formatted():
    """Multiple RAG chunks appear in system message with source labels."""
    chunks = [
        {"text": "Chunk one text.", "url": "https://example.com/1", "title": "Page One"},
        {"text": "Chunk two text.", "url": "https://example.com/2", "title": "Page Two"},
    ]
    messages = build_messages("test question", chunks, [])

    system = messages[0]["content"]
    assert "[Source 1] Page One" in system
    assert "Chunk one text." in system
    assert "[Source 2] Page Two" in system
    assert "Chunk two text." in system
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v -k "test_build_messages" 2>&1`
Expected: FAIL — `build_messages` does not exist yet.

- [ ] **Step 3: Implement `build_messages()`**

Replace `build_rag_prompt()` and `build_chat_prompt()` in `services/chatbot/src/chatbot/llm.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v -k "test_build_messages" 2>&1`
Expected: All 6 `test_build_messages_*` tests PASS.

- [ ] **Step 5: Update `LLMClient` to use `build_messages()`**

Replace the `ask()` and `chat()` methods in `LLMClient`:

```python
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
```

- [ ] **Step 6: Update the LLMClient tests**

Replace `test_llm_client_ask`, `test_llm_client_chat`, and the old `build_*` tests in `test_llm.py`:

```python
def test_llm_client_ask_delegates_to_chat():
    """ask() is just chat() with empty history."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Sally is a professor."
    mock_client.chat.completions.create.return_value = mock_response

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        answer = client.ask("Who is Sally?", [{"text": "Context", "url": "https://example.com", "title": "Test"}])

    assert answer == "Sally is a professor."
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    # System + user question (no history)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Who is Sally?"


def test_llm_client_chat_sends_proper_messages_array():
    """chat() sends history as separate messages, not concatenated text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "You asked about Denis."
    mock_client.chat.completions.create.return_value = mock_response

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        history = [
            {"role": "user", "content": "Who is Denis?"},
            {"role": "assistant", "content": "He is a prof."},
        ]
        answer = client.chat("Who did I ask about?", [], history)

    assert answer == "You asked about Denis."
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    # System + 2 history + user question = 4
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Who is Denis?"}
    assert messages[2] == {"role": "assistant", "content": "He is a prof."}
    assert messages[3] == {"role": "user", "content": "Who did I ask about?"}
    # No "CHAT HISTORY:" text block — each message is a proper dict
    assert "CHAT HISTORY:" not in messages[1]["content"]
```

- [ ] **Step 7: Remove old `build_rag_prompt` and `build_chat_prompt` functions**

Delete the `build_rag_prompt()` and `build_chat_prompt()` functions from `llm.py`. Remove any leftover imports of them. Remove the old `test_build_rag_prompt_*` and `test_build_chat_prompt_*` tests from `test_llm.py`.

- [ ] **Step 8: Run full test suite**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v 2>&1`
Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add services/chatbot/src/chatbot/llm.py services/chatbot/tests/test_llm.py
git commit -m "feat(chatbot): proper OpenAI messages array for chat completions

Replace build_rag_prompt/build_chat_prompt with build_messages() that
returns a proper messages array: system prompt with RAG context, then
alternating user/assistant history messages, then the current question.
The model now sees real conversation structure instead of concatenated text."
```

---

## Task 2: History-aware retrieval

**Files:**
- Modify: `services/chatbot/src/chatbot/retriever.py`
- Modify: `services/chatbot/tests/test_retriever.py`

Follow-up questions like "what about their research?" need conversation context to retrieve relevant chunks. We add a method that prepends the last user message to the search query for better embedding.

- [ ] **Step 1: Write failing tests**

Add to `services/chatbot/tests/test_retriever.py`:

```python
from chatbot.retriever import contextualize_query


def test_contextualize_query_no_history():
    """Without history, query is returned unchanged."""
    assert contextualize_query("Who is Sally?", []) == "Who is Sally?"


def test_contextualize_query_with_history():
    """Follow-up question gets the last user message prepended."""
    history = [
        {"role": "user", "content": "Who is Sally Hamouda?"},
        {"role": "assistant", "content": "Sally Hamouda is a professor of CS."},
    ]
    result = contextualize_query("What about their research?", history)
    assert "Sally Hamouda" in result
    assert "What about their research?" in result


def test_contextualize_query_multiple_turns():
    """With several turns, only the last user message is used for context."""
    history = [
        {"role": "user", "content": "Tell me about the graduate program."},
        {"role": "assistant", "content": "The CS department offers MS and PhD."},
        {"role": "user", "content": "Who is the department head?"},
        {"role": "assistant", "content": "Dr. Cal Ribbens is the department head."},
    ]
    result = contextualize_query("What are their research interests?", history)
    assert "department head" in result
    assert "What are their research interests?" in result
    # Should NOT include old context about grad program
    assert "graduate program" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_retriever.py -v -k "contextualize" 2>&1`
Expected: FAIL — `contextualize_query` does not exist.

- [ ] **Step 3: Implement `contextualize_query()`**

Add to `services/chatbot/src/chatbot/retriever.py`, above the `Retriever` class:

```python
def contextualize_query(query: str, history: list[dict]) -> str:
    """Enrich a follow-up question with conversation context for better retrieval.

    Prepends the last user message so pronouns like "their", "he", "she"
    resolve to the correct entity during embedding.
    """
    if not history:
        return query

    # Find the last user message
    for msg in reversed(history):
        if msg["role"] == "user":
            return f"{msg['content']} {query}"

    return query
```

- [ ] **Step 4: Add `search_with_context()` to `Retriever`**

Add a new method to the `Retriever` class:

```python
def search_with_context(self, query: str, history: list[dict]) -> list[dict]:
    """Search with conversation-aware query enrichment."""
    enriched = contextualize_query(query, history)
    logger.info("RETRIEVER contextualized query=%r -> %r", query, enriched[:120])
    return self.search(enriched)
```

- [ ] **Step 5: Run tests**

Run: `cd services/chatbot && python -m pytest tests/test_retriever.py -v 2>&1`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add services/chatbot/src/chatbot/retriever.py services/chatbot/tests/test_retriever.py
git commit -m "feat(chatbot): history-aware retrieval for follow-up questions

Add contextualize_query() that prepends the last user message to the
search query so pronouns resolve correctly during embedding. Follow-ups
like 'what about their research?' now retrieve relevant chunks."
```

---

## Task 3: Add SSE streaming to LLMClient

**Files:**
- Modify: `services/chatbot/src/chatbot/llm.py`
- Modify: `services/chatbot/tests/test_llm.py`

- [ ] **Step 1: Write failing test for `chat_stream()`**

Add to `services/chatbot/tests/test_llm.py`:

```python
def test_llm_client_chat_stream():
    """chat_stream() yields content deltas from a streaming response."""
    mock_client = MagicMock()

    # Simulate streaming chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Sally"

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " is a professor."

    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None  # final chunk

    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        tokens = list(client.chat_stream("Who is Sally?", [{"text": "Context", "url": "u", "title": "T"}], []))

    assert tokens == ["Sally", " is a professor."]
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["stream"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v -k "test_llm_client_chat_stream" 2>&1`
Expected: FAIL — `chat_stream` method does not exist.

- [ ] **Step 3: Implement `chat_stream()`**

Add this method to `LLMClient` in `services/chatbot/src/chatbot/llm.py`:

```python
from typing import Generator

# Inside LLMClient class:

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
```

- [ ] **Step 4: Run tests**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v 2>&1`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/chatbot/src/chatbot/llm.py services/chatbot/tests/test_llm.py
git commit -m "feat(chatbot): add streaming chat_stream() method to LLMClient

Yields content tokens from the OpenAI streaming API for real-time
SSE delivery to the frontend."
```

---

## Task 4: Wire up app.py — streaming endpoint + context-aware retrieval

**Files:**
- Modify: `services/chatbot/src/chatbot/app.py`
- Modify: `services/chatbot/tests/test_app.py`

- [ ] **Step 1: Write failing tests for the streaming endpoint and updated `/chat`**

Add to `services/chatbot/tests/test_app.py`:

```python
def test_chat_uses_history_aware_retrieval(client, mock_retriever, mock_llm):
    """The /chat endpoint passes history to search_with_context."""
    resp = client.post("/chat", json={
        "question": "What about their research?",
        "history": [
            {"role": "user", "content": "Who is Dr. Smith?"},
            {"role": "assistant", "content": "Dr. Smith is a professor."},
        ],
    })
    assert resp.status_code == 200
    # Verify search_with_context was called (not plain search)
    mock_retriever.search_with_context.assert_called_once()
    call_args = mock_retriever.search_with_context.call_args
    assert call_args[0][0] == "What about their research?"
    assert len(call_args[0][1]) == 2  # history passed


def test_chat_llm_receives_proper_messages(client, mock_retriever, mock_llm):
    """The /chat endpoint passes history as separate argument (not concatenated)."""
    resp = client.post("/chat", json={
        "question": "Follow-up question",
        "history": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ],
    })
    assert resp.status_code == 200
    mock_llm.chat.assert_called_once()
    call_args = mock_llm.chat.call_args
    # chat(question, chunks, history)
    assert call_args[0][0] == "Follow-up question"
    assert len(call_args[0][2]) == 2  # history passed through


def test_stream_endpoint(client, mock_retriever, mock_llm):
    """The /chat/stream endpoint returns SSE events."""
    mock_llm.chat_stream.return_value = iter(["Hello", " world"])
    mock_retriever.search_with_context.return_value = mock_retriever.search.return_value

    resp = client.post("/chat/stream", json={
        "question": "Hello",
        "history": [],
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    assert 'data: {"type": "token", "content": "Hello"}' in body
    assert 'data: {"type": "token", "content": " world"}' in body
    assert 'data: {"type": "sources"' in body
    assert 'data: {"type": "done"}' in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_app.py -v -k "test_chat_uses_history or test_chat_llm_receives or test_stream" 2>&1`
Expected: FAIL — `search_with_context` not called, `/chat/stream` doesn't exist.

- [ ] **Step 3: Update the mock_retriever fixture**

In `test_app.py`, update the `mock_retriever` fixture to support `search_with_context`:

```python
@pytest.fixture
def mock_retriever():
    r = MagicMock()
    chunks = [
        {
            "score": 0.85,
            "chunk_id": "abc_0000",
            "text": "Sally Hamouda is a professor of CS.",
            "url": "https://website.cs.vt.edu/faculty/sally.html",
            "title": "Sally Hamouda | CS | VT",
            "headings_path": ["Faculty"],
            "page_type": "faculty",
        }
    ]
    r.search.return_value = chunks
    r.search_with_context.return_value = chunks
    return r
```

- [ ] **Step 4: Update `/chat` endpoint in `app.py`**

Replace the `/chat` endpoint to use history-aware retrieval:

```python
@app.post("/chat", response_model=AskResponse)
def chat(req: ChatRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("=" * 70)
    logger.info("NEW CHAT: %s (history_turns=%d)", req.question, len(req.history))
    logger.info("=" * 70)

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    chunks = retriever.search_with_context(req.question, history_dicts)

    answer = llm_client.chat(req.question, chunks, history_dicts)

    sources = _dedup_sources(chunks)

    logger.info("CHAT RESPONSE sent — answer_len=%d  sources=%d", len(answer), len(sources))
    return AskResponse(answer=answer, sources=sources)
```

- [ ] **Step 5: Add `/chat/stream` SSE endpoint**

Add to `app.py`:

```python
import json
from fastapi.responses import StreamingResponse


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("=" * 70)
    logger.info("NEW STREAM CHAT: %s (history_turns=%d)", req.question, len(req.history))
    logger.info("=" * 70)

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    chunks = retriever.search_with_context(req.question, history_dicts)
    sources = _dedup_sources(chunks)

    def generate():
        for token in llm_client.chat_stream(req.question, chunks, history_dicts):
            yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

        yield f'data: {json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]})}\n\n'
        yield f'data: {json.dumps({"type": "done"})}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 6: Update `/ask` to delegate to chat logic**

```python
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("=" * 70)
    logger.info("NEW QUERY: %s", req.question)
    logger.info("=" * 70)

    chunks = retriever.search(req.question)
    answer = llm_client.ask(req.question, chunks)
    sources = _dedup_sources(chunks)

    logger.info("RESPONSE sent — answer_len=%d  sources=%d", len(answer), len(sources))
    return AskResponse(answer=answer, sources=sources)
```

- [ ] **Step 7: Run full test suite**

Run: `cd services/chatbot && python -m pytest tests/ -v 2>&1`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add services/chatbot/src/chatbot/app.py services/chatbot/tests/test_app.py
git commit -m "feat(chatbot): SSE streaming endpoint and history-aware retrieval

Add /chat/stream SSE endpoint for real-time token delivery.
Update /chat to use search_with_context for better follow-up retrieval.
Keep /ask as a simple non-history wrapper."
```

---

## Task 5: Frontend SSE streaming

**Files:**
- Modify: `services/frontend/app.js`
- Modify: `services/frontend/nginx.conf` (add `/api/chat/stream` route)

- [ ] **Step 1: Check current nginx.conf**

Read `services/frontend/nginx.conf` to see the proxy rules.

- [ ] **Step 2: Add `/api/chat/stream` proxy rule to nginx.conf**

Add under the existing `/api/` location block (or add a new location):

```nginx
location /api/chat/stream {
    proxy_pass http://chatbot:8000/chat/stream;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
}
```

This disables nginx buffering so SSE events flow through immediately.

- [ ] **Step 3: Rewrite `app.js` for SSE streaming**

Replace the `sendMessage` function in `services/frontend/app.js`:

```javascript
(function () {
    "use strict";

    var API_URL = "/api/chat/stream";
    var MAX_HISTORY = 50;

    var messagesEl = document.getElementById("messages");
    var formEl = document.getElementById("chat-form");
    var inputEl = document.getElementById("question-input");
    var sendBtn = document.getElementById("send-btn");
    var typingEl = document.getElementById("typing-indicator");

    var history = [];

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function renderMarkdownLight(text) {
        var html = escapeHtml(text);
        html = html.replace(/\*\*\[([^\]]+)\]\(([^)]+)\)\*\*/g,
            '<strong><a href="$2" target="_blank" rel="noopener">$1</a></strong>');
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener">$1</a>');
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    function addMessage(role, content, sources) {
        var div = document.createElement("div");
        div.classList.add("message", role);

        if (role === "assistant") {
            div.innerHTML = renderMarkdownLight(content);
            if (sources && sources.length > 0) {
                var srcDiv = document.createElement("div");
                srcDiv.classList.add("sources");
                srcDiv.innerHTML = "<strong>Sources:</strong><br>" +
                    sources.map(function (s) {
                        return '<a href="' + escapeHtml(s.url) + '" target="_blank" rel="noopener">' +
                            escapeHtml(s.title || s.url) + "</a>";
                    }).join("<br>");
                div.appendChild(srcDiv);
            }
        } else if (role === "error") {
            div.textContent = content;
        } else {
            div.textContent = content;
        }

        messagesEl.appendChild(div);
        messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
        return div;
    }

    function setLoading(loading) {
        sendBtn.disabled = loading;
        inputEl.disabled = loading;
        typingEl.classList.toggle("hidden", !loading);
        if (loading) {
            messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
        }
    }

    async function sendMessage(question) {
        addMessage("user", question);
        var sendHistory = history.slice(-MAX_HISTORY);
        setLoading(true);

        // Create assistant message div for streaming
        var assistantDiv = document.createElement("div");
        assistantDiv.classList.add("message", "assistant");
        messagesEl.appendChild(assistantDiv);

        var fullAnswer = "";

        try {
            var resp = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    history: sendHistory,
                }),
            });

            if (!resp.ok) {
                var errData = await resp.json().catch(function () { return {}; });
                throw new Error(errData.detail || "Server error " + resp.status);
            }

            var reader = resp.body.getReader();
            var decoder = new TextDecoder();
            var buffer = "";

            while (true) {
                var result = await reader.read();
                if (result.done) break;

                buffer += decoder.decode(result.value, { stream: true });
                var lines = buffer.split("\n");
                buffer = lines.pop();

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i].trim();
                    if (!line.startsWith("data: ")) continue;
                    var data = line.slice(6);

                    try {
                        var event = JSON.parse(data);
                    } catch (e) {
                        continue;
                    }

                    if (event.type === "token") {
                        fullAnswer += event.content;
                        assistantDiv.innerHTML = renderMarkdownLight(fullAnswer);
                        messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
                    } else if (event.type === "sources") {
                        if (event.sources && event.sources.length > 0) {
                            var srcDiv = document.createElement("div");
                            srcDiv.classList.add("sources");
                            srcDiv.innerHTML = "<strong>Sources:</strong><br>" +
                                event.sources.map(function (s) {
                                    return '<a href="' + escapeHtml(s.url) + '" target="_blank" rel="noopener">' +
                                        escapeHtml(s.title || s.url) + "</a>";
                                }).join("<br>");
                            assistantDiv.appendChild(srcDiv);
                        }
                    }
                }
            }

            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: fullAnswer });

        } catch (err) {
            if (!fullAnswer) {
                assistantDiv.remove();
            }
            addMessage("error", "Error: " + err.message);
        } finally {
            setLoading(false);
            inputEl.focus();
        }
    }

    formEl.addEventListener("submit", function (e) {
        e.preventDefault();
        var q = inputEl.value.trim();
        if (!q) return;
        inputEl.value = "";
        sendMessage(q);
    });

    addMessage("assistant", "Welcome to HokieHelp! Ask me anything about Virginia Tech's Computer Science department.");
})();
```

- [ ] **Step 4: Commit**

```bash
git add services/frontend/app.js services/frontend/nginx.conf
git commit -m "feat(frontend): SSE streaming for real-time chat responses

Replace fetch-then-render with SSE streaming. Tokens appear as they
arrive from the LLM. Sources are appended after the stream completes.
Add nginx proxy rules for the streaming endpoint."
```

---

## Task 6: Config update + cleanup + final integration test

**Files:**
- Modify: `services/chatbot/src/chatbot/config.py`
- Modify: `services/chatbot/tests/test_config.py`

- [ ] **Step 1: Add `max_history_messages` to config**

In `services/chatbot/src/chatbot/config.py`, add to the dataclass:

```python
max_history_messages: int
```

And in `from_env()`:

```python
max_history_messages=int(os.environ.get("MAX_HISTORY_MESSAGES", "20")),
```

- [ ] **Step 2: Wire config into `LLMClient`**

Update `LLMClient.__init__()` in `llm.py` to accept `max_history_messages`:

```python
def __init__(self, api_key: str, base_url: str, model: str, max_history_messages: int = 20) -> None:
    self._client = OpenAI(api_key=api_key, base_url=base_url)
    self._model = model
    self._max_history = max_history_messages
    logger.info("LLM client ready — model=%s  base_url=%s  max_history=%d", model, base_url, max_history_messages)
```

Update `chat()` and `chat_stream()` to pass `self._max_history` to `build_messages()`:

```python
messages = build_messages(question, chunks, history, max_history_messages=self._max_history)
```

- [ ] **Step 3: Update `app.py` startup to pass config**

In `app.py` `startup()`:

```python
llm_client = LLMClient(
    api_key=cfg.llm_api_key,
    base_url=cfg.llm_base_url,
    model=cfg.llm_model,
    max_history_messages=cfg.max_history_messages,
)
```

- [ ] **Step 4: Run full test suite**

Run: `cd services/chatbot && python -m pytest tests/ -v 2>&1`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/chatbot/src/chatbot/config.py services/chatbot/src/chatbot/llm.py services/chatbot/src/chatbot/app.py services/chatbot/tests/
git commit -m "feat(chatbot): configurable max history messages

Add MAX_HISTORY_MESSAGES env var (default 20) to control how many
history messages are included in the LLM messages array."
```

---

## Summary of Changes

| Before | After |
|--------|-------|
| History concatenated as text in one user message | History as proper alternating user/assistant messages |
| 2 messages sent to LLM: `[system, user]` | N+2 messages: `[system, ...history, user]` |
| RAG context mixed into user message | RAG context injected into system message |
| Search uses raw question only | Search contextualizes with last user message |
| No streaming — user waits for full response | SSE streaming — tokens appear in real-time |
| Separate /ask and /chat logic | /ask delegates to /chat; /chat/stream for SSE |
| Hardcoded 10-message history limit | Configurable via `MAX_HISTORY_MESSAGES` env var |
