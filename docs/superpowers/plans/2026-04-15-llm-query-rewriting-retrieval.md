# LLM Query Rewriting & Retrieval Overhaul

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace naive regex-based query contextualization with LLM-powered query rewriting for dramatically better retrieval accuracy on follow-up questions, and modernize the system prompt for versatile answer formatting.

**Architecture:** Two-stage LLM pipeline. Stage 1: rewrite user question + chat history into a standalone search query via a dedicated LLM call. Stage 2: retrieve chunks with rewritten query, then generate answer using original question + chunks + history. The retriever becomes a pure search engine — all query understanding moves to the LLM layer.

**Tech Stack:** Ollama (`ollama>=0.4.0`), Qdrant, SentenceTransformers (BGE-large-en-v1.5), FastAPI

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `services/chatbot/src/chatbot/llm.py` | Add `rewrite_query()` method, `QUERY_REWRITE_PROMPT`, rewrite `SYSTEM_PROMPT` |
| Modify | `services/chatbot/src/chatbot/config.py` | Add `rewriter_model` config field |
| Modify | `services/chatbot/src/chatbot/app.py` | Wire rewrite step into `/chat` and `/chat/stream` before retrieval |
| Modify | `services/chatbot/src/chatbot/retriever.py` | Remove `contextualize_query()`, `search_with_context()`, follow-up keyword machinery |
| Modify | `services/chatbot/tests/test_llm.py` | Add rewrite tests, update `build_messages` tests for new prompt |
| Modify | `services/chatbot/tests/test_retriever.py` | Remove contextualize tests, clean up fixture |
| Modify | `services/chatbot/tests/test_app.py` | Update to test rewrite → search → answer flow |
| Modify | `services/chatbot/src/chatbot/guard.py` | No changes — prompt guard stays as-is |

---

### Task 1: Add `rewriter_model` to Config

**Files:**
- Modify: `services/chatbot/src/chatbot/config.py`
- Modify: `services/chatbot/tests/test_config.py`

- [ ] **Step 1: Write failing test for new config field**

Add to `services/chatbot/tests/test_config.py`:

```python
def test_rewriter_model_defaults_to_llm_model(monkeypatch):
    """REWRITER_MODEL defaults to LLM_MODEL when not set."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    # Don't set REWRITER_MODEL
    cfg = ChatbotConfig.from_env()
    assert cfg.rewriter_model == "qwen2.5:14b"


def test_rewriter_model_override(monkeypatch):
    """REWRITER_MODEL can be set independently."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("REWRITER_MODEL", "qwen2.5:7b")
    cfg = ChatbotConfig.from_env()
    assert cfg.rewriter_model == "qwen2.5:7b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_config.py -v -k "rewriter"`
Expected: FAIL — `ChatbotConfig` has no `rewriter_model` field

- [ ] **Step 3: Add `rewriter_model` field to ChatbotConfig**

In `services/chatbot/src/chatbot/config.py`, add the field to the dataclass:

```python
@dataclass(frozen=True)
class ChatbotConfig:
    """Immutable configuration for the chatbot service."""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    rewriter_model: str          # <-- NEW FIELD
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model: str
    top_k: int
    min_score: float
    max_history_messages: int
    hybrid_enabled: bool
    keyword_search_limit: int
    rrf_k: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    follow_up_keywords: tuple[str, ...]
```

And in `from_env()`, after the `llm_model` line:

```python
llm_model=os.environ.get("LLM_MODEL", "qwen2.5:14b"),
rewriter_model=os.environ.get("REWRITER_MODEL", os.environ.get("LLM_MODEL", "qwen2.5:14b")),
```

Note: `rewriter_model` defaults to whatever `LLM_MODEL` is set to. This avoids loading two models in GPU memory (Ollama shares the already-loaded model). If user wants a lighter rewriter, they can set `REWRITER_MODEL=qwen2.5:7b` explicitly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/chatbot && python -m pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add services/chatbot/src/chatbot/config.py services/chatbot/tests/test_config.py
git commit -m "feat(chatbot): add rewriter_model config field"
```

---

### Task 2: Add Query Rewriter to LLMClient

**Files:**
- Modify: `services/chatbot/src/chatbot/llm.py`
- Modify: `services/chatbot/tests/test_llm.py`

- [ ] **Step 1: Write failing tests for `rewrite_query`**

Add to `services/chatbot/tests/test_llm.py`:

```python
from chatbot.llm import build_rewrite_messages, QUERY_REWRITE_PROMPT


def test_build_rewrite_messages_with_history():
    """Rewrite messages include history and current question."""
    history = [
        {"role": "user", "content": "Who is Dr. Sally Hamouda?"},
        {"role": "assistant", "content": "Dr. Sally Hamouda is an associate professor in CS."},
    ]
    messages = build_rewrite_messages("What are her research interests?", history)

    assert len(messages) == 2  # system + user
    assert messages[0]["role"] == "system"
    assert QUERY_REWRITE_PROMPT in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Sally Hamouda" in messages[1]["content"]
    assert "What are her research interests?" in messages[1]["content"]


def test_build_rewrite_messages_formats_history():
    """History is formatted as User:/Assistant: lines."""
    history = [
        {"role": "user", "content": "Tell me about grad programs."},
        {"role": "assistant", "content": "VT CS offers MS and PhD degrees."},
    ]
    messages = build_rewrite_messages("What are the requirements?", history)
    user_content = messages[1]["content"]
    assert "User: Tell me about grad programs." in user_content
    assert "Assistant: VT CS offers MS and PhD degrees." in user_content
    assert "Current question: What are the requirements?" in user_content


def test_rewrite_query_calls_ollama(mock_ollama_client):
    """rewrite_query sends proper messages and returns stripped response."""
    mock_response = MagicMock()
    mock_response.message.content = "  Dr. Sally Hamouda research interests  "
    mock_response.get = lambda k, d=None: {}.get(k, d)
    mock_ollama_client.chat.return_value = mock_response

    with patch("chatbot.llm.Client", return_value=mock_ollama_client):
        client = LLMClient(
            api_key="unused",
            base_url="http://ollama:11434/v1",
            model="qwen2.5:14b",
            rewriter_model="qwen2.5:14b",
        )
        result = client.rewrite_query(
            "What are her research interests?",
            [
                {"role": "user", "content": "Who is Dr. Sally Hamouda?"},
                {"role": "assistant", "content": "She is a professor."},
            ],
        )

    assert result == "Dr. Sally Hamouda research interests"
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "qwen2.5:14b"
    assert len(call_kwargs["messages"]) == 2


def test_rewrite_query_skips_when_no_history(mock_ollama_client):
    """rewrite_query returns original question when history is empty."""
    with patch("chatbot.llm.Client", return_value=mock_ollama_client):
        client = LLMClient(
            api_key="unused",
            base_url="http://ollama:11434/v1",
            model="qwen2.5:14b",
            rewriter_model="qwen2.5:14b",
        )
        result = client.rewrite_query("Who is Sally Hamouda?", [])

    assert result == "Who is Sally Hamouda?"
    mock_ollama_client.chat.assert_not_called()


def test_rewrite_query_fallback_on_error(mock_ollama_client):
    """On LLM error, rewrite_query falls back to original question."""
    mock_ollama_client.chat.side_effect = Exception("Ollama down")

    with patch("chatbot.llm.Client", return_value=mock_ollama_client):
        client = LLMClient(
            api_key="unused",
            base_url="http://ollama:11434/v1",
            model="qwen2.5:14b",
            rewriter_model="qwen2.5:14b",
        )
        result = client.rewrite_query(
            "What about their office hours?",
            [{"role": "user", "content": "Who is Dr. Smith?"},
             {"role": "assistant", "content": "Dr. Smith is a professor."}],
        )

    assert result == "What about their office hours?"
```

Also add this fixture near the top of the test file:

```python
@pytest.fixture
def mock_ollama_client():
    return MagicMock()
```

And add the missing import at the top:

```python
import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v -k "rewrite"`
Expected: FAIL — `build_rewrite_messages` and `QUERY_REWRITE_PROMPT` don't exist, `LLMClient` doesn't accept `rewriter_model`

- [ ] **Step 3: Implement `QUERY_REWRITE_PROMPT` and `build_rewrite_messages`**

Add to `services/chatbot/src/chatbot/llm.py`, above the `LLMClient` class:

```python
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
```

- [ ] **Step 4: Update `LLMClient.__init__` to accept `rewriter_model`**

Update the `__init__` method in `LLMClient`:

```python
class LLMClient:
    """Calls Ollama via the official ollama-python library."""

    def __init__(self, api_key: str, base_url: str, model: str, rewriter_model: str = "", max_history_messages: int = 20) -> None:
        host = base_url.rstrip("/").removesuffix("/v1")
        self._client = Client(host=host)
        self._model = model
        self._rewriter_model = rewriter_model or model
        self._max_history = max_history_messages
        logger.info(
            "LLM client ready — model=%s  rewriter=%s  host=%s  max_history=%d",
            model, self._rewriter_model, host, max_history_messages,
        )
```

- [ ] **Step 5: Implement `rewrite_query` method**

Add to the `LLMClient` class, after `ask()`:

```python
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
```

Note: `temperature=0.0` for deterministic rewrites — we want the same question + history to always produce the same search query.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v`
Expected: ALL PASS (both new rewrite tests and existing tests)

- [ ] **Step 7: Commit**

```bash
git add services/chatbot/src/chatbot/llm.py services/chatbot/tests/test_llm.py
git commit -m "feat(chatbot): add LLM-based query rewriting for follow-up questions"
```

---

### Task 3: Update System Prompt

**Files:**
- Modify: `services/chatbot/src/chatbot/llm.py`
- Modify: `services/chatbot/tests/test_llm.py`

- [ ] **Step 1: Write failing test for updated system prompt**

Update existing tests in `services/chatbot/tests/test_llm.py` that check `SYSTEM_PROMPT`:

```python
def test_system_prompt_contains_key_instructions():
    """System prompt includes grounding rules and formatting guidance."""
    assert "retrieved context" in SYSTEM_PROMPT.lower()
    assert "conversation history" in SYSTEM_PROMPT.lower()
    # New prompt should NOT have rigid bullet-only formatting
    assert "Use bullet points for all answers" not in SYSTEM_PROMPT
    # New prompt should handle different answer types
    assert "question type" in SYSTEM_PROMPT.lower() or "format" in SYSTEM_PROMPT.lower()
    # Should NOT instruct LLM to add sources (handled structurally by app)
    assert "Sources:" not in SYSTEM_PROMPT or "Do NOT add a Sources section" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py::test_system_prompt_contains_key_instructions -v`
Expected: FAIL — current prompt has "Use bullet points for all answers"

- [ ] **Step 3: Replace `SYSTEM_PROMPT` in `llm.py`**

Replace the entire `SYSTEM_PROMPT` constant:

```python
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
```

- [ ] **Step 4: Update `build_messages` to remove the sources section from context formatting**

The context injection in `build_messages` currently uses `[Source N]` labels. Keep those — they help the LLM distinguish chunks — but the system prompt now tells it not to cite them in the answer. No code change needed here.

- [ ] **Step 5: Run all LLM tests**

Run: `cd services/chatbot && python -m pytest tests/test_llm.py -v`
Expected: ALL PASS

Some existing tests check `SYSTEM_PROMPT in messages[0]["content"]` — that still works since `build_messages` puts `SYSTEM_PROMPT` in the system message. The test `test_build_messages_with_chunks_no_history` checks that chunk text appears in the system message — that's unchanged. The only test that might break is if any test checks for exact substring in the old prompt. Check and fix:

- `test_build_messages_no_chunks_no_history` checks for `"No relevant information was found"` — this string is in the context injection code (not in `SYSTEM_PROMPT`), so it still works.

- [ ] **Step 6: Commit**

```bash
git add services/chatbot/src/chatbot/llm.py services/chatbot/tests/test_llm.py
git commit -m "feat(chatbot): modernize system prompt for versatile answer formatting"
```

---

### Task 4: Clean Up Retriever — Remove Naive Contextualization

**Files:**
- Modify: `services/chatbot/src/chatbot/retriever.py`
- Modify: `services/chatbot/tests/test_retriever.py`
- Modify: `services/chatbot/src/chatbot/config.py`

- [ ] **Step 1: Remove contextualize tests from `test_retriever.py`**

Delete these test functions entirely from `services/chatbot/tests/test_retriever.py`:
- `test_search_with_context_uses_enriched_query`
- `test_contextualize_query_no_history`
- `test_contextualize_query_with_history`
- `test_contextualize_query_multiple_turns`
- `test_contextualize_query_standalone_question_not_enriched`
- `test_contextualize_query_no_pattern_always_enriches`

Also remove these imports from the test file:
```python
# Remove from imports:
contextualize_query, _build_follow_up_pattern, DEFAULT_FOLLOW_UP_KEYWORDS
```

The remaining imports should be:
```python
from chatbot.retriever import Retriever, BGE_QUERY_PREFIX
```

Also remove the `follow_up_keywords` and any contextualizing-related setup from the `retriever` fixture if present (currently the fixture doesn't pass `follow_up_keywords`, so it should be clean).

- [ ] **Step 2: Run tests to confirm remaining retriever tests still pass**

Run: `cd services/chatbot && python -m pytest tests/test_retriever.py -v`
Expected: PASS (only `test_bge_query_prefix`, `test_search_returns_results`, `test_search_prepends_bge_prefix`, `test_search_empty_results` remain)

- [ ] **Step 3: Remove contextualization code from `retriever.py`**

Delete from `services/chatbot/src/chatbot/retriever.py`:
- The `DEFAULT_FOLLOW_UP_KEYWORDS` constant
- The `_build_follow_up_pattern()` function
- The `contextualize_query()` function
- The `search_with_context()` method from the `Retriever` class
- The `follow_up_keywords` parameter from `Retriever.__init__` (and `self._follow_up_pattern`)
- The `import re` and `from typing import Sequence` (if `Sequence` is no longer used — check `List` is still imported)

The `__init__` signature becomes:

```python
def __init__(
    self,
    embedding_model: str,
    qdrant_host: str,
    qdrant_port: int,
    collection: str,
    top_k: int,
    min_score: float = 0.53,
    hybrid_enabled: bool = True,
    keyword_search_limit: int = 10,
    rrf_k: int = 60,
) -> None:
```

Remove `self._follow_up_pattern = ...` from the init body.

- [ ] **Step 4: Remove `follow_up_keywords` from `ChatbotConfig`**

In `services/chatbot/src/chatbot/config.py`:
- Remove the `follow_up_keywords: tuple[str, ...]` field from the dataclass
- Remove the `follow_up_keywords=tuple(...)` from `from_env()`

- [ ] **Step 5: Run all tests to confirm nothing broke**

Run: `cd services/chatbot && python -m pytest tests/ -v`
Expected: PASS (some app tests may fail because `app.py` still references `search_with_context` and `follow_up_keywords` — that's expected and fixed in Task 5)

- [ ] **Step 6: Commit**

```bash
git add services/chatbot/src/chatbot/retriever.py services/chatbot/tests/test_retriever.py services/chatbot/src/chatbot/config.py
git commit -m "refactor(chatbot): remove naive query contextualization from retriever"
```

---

### Task 5: Wire Rewrite Into App Endpoints

**Files:**
- Modify: `services/chatbot/src/chatbot/app.py`
- Modify: `services/chatbot/tests/test_app.py`

- [ ] **Step 1: Write failing tests for new rewrite flow**

Replace `test_chat_uses_history_aware_retrieval` in `services/chatbot/tests/test_app.py` with:

```python
def test_chat_rewrites_query_before_search(client, mock_retriever, mock_llm):
    """The /chat endpoint calls rewrite_query, then searches with rewritten query."""
    mock_llm.rewrite_query.return_value = "Dr. Smith research interests"

    resp = client.post("/chat", json={
        "question": "What about their research?",
        "history": [
            {"role": "user", "content": "Who is Dr. Smith?"},
            {"role": "assistant", "content": "Dr. Smith is a professor."},
        ],
    })
    assert resp.status_code == 200

    # rewrite_query was called with original question + history
    mock_llm.rewrite_query.assert_called_once()
    rw_args = mock_llm.rewrite_query.call_args[0]
    assert rw_args[0] == "What about their research?"
    assert len(rw_args[1]) == 2

    # retriever.search was called with the REWRITTEN query
    mock_retriever.search.assert_called_once_with("Dr. Smith research interests")

    # LLM chat was called with ORIGINAL question (not rewritten)
    mock_llm.chat.assert_called_once()
    chat_args = mock_llm.chat.call_args[0]
    assert chat_args[0] == "What about their research?"


def test_chat_skips_rewrite_when_no_history(client, mock_retriever, mock_llm):
    """Without history, /chat uses raw query directly (no rewrite call)."""
    resp = client.post("/chat", json={
        "question": "Who is Sally Hamouda?",
        "history": [],
    })
    assert resp.status_code == 200

    # rewrite_query still called but returns original (empty history)
    mock_llm.rewrite_query.assert_called_once()
    # retriever.search called with original query
    mock_retriever.search.assert_called_once_with("Who is Sally Hamouda?")


def test_stream_rewrites_query_before_search(client, mock_retriever, mock_llm):
    """The /chat/stream endpoint also uses query rewriting."""
    mock_llm.rewrite_query.return_value = "Dr. Smith office hours"
    mock_llm.chat_stream.return_value = iter(["Office hours are Monday."])

    resp = client.post("/chat/stream", json={
        "question": "What about his office hours?",
        "history": [
            {"role": "user", "content": "Who is Dr. Smith?"},
            {"role": "assistant", "content": "Dr. Smith is a professor."},
        ],
    })
    assert resp.status_code == 200

    mock_llm.rewrite_query.assert_called_once()
    mock_retriever.search.assert_called_once_with("Dr. Smith office hours")
```

Also update `test_chat_empty_history` if it uses `search_with_context`:

```python
def test_chat_empty_history(client, mock_retriever, mock_llm):
    resp = client.post("/chat", json={
        "question": "Hello",
        "history": [],
    })
    assert resp.status_code == 200
```

And update the `mock_llm` fixture to include `rewrite_query`:

```python
@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ask.return_value = "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    llm.chat.return_value = "Sally Hamouda is a professor of Computer Science at Virginia Tech."
    llm.rewrite_query.return_value = "Sally Hamouda"  # default rewrite
    return llm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/chatbot && python -m pytest tests/test_app.py -v -k "rewrite"`
Expected: FAIL — app.py still calls `search_with_context` which no longer exists

- [ ] **Step 3: Update `app.py` startup to pass `rewriter_model`**

In the `startup()` function, update the `LLMClient` initialization:

```python
llm_client = LLMClient(
    api_key=cfg.llm_api_key,
    base_url=cfg.llm_base_url,
    model=cfg.llm_model,
    rewriter_model=cfg.rewriter_model,
    max_history_messages=cfg.max_history_messages,
)
```

Also update the `Retriever` initialization to remove `follow_up_keywords`:

```python
retriever = Retriever(
    embedding_model=cfg.embedding_model,
    qdrant_host=cfg.qdrant_host,
    qdrant_port=cfg.qdrant_port,
    collection=cfg.qdrant_collection,
    top_k=cfg.top_k,
    min_score=cfg.min_score,
    hybrid_enabled=cfg.hybrid_enabled,
    keyword_search_limit=cfg.keyword_search_limit,
    rrf_k=cfg.rrf_k,
)
```

- [ ] **Step 4: Update `/chat` endpoint to use rewrite → search → answer**

Replace the `/chat` endpoint:

```python
@app.post("/chat", response_model=AskResponse)
def chat(
    req: ChatRequest,
    response: Response,
    session_id: str = Depends(_check_rate_limit),
) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        check_prompt(req.question)
    except PromptRejected as e:
        logger.info("CHAT rejected by guard: %s (session=%s)", e, session_id)
        raise HTTPException(status_code=400, detail=str(e))

    import time as _t
    t0 = _t.monotonic()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    logger.info("CHAT start session=%s history_turns=%d q=%r", session_id, len(req.history), req.question[:200])

    try:
        # Stage 1: Rewrite query for retrieval
        search_query = llm_client.rewrite_query(req.question, history_dicts)
        logger.info("CHAT rewrite in %.0fms: %r -> %r", (_t.monotonic()-t0)*1000, req.question[:100], search_query[:100])

        # Stage 2: Retrieve chunks using rewritten query
        t1 = _t.monotonic()
        chunks = retriever.search(search_query)
        logger.info("CHAT retrieved %d chunks in %.0fms", len(chunks), (_t.monotonic()-t1)*1000)

        # Stage 3: Generate answer using original question + chunks + history
        t2 = _t.monotonic()
        answer = llm_client.chat(req.question, chunks, history_dicts)
        logger.info("CHAT llm answer in %.0fms (answer_len=%d)", (_t.monotonic()-t2)*1000, len(answer))

        sources = _dedup_sources(chunks)
        _set_session_cookie(response, session_id)
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("CHAT failed session=%s: %s", session_id, exc)
        raise
```

- [ ] **Step 5: Update `/chat/stream` endpoint the same way**

Replace the `/chat/stream` endpoint:

```python
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    request: Request,
    session_id: str = Depends(_check_rate_limit),
) -> StreamingResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        check_prompt(req.question)
    except PromptRejected as e:
        logger.info("STREAM rejected by guard: %s (session=%s)", e, session_id)
        raise HTTPException(status_code=400, detail=str(e))

    import time as _t
    t0 = _t.monotonic()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    logger.info("STREAM start session=%s history_turns=%d q=%r", session_id, len(req.history), req.question[:200])

    # Stage 1: Rewrite query for retrieval
    search_query = llm_client.rewrite_query(req.question, history_dicts)
    logger.info("STREAM rewrite in %.0fms: %r -> %r", (_t.monotonic()-t0)*1000, req.question[:100], search_query[:100])

    # Stage 2: Retrieve chunks using rewritten query
    t1 = _t.monotonic()
    chunks = retriever.search(search_query)
    logger.info("STREAM retrieved %d chunks in %.0fms", len(chunks), (_t.monotonic()-t1)*1000)
    sources = _dedup_sources(chunks)

    def generate():
        import time as _t
        t2 = _t.monotonic()
        tokens = 0
        try:
            for token in llm_client.chat_stream(req.question, chunks, history_dicts):
                tokens += 1
                yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
            yield f'data: {json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'
            logger.info("STREAM done session=%s tokens=%d llm_ms=%.0f", session_id, tokens, (_t.monotonic()-t2)*1000)
        except Exception as exc:
            logger.exception("STREAM generation error session=%s after %d tokens: %s", session_id, tokens, exc)
            yield f'data: {json.dumps({"type": "error", "content": "An error occurred while generating the response."})}\n\n'

    headers = {
        "Set-Cookie": (
            f"{SESSION_COOKIE}={session_id}; Max-Age={COOKIE_MAX_AGE}; "
            "HttpOnly; SameSite=Strict; Secure; Path=/"
        )
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)
```

- [ ] **Step 6: Run all tests**

Run: `cd services/chatbot && python -m pytest tests/ -v`
Expected: ALL PASS

If `test_chat_llm_receives_proper_history` fails because it checks `search_with_context` was called, update it:

```python
def test_chat_llm_receives_proper_history(client, mock_retriever, mock_llm):
    """The /chat endpoint passes history to both rewrite and answer."""
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
```

- [ ] **Step 7: Commit**

```bash
git add services/chatbot/src/chatbot/app.py services/chatbot/tests/test_app.py
git commit -m "feat(chatbot): wire LLM query rewriting into chat and stream endpoints"
```

---

### Task 6: Final Integration Verification

**Files:**
- All modified files from Tasks 1–5

- [ ] **Step 1: Run full test suite**

Run: `cd services/chatbot && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Verify no import errors**

Run: `cd services/chatbot && python -c "from chatbot.app import app; print('OK')"`
Expected: Prints `OK` (no import errors)

- [ ] **Step 3: Verify the retriever no longer exports removed symbols**

Run: `cd services/chatbot && python -c "from chatbot.retriever import Retriever; print('Retriever OK')"`
Expected: `Retriever OK`

Run: `cd services/chatbot && python -c "from chatbot.retriever import contextualize_query" 2>&1 || true`
Expected: `ImportError` (function was removed)

- [ ] **Step 4: Verify LLM exports**

Run: `cd services/chatbot && python -c "from chatbot.llm import build_rewrite_messages, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT; print('LLM OK')"`
Expected: `LLM OK`

- [ ] **Step 5: Commit (if any fixes were needed)**

```bash
git add -u services/chatbot/
git commit -m "fix(chatbot): integration fixes for query rewriting pipeline"
```

---

## Summary of Changes

| Before | After |
|--------|-------|
| Regex keyword matching for follow-ups | LLM rewrites every follow-up into standalone query |
| Prepends last user message blindly | Replaces pronouns/references with actual entities |
| Rigid bullet-only system prompt | Format-adaptive: bullets, paragraphs, numbered steps |
| LLM generates Sources section in text | Sources handled structurally by app (no duplication) |
| `search_with_context()` in retriever | Rewrite in LLM layer → `search()` in retriever |
| `follow_up_keywords` config | `rewriter_model` config (defaults to `LLM_MODEL`) |

## Design Decisions

1. **Same model for rewriting and answering by default** — avoids loading two models in GPU memory. Ollama shares the already-loaded model when `rewriter_model == llm_model`. Set `REWRITER_MODEL` env var to use a lighter model if latency matters.

2. **`temperature=0.0` for rewriter** — deterministic rewrites. Same question + history always produces same search query. Eliminates variance in retrieval results.

3. **Graceful fallback** — if rewriter LLM call fails, falls back to original question. Retrieval quality degrades to current behavior, but doesn't crash.

4. **`/ask` endpoint unchanged** — no history, no rewrite needed. Only `/chat` and `/chat/stream` use the rewrite step.

5. **Sources removed from system prompt** — the app already returns structured sources via `_dedup_sources()`. Having the LLM also generate a Sources section was redundant and could hallucinate URLs.
