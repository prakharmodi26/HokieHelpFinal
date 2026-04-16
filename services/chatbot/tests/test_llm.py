"""Tests for the LLM client — Ollama calls are mocked."""
from unittest.mock import MagicMock, patch

import pytest

from chatbot.llm import build_messages, build_rewrite_messages, LLMClient, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT


@pytest.fixture
def mock_ollama_client():
    return MagicMock()


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


def test_llm_client_ask_delegates_to_chat():
    """ask() is just chat() with empty history."""
    mock_ollama = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Sally is a professor."
    mock_response.get = lambda k, d=None: {"model": "qwen2.5:14b", "eval_count": 10, "total_duration": 100}.get(k, d)
    mock_ollama.chat.return_value = mock_response

    with patch("chatbot.llm.Client", return_value=mock_ollama):
        client = LLMClient(api_key="unused", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        answer = client.ask("Who is Sally?", [{"text": "Context", "url": "https://example.com", "title": "Test"}])

    assert answer == "Sally is a professor."
    call_kwargs = mock_ollama.chat.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Who is Sally?"


def test_llm_client_chat_sends_proper_messages_array():
    """chat() sends history as separate messages, not concatenated text."""
    mock_ollama = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "You asked about Denis."
    mock_response.get = lambda k, d=None: {"model": "qwen2.5:14b"}.get(k, d)
    mock_ollama.chat.return_value = mock_response

    with patch("chatbot.llm.Client", return_value=mock_ollama):
        client = LLMClient(api_key="unused", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        history = [
            {"role": "user", "content": "Who is Denis?"},
            {"role": "assistant", "content": "He is a prof."},
        ]
        answer = client.chat("Who did I ask about?", [], history)

    assert answer == "You asked about Denis."
    call_kwargs = mock_ollama.chat.call_args.kwargs
    messages = call_kwargs["messages"]
    # System + 2 history + user question = 4
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Who is Denis?"}
    assert messages[2] == {"role": "assistant", "content": "He is a prof."}
    assert messages[3] == {"role": "user", "content": "Who did I ask about?"}


def test_llm_client_chat_stream():
    """chat_stream() yields content tokens from Ollama streaming response."""
    mock_ollama = MagicMock()

    chunk1 = MagicMock()
    chunk1.message.content = "Sally"
    chunk1.get = lambda k, d=None: {"done": False}.get(k, d)

    chunk2 = MagicMock()
    chunk2.message.content = " is a professor."
    chunk2.get = lambda k, d=None: {"done": False}.get(k, d)

    chunk3 = MagicMock()
    chunk3.message.content = ""
    chunk3.get = lambda k, d=None: {"done": True}.get(k, d)

    mock_ollama.chat.return_value = iter([chunk1, chunk2, chunk3])

    with patch("chatbot.llm.Client", return_value=mock_ollama):
        client = LLMClient(api_key="unused", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        tokens = list(client.chat_stream("Who is Sally?", [{"text": "Context", "url": "u", "title": "T"}], []))

    assert tokens == ["Sally", " is a professor."]
    call_kwargs = mock_ollama.chat.call_args.kwargs
    assert call_kwargs["stream"] is True


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
