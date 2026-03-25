"""Tests for the LLM client — API calls are mocked."""
from unittest.mock import MagicMock, patch

from chatbot.llm import build_messages, LLMClient, SYSTEM_PROMPT


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
