"""Tests for the LLM client — HTTP calls are mocked."""
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
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Sally is a professor."}}

    with patch("chatbot.llm.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        client = LLMClient(api_key="sk-test", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        answer = client.ask("Who is Sally?", [{"text": "Context", "url": "https://example.com", "title": "Test"}])

    assert answer == "Sally is a professor."


def test_llm_client_chat_sends_proper_messages_array():
    """chat() sends history as separate messages."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "You asked about Denis."}}

    with patch("chatbot.llm.httpx.Client") as mock_client_cls:
        http = mock_client_cls.return_value.__enter__.return_value
        http.post.return_value = mock_resp
        client = LLMClient(api_key="sk-test", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        history = [
            {"role": "user", "content": "Who is Denis?"},
            {"role": "assistant", "content": "He is a prof."},
        ]
        answer = client.chat("Who did I ask about?", [], history)

    assert answer == "You asked about Denis."
    call_kwargs = http.post.call_args.kwargs
    messages = call_kwargs["json"]["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Who is Denis?"}
    assert messages[2] == {"role": "assistant", "content": "He is a prof."}
    assert messages[3] == {"role": "user", "content": "Who did I ask about?"}


def test_llm_client_chat_stream():
    """chat_stream() yields content tokens from Ollama streaming response."""
    stream_lines = [
        '{"message": {"content": "Sally"}, "done": false}',
        '{"message": {"content": " is a professor."}, "done": false}',
        '{"message": {"content": ""}, "done": true}',
    ]

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = iter(stream_lines)
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("chatbot.llm.httpx.Client") as mock_client_cls:
        http = mock_client_cls.return_value.__enter__.return_value
        http.stream.return_value = mock_response
        client = LLMClient(api_key="sk-test", base_url="http://ollama:11434/v1", model="qwen2.5:14b")
        tokens = list(client.chat_stream("Who is Sally?", [{"text": "Context", "url": "u", "title": "T"}], []))

    assert tokens == ["Sally", " is a professor."]
    call_kwargs = http.stream.call_args.kwargs
    assert call_kwargs["json"]["stream"] is True
