"""Tests for the LLM client — API calls are mocked."""
from unittest.mock import MagicMock, patch

from chatbot.llm import LLMClient, build_rag_prompt, build_chat_messages, SYSTEM_PROMPT


def test_build_rag_prompt_with_chunks():
    chunks = [
        {"text": "Sally Hamouda is a CS professor.", "url": "https://example.com/sally", "title": "Sally Hamouda"},
        {"text": "She researches CS education.", "url": "https://example.com/sally2", "title": "Sally Hamouda"},
    ]
    prompt = build_rag_prompt("Who is Sally Hamouda?", chunks)
    assert "Sally Hamouda is a CS professor." in prompt
    assert "She researches CS education." in prompt
    assert "Who is Sally Hamouda?" in prompt
    assert "https://example.com/sally" in prompt


def test_build_rag_prompt_no_chunks():
    prompt = build_rag_prompt("random question", [])
    assert "random question" in prompt


def test_llm_client_ask():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Sally Hamouda is a professor of CS."
    mock_client.chat.completions.create.return_value = mock_response

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        answer = client.ask("Who is Sally?", [{"text": "Some context", "url": "https://example.com", "title": "Test"}])

    assert answer == "Sally Hamouda is a professor of CS."
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-oss-120b"


def test_build_chat_messages_with_history():
    history = [
        {"role": "user", "content": "Who is the department head?"},
        {"role": "assistant", "content": "Dr. Smith is the department head."},
    ]
    chunks = [{"title": "Faculty", "url": "https://example.com", "text": "Dr. Jones teaches CS."}]
    messages = build_chat_messages("What does Dr. Jones teach?", chunks, history)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Who is the department head?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Dr. Smith is the department head."
    assert messages[3]["role"] == "user"
    assert "Dr. Jones teaches CS." in messages[3]["content"]
    assert "What does Dr. Jones teach?" in messages[3]["content"]


def test_build_chat_messages_no_history():
    chunks = [{"title": "Test", "url": "https://example.com", "text": "Some text."}]
    messages = build_chat_messages("Hello?", chunks, [])
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
