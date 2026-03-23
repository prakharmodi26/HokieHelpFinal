"""Tests for the LLM client — API calls are mocked."""
from unittest.mock import MagicMock, patch

from chatbot.llm import LLMClient, build_rag_prompt, build_chat_prompt, SYSTEM_PROMPT


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


def test_build_chat_prompt_with_history_and_chunks():
    history = [
        {"role": "user", "content": "Who is the department head?"},
        {"role": "assistant", "content": "Dr. Smith is the department head."},
    ]
    chunks = [{"title": "Faculty", "url": "https://example.com", "text": "Dr. Jones teaches CS."}]
    prompt = build_chat_prompt("What does Dr. Jones teach?", chunks, history)
    assert "CHAT HISTORY:" in prompt
    assert "Who is the department head?" in prompt
    assert "Dr. Smith is the department head." in prompt
    assert "RETRIEVED CONTEXT:" in prompt
    assert "Dr. Jones teaches CS." in prompt
    assert "What does Dr. Jones teach?" in prompt


def test_build_chat_prompt_no_history():
    chunks = [{"title": "Test", "url": "https://example.com", "text": "Some text."}]
    prompt = build_chat_prompt("Hello?", chunks, [])
    assert "CHAT HISTORY:" not in prompt
    assert "RETRIEVED CONTEXT:" in prompt
    assert "Some text." in prompt
    assert "Hello?" in prompt


def test_build_chat_prompt_history_only_no_chunks():
    history = [
        {"role": "user", "content": "Who is Denis Gracanin?"},
        {"role": "assistant", "content": "Denis Gracanin is an associate professor."},
    ]
    prompt = build_chat_prompt("Who did I just ask about?", [], history)
    assert "CHAT HISTORY:" in prompt
    assert "Denis Gracanin" in prompt
    assert "RETRIEVED CONTEXT:" not in prompt
    assert "Who did I just ask about?" in prompt


def test_llm_client_chat():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "You asked about Denis Gracanin."
    mock_client.chat.completions.create.return_value = mock_response

    with patch("chatbot.llm.OpenAI", return_value=mock_client):
        client = LLMClient(api_key="sk-test", base_url="https://test.com/api/v1", model="gpt-oss-120b")
        history = [{"role": "user", "content": "Who is Denis?"}, {"role": "assistant", "content": "He is a prof."}]
        answer = client.chat("Who did I ask about?", [], history)

    assert answer == "You asked about Denis Gracanin."
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "CHAT HISTORY:" in messages[1]["content"]
