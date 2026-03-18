"""Tests for the LLM client — API calls are mocked."""
from unittest.mock import MagicMock, patch

from chatbot.llm import LLMClient, build_rag_prompt


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
