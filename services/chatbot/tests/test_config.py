import os
from unittest.mock import patch
import pytest

from chatbot.config import ChatbotConfig


def test_from_env_defaults():
    env = {"LLM_API_KEY": "sk-test-key"}
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.llm_api_key == "sk-test-key"
    assert cfg.llm_base_url == "http://ollama-cluster-ip:11434/v1"
    assert cfg.llm_model == "qwen2.5:14b"
    assert cfg.qdrant_host == "qdrant"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp_chunks"
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.top_k == 5
    assert cfg.max_history_messages == 20
    assert "their" in cfg.follow_up_keywords
    assert "what about" in cfg.follow_up_keywords


def test_from_env_overrides():
    env = {
        "LLM_API_KEY": "sk-override",
        "LLM_MODEL": "custom-model",
        "TOP_K": "10",
        "QDRANT_HOST": "custom-host",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.llm_model == "custom-model"
    assert cfg.top_k == 10
    assert cfg.qdrant_host == "custom-host"


def test_from_env_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            ChatbotConfig.from_env()


def test_rate_limit_defaults():
    """Config should default to 100 requests / 3600 seconds when env vars are absent."""
    env = {
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "http://localhost",
        "LLM_MODEL": "test-model",
    }
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("RATE_LIMIT_REQUESTS", None)
        os.environ.pop("RATE_LIMIT_WINDOW_SECONDS", None)
        cfg = ChatbotConfig.from_env()
    assert cfg.rate_limit_requests == 100
    assert cfg.rate_limit_window_seconds == 3600


def test_rate_limit_from_env():
    """Config should honour RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW_SECONDS."""
    env = {
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "http://localhost",
        "LLM_MODEL": "test-model",
        "RATE_LIMIT_REQUESTS": "50",
        "RATE_LIMIT_WINDOW_SECONDS": "1800",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.rate_limit_requests == 50
    assert cfg.rate_limit_window_seconds == 1800
