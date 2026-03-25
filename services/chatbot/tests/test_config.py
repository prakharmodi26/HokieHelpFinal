import os
from unittest.mock import patch
import pytest

from chatbot.config import ChatbotConfig


def test_from_env_defaults():
    env = {"LLM_API_KEY": "sk-test-key"}
    with patch.dict(os.environ, env, clear=False):
        cfg = ChatbotConfig.from_env()
    assert cfg.llm_api_key == "sk-test-key"
    assert cfg.llm_base_url == "https://llm-api.arc.vt.edu/api/v1"
    assert cfg.llm_model == "gpt-oss-120b"
    assert cfg.qdrant_host == "qdrant"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp_chunks"
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.top_k == 5
    assert cfg.max_history_messages == 20


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
