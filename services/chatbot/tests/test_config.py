import os

import pytest

from chatbot.config import ChatbotConfig


def test_defaults_match_live_arc_bge_m3_stack(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    cfg = ChatbotConfig.from_env()

    assert cfg.llm_api_key == "sk-test-key"
    assert cfg.llm_base_url == "https://llm-api.arc.vt.edu/api/v1"
    assert cfg.llm_model == "GLM-5.3"
    assert cfg.rewriter_model == "DeepSeek-V4.1-Flash"
    assert cfg.qdrant_host == "qdrant"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_collection == "hokiehelp-test"
    assert cfg.embedding_model == "BAAI/bge-m3"
    assert cfg.embeddings_url == "http://embeddings-service"
    assert cfg.top_k == 12
    assert cfg.candidate_k == 30
    assert cfg.min_score == 0.3
    assert cfg.hybrid_enabled is True
    assert cfg.program_candidates_enabled is True
    assert cfg.max_history_messages == 20
    assert cfg.max_history_chars == 12000
    assert cfg.max_context_chars == 24000
    assert cfg.max_chunk_chars == 4000
    assert cfg.llm_timeout == 60.0
    assert cfg.rewrite_timeout == 20.0
    assert cfg.dependency_timeout == 20.0
    assert cfg.llm_max_tokens == 2048
    assert cfg.rewrite_max_tokens == 1024


def test_from_env_overrides_types_and_bool(monkeypatch):
    env = {
        "LLM_API_KEY": "sk-override",
        "LLM_BASE_URL": "https://arc.example/api/v1",
        "LLM_MODEL": "answer-model",
        "REWRITER_MODEL": "planner-model",
        "QDRANT_HOST": "qdrant.example",
        "QDRANT_PORT": "6334",
        "QDRANT_COLLECTION": "prod-collection",
        "EMBEDDINGS_URL": "http://embeddings",
        "TOP_K": "7",
        "CANDIDATE_K": "30",
        "MIN_SCORE": "0.42",
        "MAX_HISTORY_MESSAGES": "9",
        "MAX_HISTORY_CHARS": "1000",
        "MAX_CONTEXT_CHARS": "9000",
        "MAX_CHUNK_CHARS": "3000",
        "HYBRID_ENABLED": "false",
        "RATE_LIMIT_REQUESTS": "50",
        "RATE_LIMIT_WINDOW_SECONDS": "1800",
        "LLM_TIMEOUT": "31.5",
        "REWRITE_TIMEOUT": "6.5",
        "DEPENDENCY_TIMEOUT": "7.5",
        "LLM_MAX_TOKENS": "900",
        "REWRITE_MAX_TOKENS": "300",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    cfg = ChatbotConfig.from_env()

    assert cfg.llm_model == "answer-model"
    assert cfg.rewriter_model == "planner-model"
    assert cfg.qdrant_port == 6334
    assert cfg.top_k == 7
    assert cfg.candidate_k == 30
    assert cfg.min_score == 0.42
    assert cfg.hybrid_enabled is False
    monkeypatch.setenv("PROGRAM_CANDIDATES_ENABLED", "false")
    assert ChatbotConfig.from_env().program_candidates_enabled is False
    assert cfg.llm_timeout == 31.5
    assert cfg.rewrite_max_tokens == 300


def test_missing_and_blank_api_key_are_rejected(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(KeyError):
        ChatbotConfig.from_env()

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        ChatbotConfig(llm_api_key="  ")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"top_k": 0}, "top_k must be positive"),
        ({"candidate_k": 0}, "candidate_k must be positive"),
        ({"top_k": 31, "candidate_k": 30}, "top_k must not exceed candidate_k"),
        ({"min_score": -0.01}, "min_score must be in"),
        ({"min_score": 1.01}, "min_score must be in"),
        ({"embedding_model": "BAAI/bge-large-en-v1.5"}, "BGE-M3"),
        ({"dependency_timeout": 0}, "dependency_timeout must be positive"),
        ({"llm_timeout": float("nan")}, "positive and finite"),
        ({"rewrite_timeout": float("inf")}, "positive and finite"),
    ],
)
def test_constructor_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChatbotConfig(llm_api_key="sk", **kwargs)


def test_bad_boolean_env_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk")
    monkeypatch.setenv("HYBRID_ENABLED", "yes")

    with pytest.raises(ValueError, match="HYBRID_ENABLED"):
        ChatbotConfig.from_env()


def test_secret_not_in_repr():
    cfg = ChatbotConfig(llm_api_key="super-secret-token")

    rendered = repr(cfg)
    assert "super-secret-token" not in rendered
    assert "llm_api_key" not in rendered


def test_from_env_reads_current_environment_without_pi_files(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("CANDIDATE_K", "3")

    cfg = ChatbotConfig.from_env()

    assert cfg.llm_api_key == "sk-env"
    assert cfg.top_k == 3
    assert cfg.candidate_k == 3
