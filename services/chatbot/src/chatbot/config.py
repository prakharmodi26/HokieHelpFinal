"""Chatbot service configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatbotConfig:
    """Immutable configuration for the chatbot service."""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    rewriter_model: str
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

    @classmethod
    def from_env(cls) -> ChatbotConfig:
        return cls(
            llm_api_key=os.environ["LLM_API_KEY"],
            llm_base_url=os.environ.get("LLM_BASE_URL", "http://ollama-cluster-ip:11434/v1"),
            llm_model=os.environ.get("LLM_MODEL", "qwen2.5:14b"),
            rewriter_model=os.environ.get("REWRITER_MODEL", os.environ.get("LLM_MODEL", "qwen2.5:14b")),
            qdrant_host=os.environ.get("QDRANT_HOST", "qdrant"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "hokiehelp_chunks"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            top_k=int(os.environ.get("TOP_K", "5")),
            min_score=float(os.environ.get("MIN_SCORE", "0.53")),
            max_history_messages=int(os.environ.get("MAX_HISTORY_MESSAGES", "20")),
            hybrid_enabled=os.environ.get("HYBRID_ENABLED", "true").lower() == "true",
            keyword_search_limit=int(os.environ.get("KEYWORD_SEARCH_LIMIT", "10")),
            rrf_k=int(os.environ.get("RRF_K", "60")),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "3600")),
            follow_up_keywords=tuple(
                kw.strip()
                for kw in os.environ.get(
                    "FOLLOW_UP_KEYWORDS",
                    "he,she,they,them,their,his,her,its,it,that,this,those,these,"
                    "the same,above,previous,mentioned,what about,tell me more,"
                    "elaborate,expand,who did,more about,also,and what",
                ).split(",")
                if kw.strip()
            ),
        )
