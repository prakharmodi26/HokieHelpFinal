"""Environment configuration; secrets are supplied by the runtime, not pi files."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatbotConfig:
    llm_api_key: str = field(repr=False)
    llm_base_url: str = "https://llm-api.arc.vt.edu/api/v1"
    llm_model: str = "GLM-5.3"
    rewriter_model: str = "DeepSeek-V4.1-Flash"
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "hokiehelp-test"
    embedding_model: str = "BAAI/bge-m3"
    embeddings_url: str = "http://embeddings-service"
    top_k: int = 12
    candidate_k: int = 30
    min_score: float = 0.3  # existing cross-encoder score floor, NOT cosine
    max_history_messages: int = 20
    max_history_chars: int = 12000
    max_context_chars: int = 24000
    max_chunk_chars: int = 4000
    hybrid_enabled: bool = True
    program_candidates_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 3600
    llm_timeout: float = 60.0
    rewrite_timeout: float = 20.0
    dependency_timeout: float = 20.0
    llm_max_tokens: int = 2048
    rewrite_max_tokens: int = 1024

    def __post_init__(self):
        positive = ("qdrant_port", "top_k", "candidate_k", "max_history_messages",
                    "max_history_chars", "max_context_chars", "max_chunk_chars",
                    "rate_limit_requests", "rate_limit_window_seconds", "llm_timeout",
                    "rewrite_timeout", "dependency_timeout", "llm_max_tokens", "rewrite_max_tokens")
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be in [0, 1]")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        if not self.llm_api_key.strip():
            raise ValueError("LLM_API_KEY is required")
        if self.embedding_model != "BAAI/bge-m3":
            raise ValueError("This retriever targets the existing BGE-M3 index; model changes require a compatible index")

    @classmethod
    def from_env(cls) -> ChatbotConfig:
        values = {"llm_api_key": os.environ["LLM_API_KEY"]}
        for name, definition in cls.__dataclass_fields__.items():
            if name == "llm_api_key" or name.upper() not in os.environ:
                continue
            raw = os.environ[name.upper()]
            default = definition.default
            if isinstance(default, bool):
                if raw.lower() not in {"true", "false"}:
                    raise ValueError(f"{name.upper()} must be true or false")
                values[name] = raw.lower() == "true"
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                values[name] = float(raw)
            else:
                values[name] = raw
        return cls(**values)
