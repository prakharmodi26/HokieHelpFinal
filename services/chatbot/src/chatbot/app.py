"""FastAPI application for HokieHelp chatbot."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from chatbot.config import ChatbotConfig
from chatbot.retriever import Retriever
from chatbot.llm import LLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="HokieHelp Chatbot", version="0.1.0")

# Initialized at startup
retriever: Retriever | None = None
llm_client: LLMClient | None = None


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class Source(BaseModel):
    title: str
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @field_validator("question")
    @classmethod
    def question_not_too_long(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError("question must not exceed 2000 characters")
        return v

    @field_validator("history")
    @classmethod
    def history_not_too_long(cls, v: list) -> list:
        if len(v) > 100:
            raise ValueError("history must not exceed 100 messages")
        return v


@app.on_event("startup")
def startup() -> None:
    global retriever, llm_client
    if retriever is not None:
        return
    cfg = ChatbotConfig.from_env()
    retriever = Retriever(
        embedding_model=cfg.embedding_model,
        qdrant_host=cfg.qdrant_host,
        qdrant_port=cfg.qdrant_port,
        collection=cfg.qdrant_collection,
        top_k=cfg.top_k,
        min_score=cfg.min_score,
    )
    llm_client = LLMClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    logger.info("Chatbot startup complete")


def _dedup_sources(chunks: list[dict]) -> list[Source]:
    """Deduplicate sources by URL, keeping the highest-scoring chunk per URL."""
    seen: dict[str, Source] = {}
    for c in chunks:
        url = c.get("url", "")
        score = c["score"]
        if url not in seen or score > seen[url].score:
            seen[url] = Source(
                title=c.get("title", ""),
                url=url,
                score=score,
            )
    return sorted(seen.values(), key=lambda s: s.score, reverse=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("=" * 70)
    logger.info("NEW QUERY: %s", req.question)
    logger.info("=" * 70)

    chunks = retriever.search(req.question)

    non_empty = sum(1 for c in chunks if c.get("text"))
    logger.info("CHUNKS with text: %d / %d", non_empty, len(chunks))

    answer = llm_client.ask(req.question, chunks)

    sources = _dedup_sources(chunks)

    logger.info("RESPONSE sent — answer_len=%d  sources=%d", len(answer), len(sources))
    return AskResponse(answer=answer, sources=sources)


@app.post("/chat", response_model=AskResponse)
def chat(req: ChatRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("=" * 70)
    logger.info("NEW CHAT: %s (history_turns=%d)", req.question, len(req.history))
    logger.info("=" * 70)

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # Rewrite follow-up questions into standalone search queries
    search_query = llm_client.rewrite_query(req.question, history_dicts)
    chunks = retriever.search(search_query)

    answer = llm_client.chat(req.question, chunks, history_dicts)

    sources = _dedup_sources(chunks)

    logger.info("CHAT RESPONSE sent — answer_len=%d  sources=%d", len(answer), len(sources))
    return AskResponse(answer=answer, sources=sources)
