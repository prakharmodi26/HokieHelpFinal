"""FastAPI application for HokieHelp chatbot."""
from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from chatbot.config import ChatbotConfig
from chatbot.guard import PromptRejected, check_prompt
from chatbot.retriever import Retriever
from chatbot.llm import LLMClient
from chatbot.session_store import SessionStore

logger = logging.getLogger(__name__)

app = FastAPI(title="HokieHelp Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# Initialized at startup
retriever: Retriever | None = None
llm_client: LLMClient | None = None
_session_store: SessionStore | None = None

SESSION_COOKIE = "hokiehelp_session"
COOKIE_MAX_AGE = 7 * 24 * 3600  # 1 week


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
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        stripped = v.strip()
        if len(stripped) > 2000:
            raise ValueError("question must not exceed 2000 characters")
        return stripped

    @field_validator("history")
    @classmethod
    def history_not_too_long(cls, v: list) -> list:
        if len(v) > 100:
            raise ValueError("history must not exceed 100 messages")
        return v


@app.on_event("startup")
def startup() -> None:
    global retriever, llm_client, _session_store
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
        hybrid_enabled=cfg.hybrid_enabled,
        keyword_search_limit=cfg.keyword_search_limit,
        rrf_k=cfg.rrf_k,
    )
    llm_client = LLMClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        max_history_messages=cfg.max_history_messages,
    )
    _session_store = SessionStore(
        max_requests=cfg.rate_limit_requests,
        window_seconds=cfg.rate_limit_window_seconds,
    )
    logger.info(
        "Chatbot startup complete — rate_limit=%d/%ds",
        cfg.rate_limit_requests,
        cfg.rate_limit_window_seconds,
    )


def _get_session_id(request: Request) -> str:
    """FastAPI dependency: return existing session cookie or mint a new UUID."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _session_store.get_or_create_session(
        request.cookies.get(SESSION_COOKIE)
    )


def _check_rate_limit(session_id: str = Depends(_get_session_id)) -> str:
    """FastAPI dependency: raise 429 if session has exceeded its hourly limit."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not _session_store.is_allowed(session_id):
        remaining = _session_store.remaining(session_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You may send up to {_session_store._max} messages per hour.",
            headers={"Retry-After": str(_session_store._window), "X-RateLimit-Remaining": str(remaining)},
        )
    return session_id


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


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=True,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    response: Response,
    session_id: str = Depends(_check_rate_limit),
) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        check_prompt(req.question)
    except PromptRejected as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("NEW QUERY: %s", req.question)
    chunks = retriever.search(req.question)
    answer = llm_client.ask(req.question, chunks)
    sources = _dedup_sources(chunks)
    _set_session_cookie(response, session_id)
    return AskResponse(answer=answer, sources=sources)


@app.post("/chat", response_model=AskResponse)
def chat(
    req: ChatRequest,
    response: Response,
    session_id: str = Depends(_check_rate_limit),
) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        check_prompt(req.question)
    except PromptRejected as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("NEW CHAT: %s (history_turns=%d)", req.question, len(req.history))
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    chunks = retriever.search_with_context(req.question, history_dicts)
    answer = llm_client.chat(req.question, chunks, history_dicts)
    sources = _dedup_sources(chunks)
    _set_session_cookie(response, session_id)
    return AskResponse(answer=answer, sources=sources)


@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    request: Request,
    session_id: str = Depends(_check_rate_limit),
) -> StreamingResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        check_prompt(req.question)
    except PromptRejected as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("NEW STREAM CHAT: %s (history_turns=%d)", req.question, len(req.history))
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    chunks = retriever.search_with_context(req.question, history_dicts)
    sources = _dedup_sources(chunks)

    def generate():
        for token in llm_client.chat_stream(req.question, chunks, history_dicts):
            yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
        yield f'data: {json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]})}\n\n'
        yield f'data: {json.dumps({"type": "done"})}\n\n'

    headers = {
        "Set-Cookie": (
            f"{SESSION_COOKIE}={session_id}; Max-Age={COOKIE_MAX_AGE}; "
            "HttpOnly; SameSite=Strict; Secure; Path=/"
        )
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)
