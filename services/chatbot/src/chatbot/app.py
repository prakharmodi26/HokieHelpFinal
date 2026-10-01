"""FastAPI endpoints sharing one planning, retrieval, and evidence pipeline."""
from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from chatbot import logbuffer
from chatbot.config import ChatbotConfig
from chatbot.evidence import NO_EVIDENCE_ANSWER, finalize_answer, select_evidence
from chatbot.guard import PromptRejected, check_prompt
from chatbot.llm import LLMClient
from chatbot.retriever import Retriever
from chatbot.session_store import SessionStore

logbuffer.install(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="HokieHelp Chatbot", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=["POST", "GET"],
                   allow_headers=["Content-Type"], allow_credentials=False)
retriever: Retriever | None = None
llm_client: LLMClient | None = None
_session_store: SessionStore | None = None
_config: ChatbotConfig | None = None
SESSION_COOKIE = "hokiehelp_session"
COOKIE_MAX_AGE = 7 * 24 * 3600


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class Source(BaseModel):
    id: int
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        return value


class ChatRequest(AskRequest):
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)

    @field_validator("history")
    @classmethod
    def bound_history_size(cls, value: list[HistoryMessage]) -> list[HistoryMessage]:
        if sum(len(message.content) for message in value) > 64000:
            raise ValueError("history must not exceed 64000 characters")
        return value


@app.on_event("startup")
def startup() -> None:
    global retriever, llm_client, _session_store, _config
    if retriever is not None:
        return
    cfg = _config = ChatbotConfig.from_env()
    retriever = Retriever(
        embedding_model=cfg.embedding_model, qdrant_host=cfg.qdrant_host,
        qdrant_port=cfg.qdrant_port, collection=cfg.qdrant_collection,
        top_k=cfg.top_k, candidate_k=cfg.candidate_k, min_score=cfg.min_score,
        hybrid_enabled=cfg.hybrid_enabled, embeddings_url=cfg.embeddings_url,
        timeout=cfg.dependency_timeout, program_candidates_enabled=cfg.program_candidates_enabled,
    )
    llm_client = LLMClient(
        api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model,
        rewriter_model=cfg.rewriter_model, max_history_messages=cfg.max_history_messages,
        timeout=cfg.llm_timeout, rewrite_timeout=cfg.rewrite_timeout,
        max_tokens=cfg.llm_max_tokens, rewrite_max_tokens=cfg.rewrite_max_tokens,
        max_history_chars=cfg.max_history_chars,
    )
    _session_store = SessionStore(max_requests=cfg.rate_limit_requests,
                                  window_seconds=cfg.rate_limit_window_seconds)


@app.on_event("shutdown")
def shutdown() -> None:
    if llm_client is not None:
        llm_client.close()
    if retriever is not None:
        retriever.close()


def _get_session_id(request: Request) -> str:
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _session_store.get_or_create_session(request.cookies.get(SESSION_COOKIE))


def _check_rate_limit(session_id: str = Depends(_get_session_id)) -> str:
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not _session_store.is_allowed(session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You may send up to {_session_store._max} messages per hour.",
            headers={"Retry-After": str(_session_store._window),
                     "X-RateLimit-Remaining": str(_session_store.remaining(session_id))},
        )
    return session_id


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(key=SESSION_COOKIE, value=session_id, max_age=COOKIE_MAX_AGE,
                        httponly=True, samesite="strict", secure=True)


def _prepare(question: str, history: list[dict]) -> tuple[list[dict], str | None]:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        check_prompt(question)
    except PromptRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    plan = llm_client.plan_query(question, history)
    logger.info("PLAN intent=%s query_count=%d", plan.intent, len(plan.queries))
    if plan.intent == "clarify":
        return [], plan.clarification
    if plan.intent == "conversation":
        return [], "Hi! I can help with questions about Virginia Tech. What would you like to know?"
    try:
        chunks = retriever.search_many(plan.queries, original_query=question)
    except Exception as exc:
        logger.error("Retrieval unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Search is temporarily unavailable. Please try again.") from exc
    selected = select_evidence(
        chunks, max_chunks=_config.top_k if _config else 12,
        max_chars=_config.max_context_chars if _config else 24000,
        max_chunk_chars=_config.max_chunk_chars if _config else 4000,
    )
    return selected, None if selected else NO_EVIDENCE_ANSWER


def _answer(question: str, history: list[dict], response: Response, session_id: str) -> AskResponse:
    chunks, direct_answer = _prepare(question, history)
    if direct_answer is not None:
        answer, sources = direct_answer, []
    else:
        try:
            answer, sources = finalize_answer(llm_client.chat(question, chunks, history), chunks)
        except Exception as exc:
            logger.error("Answer unavailable (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="The language model is temporarily unavailable. Please try again.") from exc
    _set_session_cookie(response, session_id)
    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if retriever is None or llm_client is None or _session_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {"status": "ready"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, response: Response,
        session_id: str = Depends(_check_rate_limit)) -> AskResponse:
    return _answer(req.question, [], response, session_id)


@app.post("/chat", response_model=AskResponse)
def chat(req: ChatRequest, response: Response,
         session_id: str = Depends(_check_rate_limit)) -> AskResponse:
    return _answer(req.question, [m.model_dump() for m in req.history], response, session_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, session_id: str = Depends(_check_rate_limit)) -> StreamingResponse:
    history = [m.model_dump() for m in req.history]
    chunks, direct_answer = _prepare(req.question, history)

    def event(kind: str, **data) -> str:
        return f"data: {json.dumps({'type': kind, **data})}\n\n"

    def generate():
        try:
            if direct_answer is not None:
                answer, sources = direct_answer, []
                yield event("token", content=answer)
            else:
                parts: list[str] = []
                for token in llm_client.chat_stream(req.question, chunks, history):
                    parts.append(token)
                    yield event("token", content=token)
                answer, sources = finalize_answer("".join(parts), chunks)
            # Tokens are provisional; final answer is authoritative after reference validation.
            yield event("answer", content=answer)
            yield event("sources", sources=sources)
            yield event("done")
        except Exception as exc:
            logger.error("Stream failed (%s)", type(exc).__name__)
            yield event("error", content="The response was interrupted. Please try again.")

    response = StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    _set_session_cookie(response, session_id)
    return response
