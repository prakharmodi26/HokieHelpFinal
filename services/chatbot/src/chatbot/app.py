"""FastAPI application for HokieHelp chatbot."""
from __future__ import annotations

import logging

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
    )
    llm_client = LLMClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    logger.info("Chatbot startup complete")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    chunks = retriever.search(req.question)
    answer = llm_client.ask(req.question, chunks)

    sources = [
        Source(title=c.get("title", ""), url=c.get("url", ""), score=c["score"])
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
