"""
FastAPI application — Healthcare AI Assistant v2.

Endpoints
---------
GET  /health  Returns Weaviate + Ollama readiness, model names.
POST /ingest  Ingest .txt and CSV files into Weaviate (idempotent).
POST /ask     Answer a question via the 3-way router + RAG pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import handle_query
from app.config import settings
from app.llm import pull_model_background
from app.rag import ingest_documents
from app.weaviate_client import ensure_collections, get_weaviate_client

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init Weaviate schema + kick off model pull. Shutdown: close client."""
    client = get_weaviate_client()
    ensure_collections(client)
    logger.info("Weaviate collections ready.")
    asyncio.create_task(pull_model_background())
    yield
    try:
        get_weaviate_client().close()
    except Exception:
        pass


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare AI Assistant",
    description=(
        "Production RAG service with Weaviate hybrid search, BGE-M3 embeddings, "
        "BGE reranker, qwen3:8b via Ollama (Gemini 2.5 Flash fallback), "
        "and a 3-way agentic router."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")


class SourceItem(BaseModel):
    document: str = Field(..., description="Source file name or drug name.")
    chunk: str = Field(..., description="Relevant text chunk.")


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: str
    route: str
    retrieval_time_ms: float
    generation_time_ms: float


class IngestResponse(BaseModel):
    status: str
    txt_chunks: int
    csv_rows: int
    duration_ms: float


class HealthResponse(BaseModel):
    status: str
    weaviate_ready: bool
    ollama_ready: bool
    model_name: str
    embed_model: str


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check() -> HealthResponse:
    """Return service health and downstream dependency readiness."""
    # Check Weaviate
    weaviate_ready = False
    try:
        client = get_weaviate_client()
        weaviate_ready = client.is_ready()
    except Exception:
        pass

    # Check Ollama
    ollama_ready = False
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            resp = await http.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_ready = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        weaviate_ready=weaviate_ready,
        ollama_ready=ollama_ready,
        model_name=settings.LOCAL_MODEL_NAME,
        embed_model=settings.EMBED_MODEL_NAME,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest() -> IngestResponse:
    """Ingest all .txt and CSV data files into Weaviate (idempotent)."""
    t0 = time.perf_counter()
    try:
        result = ingest_documents()
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Ingestion complete. txt_chunks=%d, csv_rows=%d, duration=%.1f ms",
            result["txt_chunks"],
            result["csv_rows"],
            duration_ms,
        )
        return IngestResponse(
            status="success",
            txt_chunks=result["txt_chunks"],
            csv_rows=result["csv_rows"],
            duration_ms=duration_ms,
        )
    except Exception as exc:
        logger.exception("Ingestion failed.")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    """
    Answer a healthcare question using the 3-way agentic router.

    This endpoint NEVER returns HTTP 500 — all exceptions are caught and
    mapped to a clean AskResponse with an error indication.
    """
    try:
        result = handle_query(request.question)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            confidence=result["confidence"],
            route=result["route"],
            retrieval_time_ms=result["retrieval_time_ms"],
            generation_time_ms=result["generation_time_ms"],
        )
    except Exception as exc:
        logger.exception("Unhandled error in /ask endpoint.")
        return AskResponse(
            answer="I could not find this information in the provided documents.",
            sources=[],
            confidence="low",
            route="error",
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
        )
