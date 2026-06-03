"""
LLM factory with Ollama primary (qwen3:8b) and Gemini 2.5 Flash fallback.

Public API
----------
generate_answer(context, question) -> tuple[str, str]
    Returns (answer_text, source) where source is "ollama", "gemini", or "error".
pull_model_background()  Async task: pulls qwen3:8b in background after startup.
PROMPT                   ChatPromptTemplate used by both backends.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


# ── System Prompt ─────────────────────────────────────────────────────────

_SYSTEM = """\
You are a knowledgeable and cautious healthcare information assistant.

STRICT RULES — follow every one without exception:

1. Answer the user's question ONLY using the context provided below.
   Do not use any prior knowledge or information outside this context.

2. If the context does not contain enough information to answer, respond
   with EXACTLY this phrase and nothing else:
   "I could not find this information in the provided documents."

3. Never provide a direct medical diagnosis, recommend specific medications,
   or give personalised medical advice. Always advise the user to consult
   a qualified healthcare professional for personal medical decisions.

4. Cite the source document name when referring to specific information.

5. Be concise, factual, and professional in tone.\

Context:
{context}\
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", "{question}"),
    ]
)


# ── LLM constructors ──────────────────────────────────────────────────────

def _make_ollama_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"{settings.OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        model=settings.LOCAL_MODEL_NAME,
        temperature=0.1,
        timeout=120,
    )


def _make_gemini_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL_NAME,
        temperature=0.1,
        timeout=60,
    )


# ── Main generation entry point ───────────────────────────────────────────

def generate_answer(context: str, question: str) -> tuple[str, str]:
    """
    Generate an answer using Ollama (primary) with silent Gemini fallback.

    Returns
    -------
    (answer: str, source: str)
        source is "ollama", "gemini", or "error"
    """
    chain = PROMPT | _make_ollama_llm()
    try:
        result = chain.invoke({"context": context, "question": question})
        return result.content, "ollama"
    except Exception as ollama_err:
        err_msg = str(ollama_err).lower()
        if "oom" in err_msg or "out of memory" in err_msg:
            logger.warning("Ollama OOM — pivoting to Gemini 2.5 Flash.")
        elif "timeout" in err_msg or "timed out" in err_msg:
            logger.warning("Ollama timeout — pivoting to Gemini 2.5 Flash.")
        else:
            logger.warning("Ollama unavailable (%s) — falling back to Gemini.", ollama_err)

    if not settings.GEMINI_API_KEY:
        logger.error("No GEMINI_API_KEY set; cannot fall back.")
        return "I could not find this information in the provided documents.", "error"

    try:
        chain = PROMPT | _make_gemini_llm()
        result = chain.invoke({"context": context, "question": question})
        return result.content, "gemini"
    except Exception as gemini_err:
        logger.error("Gemini fallback also failed: %s", gemini_err)
        return "I could not find this information in the provided documents.", "error"


# ── Background model pull ─────────────────────────────────────────────────

async def pull_model_background() -> None:
    """
    Pull qwen3:8b from Ollama registry in the background after API startup.
    The API continues serving (using Gemini fallback) during the download.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/pull"
    payload = {"name": settings.LOCAL_MODEL_NAME, "stream": False}
    logger.info("Starting background pull of '%s' …", settings.LOCAL_MODEL_NAME)
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        logger.info("Model '%s' pull complete.", settings.LOCAL_MODEL_NAME)
    except Exception as exc:
        logger.warning("Background model pull failed (non-fatal): %s", exc)