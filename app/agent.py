"""
Agentic query router — 3 routes with timing metrics.

Routes:
  1. appointment   — scheduling/booking keywords → mock availability tool
  2. drug_recall   — FDA/recall keywords → DrugRecalls Weaviate collection
  3. healthcare_rag (default) — HealthcareText Weaviate collection

Public API
----------
handle_query(question: str) -> dict
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.llm import generate_answer
from app.rag import search_drug_recalls, search_healthcare

logger = logging.getLogger(__name__)

# ── Keyword sets ──────────────────────────────────────────────────────────

_APPOINTMENT_KW = frozenset(
    {
        "appointment",
        "book",
        "schedule",
        "slot",
        "visit",
        "reschedule",
        "cancel",
        "availability",
    }
)

_DRUG_RECALL_KW = frozenset(
    {
        "fda",
        "recall",
        "recalled",
        "manufacturer",
        "medication recall",
        "drug recall",
        "drug alert",
    }
)

# ── Mock appointment data ─────────────────────────────────────────────────

_MOCK_SLOTS = {
    "Dr. Sarah Mitchell (General Practice)": [
        "Monday 9:00 AM",
        "Wednesday 2:00 PM",
        "Friday 10:30 AM",
    ],
    "Dr. James Patel (Cardiology)": [
        "Tuesday 11:00 AM",
        "Thursday 3:00 PM",
    ],
    "Dr. Anna Nguyen (Internal Medicine)": [
        "Monday 1:00 PM",
        "Wednesday 4:00 PM",
        "Thursday 9:30 AM",
    ],
}


def check_available_slots() -> str:
    lines = ["Available appointment slots this week:\n"]
    for doctor, slots in _MOCK_SLOTS.items():
        lines.append(f"  {doctor}:")
        for s in slots:
            lines.append(f"    • {s}")
    lines.append(
        "\nTo book an appointment, please contact scheduling at (555) 800-1234 "
        "or visit portal.northridgemedical.example."
    )
    return "\n".join(lines)


# ── Routing helpers ───────────────────────────────────────────────────────

def _route(question: str) -> str:
    lower = question.lower()
    if any(kw in lower for kw in _APPOINTMENT_KW):
        return "appointment"
    if any(kw in lower for kw in _DRUG_RECALL_KW):
        return "drug_recall"
    return "healthcare_rag"


def _score_confidence(docs: list[dict]) -> str:
    n = len(docs)
    if n >= 3:
        return "high"
    if n >= 1:
        return "medium"
    return "low"


def _format_sources(docs: list[dict]) -> list[dict[str, str]]:
    results = []
    for doc in docs:
        results.append(
            {
                "document": doc.get("source") or doc.get("drug_name", "unknown"),
                "chunk": doc.get("content", ""),
            }
        )
    return results


def _build_context(docs: list[dict]) -> str:
    return "\n\n".join(doc.get("content", "") for doc in docs)


# ── Public entry point ────────────────────────────────────────────────────

def handle_query(question: str) -> dict[str, Any]:
    """
    Route the question and return a response dict including timing metrics.

    This function NEVER raises — all exceptions are caught internally and
    mapped to a safe, user-friendly answer.

    Returns
    -------
    dict with keys: answer, sources, confidence, route,
                    retrieval_time_ms, generation_time_ms
    """
    route = _route(question)
    logger.info("Route: %s | question=%r", route, question)

    # ── Appointment mock tool ─────────────────────────────────────────────
    if route == "appointment":
        t0 = time.perf_counter()
        answer = check_available_slots()
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "sources": [],
            "confidence": "N/A",
            "route": "appointment",
            "retrieval_time_ms": 0.0,
            "generation_time_ms": round(elapsed, 2),
        }

    # ── Retrieval phase ───────────────────────────────────────────────────
    docs: list[dict] = []
    retrieval_time_ms = 0.0
    try:
        t_ret_start = time.perf_counter()
        if route == "drug_recall":
            docs = search_drug_recalls(question)
        else:
            docs = search_healthcare(question)
        retrieval_time_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)
        logger.debug("Retrieved %d doc(s) in %.1f ms.", len(docs), retrieval_time_ms)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)

    # ── Generation phase ──────────────────────────────────────────────────
    generation_time_ms = 0.0

    if not docs:
        # No context — short-circuit without calling the LLM
        answer = (
            "I could not find relevant information. "
            "Please ensure documents have been ingested via the /ingest endpoint."
        )
    else:
        try:
            context = _build_context(docs)
            t_gen_start = time.perf_counter()
            answer, _llm_source = generate_answer(context, question)
            generation_time_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            answer = "I could not find this information in the provided documents."

    return {
        "answer": answer,
        "sources": _format_sources(docs),
        "confidence": _score_confidence(docs),
        "route": route,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
    }
