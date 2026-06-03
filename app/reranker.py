"""BGE CrossEncoder reranker singleton with graceful fallback."""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_reranker = None
_reranker_available: bool | None = None  # None = not yet attempted


def _try_init_reranker() -> None:
    """Attempt to load the CrossEncoder reranker model."""
    global _reranker, _reranker_available
    try:
        from sentence_transformers.cross_encoder import CrossEncoder

        logger.info("Loading reranker model '%s' …", settings.RERANKER_MODEL_NAME)
        _reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        _reranker_available = True
        logger.info("Reranker model loaded.")
    except Exception as exc:
        logger.warning(
            "Reranker init failed (%s) — will use hybrid-score ranking as fallback.",
            exc,
        )
        _reranker_available = False


def rerank(query: str, docs: List[dict], top_k: int) -> List[dict]:
    """
    Rerank *docs* against *query* using cross-encoder scores.

    Primary : BGE CrossEncoder scoring.
    Fallback: Sort by Weaviate hybrid_score (no reranking).

    Each doc dict must contain at least a 'content' key.
    Returns the top_k docs sorted by descending score,
    with an added 'rerank_score' field when using the cross-encoder.
    """
    if not docs:
        return []

    # Lazy init — first call only
    if _reranker_available is None:
        _try_init_reranker()

    if _reranker_available and _reranker is not None:
        try:
            pairs = [[query, doc["content"]] for doc in docs]
            scores: np.ndarray = _reranker.predict(pairs)
            for doc, score in zip(docs, scores):
                doc["rerank_score"] = float(score)
            return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]
        except Exception as exc:
            logger.warning(
                "Reranker prediction failed (%s) — falling back to hybrid scores.", exc
            )

    # Fallback: sort by the hybrid search score Weaviate already provided
    return sorted(docs, key=lambda d: d.get("hybrid_score", 0.0), reverse=True)[:top_k]
