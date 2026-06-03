"""
BGE-M3 embedding singleton with Gemini cloud fallback.

Public API
----------
embed_texts(texts)           → List[List[float]]  Batch-encode texts.
embed_query(query)           → List[float]         Encode a single query.
get_embedding_dimension()    → int                 Active model vector dimension.
get_active_model_name()      → str                 Active model identifier.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────

_local_model: Optional[object] = None       # SentenceTransformer (if loaded)
_cloud_embedder: Optional[object] = None    # OpenAIEmbeddings  (if cloud active)
_active_model_name: str = ""
_active_dimension: int = 1024
_using_cloud: bool = False
_initialized: bool = False


# ── Lazy initialization ──────────────────────────────────────────────────

def _try_init_local() -> bool:
    """Attempt to load the local BGE-M3 model via sentence-transformers."""
    global _local_model, _active_model_name, _active_dimension, _using_cloud
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model '%s' …", settings.EMBED_MODEL_NAME)
        model = SentenceTransformer(settings.EMBED_MODEL_NAME)

        # Use the non-deprecated method name (sentence-transformers >= 3.3)
        _active_dimension = model.get_embedding_dimension()
        _local_model = model
        _active_model_name = settings.EMBED_MODEL_NAME
        _using_cloud = False
        logger.info(
            "Local embedding model ready — %s (dim=%d).",
            _active_model_name,
            _active_dimension,
        )
        return True
    except Exception as exc:
        logger.warning("Local embedding model failed to load: %s", exc)
        return False


def _try_init_cloud() -> bool:
    """Set up Gemini cloud embeddings via the OpenAI-compatible endpoint."""
    global _cloud_embedder, _active_model_name, _active_dimension, _using_cloud
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — cloud embedding fallback unavailable.")
        return False
    try:
        from langchain_openai import OpenAIEmbeddings

        _cloud_embedder = OpenAIEmbeddings(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_EMBED_MODEL_NAME,
        )
        _active_model_name = settings.GEMINI_EMBED_MODEL_NAME
        _active_dimension = 768  # text-embedding-004 output dimension
        _using_cloud = True
        logger.info(
            "Cloud embedding fallback ready — %s (dim=%d).",
            _active_model_name,
            _active_dimension,
        )
        return True
    except Exception as exc:
        logger.error("Cloud embedding init also failed: %s", exc)
        return False


def _ensure_init() -> None:
    """Lazy initialization — try local first, fall back to cloud."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    if not _try_init_local():
        if not _try_init_cloud():
            raise RuntimeError(
                "No embedding model available. "
                "Both local (sentence-transformers) and cloud (Gemini) initialisation failed."
            )


# ── Public API ────────────────────────────────────────────────────────────

def get_embedding_dimension() -> int:
    """Return the vector dimension of the currently active embedding model."""
    _ensure_init()
    return _active_dimension


def get_active_model_name() -> str:
    """Return the identifier of the currently active embedding model."""
    _ensure_init()
    return _active_model_name


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encode a batch of texts and return normalised float vectors."""
    _ensure_init()
    if _using_cloud:
        logger.debug("Encoding %d text(s) via Gemini cloud embeddings.", len(texts))
        return _cloud_embedder.embed_documents(texts)

    vectors = _local_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Encode a single query string and return a normalised float vector."""
    _ensure_init()
    if _using_cloud:
        return _cloud_embedder.embed_query(query)
    return embed_texts([query])[0]
