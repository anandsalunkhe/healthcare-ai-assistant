"""Weaviate v4 client singleton and collection schema management."""
from __future__ import annotations

import logging
import time
import warnings

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from app.config import settings

# ── Suppress known Weaviate deprecation warnings ──────────────────────────
# Dep024: vectorizer_config parameter still functions correctly in
# weaviate-client >=4.9; the named-vectors API is not required for
# single-vector collections.  Warning is cosmetic, not a bug.
warnings.filterwarnings("ignore", message=".*Dep024.*", category=DeprecationWarning)

logger = logging.getLogger(__name__)

HEALTHCARE_TEXT = "HealthcareText"
DRUG_RECALLS = "DrugRecalls"

_client: weaviate.WeaviateClient | None = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return (and cache) a Weaviate v4 client with exponential-backoff retry."""
    global _client
    if _client is not None:
        try:
            if _client.is_connected():
                return _client
        except Exception:
            _client = None  # stale reference — reconnect

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            _client = weaviate.connect_to_custom(
                http_host=settings.WEAVIATE_HOST,
                http_port=settings.WEAVIATE_HTTP_PORT,
                http_secure=False,
                grpc_host=settings.WEAVIATE_HOST,
                grpc_port=settings.WEAVIATE_GRPC_PORT,
                grpc_secure=False,
            )
            logger.info(
                "Connected to Weaviate at %s:%s",
                settings.WEAVIATE_HOST,
                settings.WEAVIATE_HTTP_PORT,
            )
            return _client
        except Exception as exc:
            if attempt < max_retries:
                delay = min(2 ** attempt, 16)
                logger.warning(
                    "Weaviate connection attempt %d/%d failed (%s) — retrying in %ds.",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("Weaviate connection failed after %d attempts.", max_retries)
                raise


def ensure_collections(client: weaviate.WeaviateClient) -> None:
    """Create Weaviate collections if they do not already exist."""
    _ensure_healthcare_text(client)
    _ensure_drug_recalls(client)


def _ensure_healthcare_text(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(HEALTHCARE_TEXT):
        logger.info("Collection '%s' already exists — skipping creation.", HEALTHCARE_TEXT)
        return

    client.collections.create(
        name=HEALTHCARE_TEXT,
        description="Chunked healthcare policy and clinical documents with BGE-M3 vectors.",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.INT),
        ],
    )
    logger.info("Created collection '%s'.", HEALTHCARE_TEXT)


def _ensure_drug_recalls(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(DRUG_RECALLS):
        logger.info("Collection '%s' already exists — skipping creation.", DRUG_RECALLS)
        return

    client.collections.create(
        name=DRUG_RECALLS,
        description="FDA drug recall records with BGE-M3 vectors.",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="drug_name", data_type=DataType.TEXT),
            Property(name="manufacturer", data_type=DataType.TEXT),
            Property(name="recall_reason", data_type=DataType.TEXT),
            Property(name="recall_date", data_type=DataType.TEXT),
            Property(name="recall_class", data_type=DataType.TEXT),
        ],
    )
    logger.info("Created collection '%s'.", DRUG_RECALLS)
