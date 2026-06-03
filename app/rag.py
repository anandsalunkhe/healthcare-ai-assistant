"""
RAG pipeline: Weaviate ingestion (hybrid search) + BGE-M3 + BGE reranker.

Public API
----------
ingest_documents() -> dict           Ingest .txt files and CSV; returns summary.
search_healthcare(query) -> list     Hybrid search + rerank on HealthcareText.
search_drug_recalls(query, drug_name=None) -> list  Hybrid search on DrugRecalls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.embeddings import embed_query, embed_texts
from app.reranker import rerank
from app.weaviate_client import DRUG_RECALLS, HEALTHCARE_TEXT, get_weaviate_client

try:
    from weaviate.classes.query import Filter, MetadataQuery
except ImportError:
    from weaviate.classes.query import MetadataQuery  # type: ignore
    Filter = None  # type: ignore

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


# ── Ingestion ─────────────────────────────────────────────────────────────

def ingest_documents() -> dict:
    """
    Ingest all .txt files and fda_drug_recalls.csv from DATA_DIR into Weaviate.
    Returns a summary dict with txt_chunks, csv_rows, and status.
    """
    client = get_weaviate_client()
    txt_chunks = _ingest_text_documents(client)
    csv_rows = _ingest_drug_recalls(client)
    return {"txt_chunks": txt_chunks, "csv_rows": csv_rows, "status": "ok"}


def _ingest_text_documents(client) -> int:
    """Chunk all .txt files and insert into HealthcareText collection."""
    data_path = Path(settings.DATA_DIR)
    txt_files = sorted(data_path.glob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in '%s'.", data_path.resolve())
        return 0

    # Clear and recreate for idempotent re-ingestion
    client.collections.delete(HEALTHCARE_TEXT)
    from app.weaviate_client import _ensure_healthcare_text
    _ensure_healthcare_text(client)
    collection = client.collections.get(HEALTHCARE_TEXT)

    all_chunks: list[dict] = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        chunks = _splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({"content": chunk, "source": txt_file.name, "chunk_id": idx})

    if not all_chunks:
        return 0

    texts = [c["content"] for c in all_chunks]
    vectors = embed_texts(texts)

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(all_chunks, vectors):
            batch.add_object(
                properties={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                },
                vector=vector,
            )

    logger.info("Ingested %d text chunks into '%s'.", len(all_chunks), HEALTHCARE_TEXT)
    return len(all_chunks)


def _ingest_drug_recalls(client) -> int:
    """Parse fda_drug_recalls.csv and insert into DrugRecalls collection."""
    csv_path = Path(settings.DATA_DIR) / "fda_drug_recalls.csv"
    if not csv_path.exists():
        logger.warning("fda_drug_recalls.csv not found — skipping drug recall ingestion.")
        return 0

    df = pd.read_csv(csv_path)
    required_cols = {"drug_name", "manufacturer", "recall_reason", "recall_date", "recall_class"}
    if not required_cols.issubset(set(df.columns)):
        logger.error("CSV missing required columns: %s", required_cols - set(df.columns))
        return 0

    # Clear and recreate for idempotent re-ingestion
    client.collections.delete(DRUG_RECALLS)
    from app.weaviate_client import _ensure_drug_recalls
    _ensure_drug_recalls(client)
    collection = client.collections.get(DRUG_RECALLS)

    rows = []
    for _, row in df.iterrows():
        content = (
            f"Drug: {row['drug_name']}. "
            f"Manufacturer: {row['manufacturer']}. "
            f"Recall Reason: {row['recall_reason']}. "
            f"Recall Date: {row['recall_date']}. "
            f"Class: {row['recall_class']}."
        )
        rows.append({
            "content": content,
            "drug_name": str(row["drug_name"]),
            "manufacturer": str(row["manufacturer"]),
            "recall_reason": str(row["recall_reason"]),
            "recall_date": str(row["recall_date"]),
            "recall_class": str(row["recall_class"]),
        })

    if not rows:
        return 0

    texts = [r["content"] for r in rows]
    vectors = embed_texts(texts)

    # Use enumerate to safely index vectors by integer position (not pandas row index)
    with collection.batch.dynamic() as batch:
        for i, (_, row) in enumerate(df.iterrows()):
            batch.add_object(properties=rows[i], vector=vectors[i])

    logger.info("Ingested %d drug recall rows into '%s'.", len(rows), DRUG_RECALLS)
    return len(rows)


# ── Search ────────────────────────────────────────────────────────────────

def search_healthcare(query: str) -> List[dict]:
    """
    Hybrid BM25 + ANN search on HealthcareText, then rerank with BGE CrossEncoder.
    Returns top RETRIEVER_FINAL_K docs as list of dicts.

    Returns an empty list (not an exception) if the collection is missing or
    if embedding / search fails for any reason.
    """
    try:
        client = get_weaviate_client()

        if not client.collections.exists(HEALTHCARE_TEXT):
            logger.warning("Collection '%s' does not exist — ingest documents first.", HEALTHCARE_TEXT)
            return []

        collection = client.collections.get(HEALTHCARE_TEXT)
        query_vector = embed_query(query)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=settings.HYBRID_ALPHA,
            limit=settings.RETRIEVER_INITIAL_K,
            return_metadata=MetadataQuery(score=True),
        )

        docs = []
        for obj in response.objects:
            docs.append({
                "content": obj.properties.get("content", ""),
                "source": obj.properties.get("source", ""),
                "chunk_id": obj.properties.get("chunk_id", 0),
                "hybrid_score": obj.metadata.score if obj.metadata else 0.0,
            })

        return rerank(query, docs, top_k=settings.RETRIEVER_FINAL_K)

    except Exception as exc:
        logger.error("Healthcare search failed: %s", exc)
        return []


def search_drug_recalls(query: str, drug_name: str | None = None) -> List[dict]:
    """
    Hybrid search on DrugRecalls, optionally filtered by drug_name.
    Returns top RETRIEVER_FINAL_K docs as list of dicts.

    Returns an empty list (not an exception) if the collection is missing or
    if embedding / search fails for any reason.
    """
    try:
        client = get_weaviate_client()

        if not client.collections.exists(DRUG_RECALLS):
            logger.warning("Collection '%s' does not exist — ingest documents first.", DRUG_RECALLS)
            return []

        collection = client.collections.get(DRUG_RECALLS)
        query_vector = embed_query(query)

        kwargs: dict = dict(
            query=query,
            vector=query_vector,
            alpha=settings.HYBRID_ALPHA,
            limit=settings.RETRIEVER_INITIAL_K,
            return_metadata=MetadataQuery(score=True),
        )

        if drug_name and Filter is not None:
            kwargs["filters"] = Filter.by_property("drug_name").like(f"*{drug_name}*")

        response = collection.query.hybrid(**kwargs)

        docs = []
        for obj in response.objects:
            docs.append({
                "content": obj.properties.get("content", ""),
                "drug_name": obj.properties.get("drug_name", ""),
                "manufacturer": obj.properties.get("manufacturer", ""),
                "recall_reason": obj.properties.get("recall_reason", ""),
                "recall_date": obj.properties.get("recall_date", ""),
                "recall_class": obj.properties.get("recall_class", ""),
                "hybrid_score": obj.metadata.score if obj.metadata else 0.0,
            })

        return rerank(query, docs, top_k=settings.RETRIEVER_FINAL_K)

    except Exception as exc:
        logger.error("Drug recall search failed: %s", exc)
        return []
