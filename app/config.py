"""
Application configuration — loaded from environment variables via pydantic-settings.

All values can be overridden by a .env file or real environment variables.
Import the module-level `settings` singleton; never instantiate Settings directly.
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Weaviate ────────────────────────────────────────────────────────────
    # Use the Docker service name when running in Docker Compose;
    # override to "localhost" for local development.
    WEAVIATE_HOST: str = "weaviate"
    WEAVIATE_HTTP_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051

    # ── Primary LLM — Ollama (local) ────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    LOCAL_MODEL_NAME: str = "qwen3:8b"

    # ── Fallback LLM — Google Gemini (cloud) ───────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # ── Cloud Embedding Fallback — Gemini text-embedding-004 ───────────────
    GEMINI_EMBED_MODEL_NAME: str = "text-embedding-004"

    # ── Embedding & reranking models ────────────────────────────────────────
    EMBED_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # ── RAG retrieval pipeline ──────────────────────────────────────────────
    RETRIEVER_INITIAL_K: int = 15    # Hybrid search candidates (pre-rerank)
    RETRIEVER_FINAL_K: int = 3       # Top results sent to LLM (post-rerank)
    HYBRID_ALPHA: float = 0.5        # 0.0 = pure BM25, 1.0 = pure vector
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Storage ─────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path("data")


# Module-level singleton — import this everywhere
settings = Settings()
