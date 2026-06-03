# Healthcare AI Assistant — Technical Deep Dive

**Stack:** FastAPI · Weaviate · BGE-M3 · BGE Reranker · qwen3:8b · Gemini 2.5 Flash · Streamlit · Docker Compose

---

## System Design

### Two-Stage Retrieval Pipeline

```
Query
  │
  ▼
BGE-M3 embed_query()            [1024-dim vector, normalized]
  │
  ▼
Weaviate Hybrid Search          [BM25 (sparse) + ANN (dense), α=0.5, top-15]
  │
  ▼
BGE Reranker CrossEncoder       [cross-attention scoring, top-3 selected]
  │
  ▼
Context Assembly                [3 chunks, ~1500 tokens]
  │
  ▼
qwen3:8b via Ollama             [primary]
  └── exception → Gemini 2.5 Flash [fallback, silent]
```

### Why Hybrid Search?
BM25 captures exact keyword matches (drug names, medical terms). ANN captures semantic similarity. Hybrid fusion combines both signals — critical in healthcare where exact terminology matters (e.g., "NDMA" must match exactly, not just semantically).

### Why Two-Stage Retrieval?
- Stage 1 (hybrid): Fast approximate retrieval, casts a wide net (top-15)
- Stage 2 (reranker): Slow but precise cross-encoder scoring on the 15 candidates → selects top-3 for LLM context

The CrossEncoder reads the full (query, document) pair jointly — far more accurate than embedding similarity but too slow to run on the full corpus.

---

## Weaviate v4 Client Details

### Connection
```python
weaviate.connect_to_custom(
    http_host="weaviate",  # Docker service name
    http_port=8080,
    grpc_host="weaviate",
    grpc_port=50051,
    http_secure=False,
    grpc_secure=False,
)
```

Weaviate v4 uses gRPC for vector operations (faster than REST). `connect_to_local()` would not work here because the host is the Docker service name `weaviate`, not `localhost`.

### Schema
```python
Configure.Vectorizer.none()  # We supply BGE-M3 vectors; Weaviate does not vectorize
```

Two collections:
- **HealthcareText**: `content (TEXT)`, `source (TEXT)`, `chunk_id (INT)`
- **DrugRecalls**: `content (TEXT)`, `drug_name (TEXT)`, `manufacturer (TEXT)`, `recall_reason (TEXT)`, `recall_date (TEXT)`, `recall_class (TEXT)`

### Hybrid Search Pattern
```python
collection.query.hybrid(
    query=query_text,    # for BM25 tokenization
    vector=query_vector, # for ANN cosine similarity
    alpha=0.5,           # 0=pure BM25, 1=pure ANN, 0.5=balanced
    limit=15,
    return_metadata=MetadataQuery(score=True),
)
```

---

## BGE-M3 Embeddings

Model: `BAAI/bge-m3` (1024 dimensions, ~570M params)

```python
SentenceTransformer("BAAI/bge-m3").encode(
    texts,
    normalize_embeddings=True,  # L2-normalize for cosine similarity
    show_progress_bar=False,
)
```

Why BGE-M3?
- Multi-lingual (100+ languages)
- Multi-functionality: dense, sparse, and multi-vector retrieval in one model
- MTEB leaderboard top-tier performance
- 1024-dim vectors encode rich semantic information

Singleton pattern ensures the ~1.8 GB model loads once and stays in memory for the container lifetime.

---

## BGE Reranker

Model: `BAAI/bge-reranker-base` (CrossEncoder)

```python
CrossEncoder("BAAI/bge-reranker-base").predict(
    [[query, doc] for doc in candidate_docs]
)
```

CrossEncoder architecture: both query AND document are fed together through a transformer's attention mechanism. This allows cross-attention between query and document tokens — far more accurate than computing independent embeddings and taking dot products.

Output: scalar relevance score per (query, doc) pair. We sort descending, take top-3.

---

## LLM Strategy

### Primary: qwen3:8b via Ollama
```python
ChatOpenAI(
    base_url="http://ollama:11434/v1",  # OpenAI-compatible endpoint
    api_key="ollama",                    # placeholder, not validated
    model="qwen3:8b",
    temperature=0.1,
    timeout=120,
)
```

### Fallback: Gemini 2.5 Flash
```python
ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.1,
    timeout=60,
)
```

### Fallback Logic
```python
def generate_answer(context, question):
    try:
        result = (PROMPT | ollama_llm).invoke(...)
        return result.content, "ollama"
    except Exception:          # catch ANY exception silently
        pass
    try:
        result = (PROMPT | gemini_llm).invoke(...)
        return result.content, "gemini"
    except Exception:
        return "I could not find this information...", "error"
```

### Background Model Pull
On startup, the API kicks off an async background task that POSTs to Ollama's pull endpoint. The API is immediately available (using Gemini) while the ~5 GB qwen3:8b model downloads. Once downloaded, subsequent requests use Ollama.

```python
asyncio.create_task(pull_model_background())  # non-blocking
```

---

## 3-Way Agentic Router

```python
def _route(question: str) -> str:
    lower = question.lower()
    if any(kw in lower for kw in APPOINTMENT_KW):
        return "appointment"
    if any(kw in lower for kw in DRUG_RECALL_KW):
        return "drug_recall"
    return "healthcare_rag"
```

**appointment**: Returns mock slot data instantly — no LLM, no retrieval, sub-millisecond.
**drug_recall**: Searches DrugRecalls collection. Drug name can be extracted for filtered search.
**healthcare_rag**: Default. Searches HealthcareText collection across all 5 policy/clinical docs.

---

## Response Schema

```json
{
  "answer": "string",
  "sources": [{"document": "string", "chunk": "string"}],
  "confidence": "high | medium | low | N/A",
  "route": "appointment | drug_recall | healthcare_rag",
  "retrieval_time_ms": 245.3,
  "generation_time_ms": 1823.7
}
```

Confidence scoring:
- `high`: ≥3 docs returned by reranker
- `medium`: 1–2 docs
- `low`: 0 docs
- `N/A`: appointment route (no retrieval)

---

## Docker Compose Architecture

```yaml
services:
  weaviate:   semitechnologies/weaviate:1.26.1  # ports 8080, 50051
  ollama:     ollama/ollama:latest              # port 11434
  api:        build from root Dockerfile        # port 8000
  frontend:   build from ./frontend/            # port 8501
```

**Dependency chain:**
`frontend` → `api` (healthy) → `weaviate` (healthy) + `ollama` (healthy)

**Named volumes:**
- `weaviate_data`: Weaviate persistence (survives `docker compose down`)
- `ollama_data`: Ollama model storage (~5 GB for qwen3:8b)
- `hf_cache`: HuggingFace models (BGE-M3 + reranker, ~1.8 GB)

The `hf_cache` volume ensures models download once and persist. Subsequent `docker compose up` uses cached models.

---

## Interview Questions & Answers

**Q: Why Weaviate over ChromaDB, Pinecone, or FAISS?**
A: Weaviate provides native hybrid search (BM25 + ANN) in a single query, whereas ChromaDB only supports pure vector search. For healthcare, exact terminology matters enormously (drug names, ICD codes). Weaviate is production-grade, supports gRPC for fast vector operations, and has a strong v4 Python client. Pinecone is managed-only (no local Docker). FAISS has no BM25 support.

**Q: Why BGE-M3 instead of OpenAI Ada-002 or all-MiniLM?**
A: BGE-M3 is open-source, runs locally (no API cost or latency), achieves top-tier MTEB scores, supports multi-lingual text, and outputs 1024-dimensional vectors for high-fidelity semantic representation. all-MiniLM is fast but only 384-dim — significant information loss. Ada-002 requires external API calls adding latency and cost.

**Q: Explain the hybrid search alpha parameter.**
A: `alpha=0.5` means equal weighting of BM25 and ANN scores. `alpha=0` is pure BM25 (exact keyword matching). `alpha=1` is pure ANN (semantic similarity). For healthcare, 0.5 is a good default because medical queries benefit from both: exact drug names need BM25, contextual understanding needs ANN. Weaviate uses Reciprocal Rank Fusion (RRF) to combine the two ranked lists.

**Q: Why use a CrossEncoder reranker as a second stage?**
A: Bi-encoders (like BGE-M3) encode query and document independently — efficient but lose cross-attention context. CrossEncoders process (query, document) pairs together through full transformer attention, seeing both simultaneously. This enables token-level interaction and produces more accurate relevance scores. Too slow for first-stage retrieval across thousands of docs, but ideal for re-scoring 15 candidates.

**Q: How does the Gemini fallback work technically?**
A: Google's Gemini API exposes an OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai/`. The `langchain-openai` `ChatOpenAI` class works against it with no code changes — just a different `base_url` and `api_key`. The fallback fires on `ANY` exception from the Ollama call, including connection refused, timeout, and model-not-found errors.

**Q: What happens if both Ollama and Gemini fail?**
A: `generate_answer()` returns the exact phrase `"I could not find this information in the provided documents."` with source `"error"`. The API never raises an exception to the client — it always returns a 200 with a degraded answer.

**Q: How is the system idempotent for ingestion?**
A: `_ingest_text_documents()` calls `client.collections.delete(HEALTHCARE_TEXT)` before re-creating the collection and re-inserting all chunks. Same pattern for DrugRecalls. This means `POST /ingest` can be called multiple times safely — it always results in exactly one copy of the data.

**Q: Why `enumerate(df.iterrows())` instead of using the pandas index?**
A: `df.iterrows()` returns `(pandas_index, row)` tuples where `pandas_index` is the DataFrame's row label — which may be non-sequential integers, strings, or multi-level. The `vectors` list is a Python list indexed 0, 1, 2... so using `vectors[pandas_index]` would fail with a non-zero-based index. `enumerate()` gives a reliable 0-based counter `i` that always aligns with `vectors[i]`.

**Q: How does the API start fast despite needing a large model download?**
A: `asyncio.create_task(pull_model_background())` schedules the pull coroutine on the event loop without blocking the FastAPI startup. The lifespan context manager returns immediately. The API accepts requests right away. `pull_model_background()` uses `httpx.AsyncClient` with a 600-second timeout to POST to `{OLLAMA_BASE_URL}/api/pull`. While downloading, all LLM calls use Gemini fallback transparently.

**Q: What are the HIPAA considerations in this system?**
A: This is a demo system with synthetic data. In production: (1) All PHI must be encrypted at rest (AES-256) and in transit (TLS). (2) Weaviate should be deployed with authentication enabled (not anonymous). (3) LLM API calls must be to HIPAA Business Associate-compliant providers. (4) Audit logs of all queries must be maintained. (5) No actual patient data should be ingested without a proper BAA in place.

**Q: How would you scale this system for 10,000 concurrent users?**
A: (1) Deploy Weaviate in multi-node cluster mode with replication factor ≥2. (2) Horizontally scale the FastAPI API behind a load balancer (e.g., AWS ALB). (3) Add a Redis semantic cache layer — cache (query_hash, answer) pairs for common questions. (4) Use an async job queue (Celery + Redis) for ingestion. (5) Deploy qwen3:8b on GPU instances for <1s generation. (6) BGE-M3 and reranker on separate inference servers (Triton or TorchServe). (7) Rate limiting per user to prevent abuse.

**Q: Why does the system use `uv` instead of `pip` or `poetry`?**
A: `uv` is 10-100x faster than pip for dependency resolution and installation, written in Rust. Unlike Poetry, it's fully PEP 517/518 compliant and uses standard `pyproject.toml`. It creates lockfiles (`uv.lock`) for reproducible builds. The `uv sync --no-dev --no-install-project` pattern in Docker is optimal: install only runtime deps, don't install the package itself (we COPY the source directly).

**Q: How does the Streamlit frontend handle the case where qwen3:8b is still downloading?**
A: The API always returns a response (using Gemini fallback). The frontend has a 120-second timeout on the `/ask` request. If the API is completely unreachable, it catches the exception and displays a user-friendly warning message rather than crashing. The `/health` endpoint shows `ollama_ready: false` during download so users can see the status.

**Q: Explain the confidence scoring logic.**
A: After reranking, we count the number of docs returned (≤ RETRIEVER_FINAL_K=3). ≥3 docs = `high` (strong evidence from multiple sources). 1-2 docs = `medium` (some evidence but not comprehensive). 0 docs = `low` (no relevant context found — LLM likely triggered the "I could not find" fallback phrase). `N/A` for appointment routes which don't use retrieval.

**Q: What security vulnerabilities exist and how are they mitigated?**
A: (1) Prompt injection: mitigated by strict system prompt rules — LLM is instructed to use ONLY the provided context. (2) SSRF: the API doesn't accept user-provided URLs. (3) Path traversal: `DATA_DIR` is a fixed config value, not user-supplied. (4) Dependency vulnerabilities: `uv` with lockfile for reproducible, auditable builds. (5) Weaviate anonymous auth: acceptable for demo, must be changed with API keys for production.
