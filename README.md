# 🏥 Healthcare AI Assistant

**Mindbowser AI Engineer Hackathon — Production Submission**

A fully containerized, production-grade RAG system for healthcare Q&A.  
Single command deployment: `docker compose up -d`

---

## Architecture

```
User Query
    │
    ▼
Streamlit Frontend (port 8501)
    │  REST
    ▼
FastAPI Backend (port 8000)
    │
    ▼
3-Way Agentic Router
    ├── appointment keywords  ──► Mock scheduling tool (instant, no LLM)
    ├── FDA/recall keywords   ──► DrugRecalls collection (Weaviate)
    └── everything else       ──► HealthcareText collection (Weaviate)
                                        │
                                        ▼
                              BGE-M3 Hybrid Search (BM25 + ANN, top 15)
                                        │
                                        ▼
                              BGE Reranker CrossEncoder (top 3)
                                        │
                                        ▼
                              qwen3:8b via Ollama  ──(fallback)──► Gemini 2.5 Flash
                                        │
                                        ▼
                              Structured Response (answer + sources + timing)
```

### Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI 0.115+ |
| Vector DB | Weaviate 1.26.1 |
| Embeddings | BAAI/bge-m3 (1024-dim, sentence-transformers) |
| Reranker | BAAI/bge-reranker-base (CrossEncoder) |
| Search | Weaviate Hybrid (BM25 + ANN, α=0.5) |
| Primary LLM | qwen3:8b via Ollama |
| Fallback LLM | Gemini 2.5 Flash |
| Frontend | Streamlit |
| Package Mgr | uv + hatchling |
| Containers | Docker Compose (4 services) |

---

## Quick Start — Single Command

```bash
# 1. Copy env file and set your Gemini API key (used as fallback while qwen3:8b downloads)
cp .env.example .env
# Edit .env — set GEMINI_API_KEY=your_key_here

# 2. Launch all 4 services
docker compose up -d

# 3. Ingest documents (run once after startup)
curl -X POST http://localhost:8000/ingest

# 4. Open the UI
start http://localhost:8501
```

> **Note:** On first boot, `qwen3:8b` (~5 GB) downloads in the background.
> The API starts immediately and uses Gemini as fallback until the download completes.
> Check Ollama status at `GET /health`.

---

## Services

| Service | URL | Description |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Chat interface |
| FastAPI API | http://localhost:8000 | REST backend |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Weaviate | http://localhost:8080 | Vector database |
| Ollama | http://localhost:11434 | LLM runtime |

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "weaviate_ready": true,
  "ollama_ready": true,
  "model_name": "qwen3:8b",
  "embed_model": "BAAI/bge-m3"
}
```

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{
  "status": "success",
  "txt_chunks": 142,
  "csv_rows": 10,
  "duration_ms": 8432.5
}
```

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the signs of wound infection after surgery?"}'
```

```json
{
  "answer": "Signs of wound infection include increased redness, swelling, warmth...",
  "sources": [{"document": "discharge_instructions.txt", "chunk": "..."}],
  "confidence": "high",
  "route": "healthcare_rag",
  "retrieval_time_ms": 245.3,
  "generation_time_ms": 1823.7
}
```

---

## Knowledge Base Documents

| File | Content |
|---|---|
| `discharge_instructions.txt` | Post-surgery care, wound care, medications, follow-up |
| `appointment_policy.txt` | Scheduling, cancellations, no-show policy |
| `insurance_eligibility_faq.txt` | Copays, deductibles, prior auth, billing |
| `hipaa_privacy_guidelines.txt` | Patient rights, PHI handling, breach notification |
| `medication_refill_policy.txt` | Refill requests, controlled substances, assistance |
| `telehealth_guidelines.txt` | Virtual visit eligibility, tech requirements |
| `fda_drug_recalls.csv` | 10 synthetic FDA drug recall records |

---

## Environment Variables

See `.env.example` for all options. Minimum required:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

All other values have working Docker defaults.

---

## Development

```bash
# Install dependencies locally
uv sync

# Run API locally (requires Weaviate + Ollama running)
uv run uvicorn app.main:app --reload --port 8000

# Run frontend locally
cd frontend && streamlit run app.py
```

---

## Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── main.py            # FastAPI app, lifespan, endpoints
│   ├── agent.py           # 3-way router with timing metrics
│   ├── rag.py             # Weaviate ingestion + hybrid search
│   ├── llm.py             # Ollama primary + Gemini fallback
│   ├── embeddings.py      # BGE-M3 singleton
│   ├── reranker.py        # BGE CrossEncoder reranker
│   ├── weaviate_client.py # Weaviate v4 client + schema
│   └── config.py          # pydantic-settings Settings
├── frontend/
│   ├── app.py             # Streamlit chat UI
│   ├── Dockerfile
│   └── requirements.txt
├── data/                  # Healthcare knowledge base
├── Dockerfile             # API container
├── docker-compose.yml     # 4-service orchestration
├── pyproject.toml         # uv / hatchling manifest
└── .env.example           # Environment template
```


---

## Architectural Decisions

### System Prompt

The exact system prompt injected into every LLM call (`_SYSTEM` in `app/llm.py`):

```
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

5. Be concise, factual, and professional in tone.

Context:
{context}
```

### Prompting Strategy

The prompt uses a strict **context-grounding** approach: the LLM is explicitly forbidden from drawing on prior training knowledge, and must respond with a fixed fallback phrase when the retrieved context is insufficient. Rule 3 directly prevents the model from issuing medical diagnoses or personalised medication advice, redirecting users to qualified professionals — this is the primary hallucination and liability guard.

### LLM Choice — `qwen3:8b`

`qwen3:8b` was chosen because it runs fully locally via Ollama, eliminating external API latency and data-privacy concerns for a healthcare context. At ~5 GB it delivers strong instruction-following and reasoning quality for its size, making it practical on commodity hardware (16 GB RAM) without sacrificing response coherence. Gemini 2.5 Flash is retained as an automatic fallback for the cold-start window while the model downloads.

### Embedding Model Choice — `BAAI/bge-m3`

`BAAI/bge-m3` produces 1024-dimensional dense vectors and also exposes sparse (BM25-compatible) weights, making it natively suited for hybrid retrieval. A single model handles both the dense ANN pass and the keyword-aware sparse pass, avoiding the complexity and latency of maintaining two separate embedding pipelines.

### Vector Database — Weaviate

Weaviate was chosen for its **native hybrid search** (BM25 + ANN in a single query), which is essential for healthcare queries that mix clinical terminology (keyword-sensitive) with semantic intent. Weaviate's v4 Python client also provides schema-as-code collection management and built-in multi-tenancy readiness, reducing operational overhead.

### Agent Router Workflow

The 3-way router in `app/agent.py` applies deterministic keyword matching before ever touching the LLM. Appointment-scheduling intents are resolved instantly via a mock tool; FDA/recall intents route to a dedicated Weaviate collection; only genuinely ambiguous clinical questions reach the full RAG + LLM pipeline. This tiered approach reduces median latency for the two most common intent categories to sub-100 ms and avoids unnecessary GPU/API cost.

---

## Limitations & Future Improvements

### Current Limitations

| Area | Limitation |
|---|---|
| **Intent Router** | Keyword-based matching is brittle — uncommon phrasings can misroute queries (e.g., "book a slot" may miss the appointment branch). |
| **Knowledge Base** | The 7-document corpus is static; there is no mechanism for real-time document updates or version control of ingested content. |
| **Appointment Tool** | Scheduling is a mock — no integration with a real calendar, EHR, or booking API exists. |
| **Evaluation** | There is no automated RAG evaluation pipeline (e.g., RAGAS faithfulness/relevancy scores) to catch regressions between releases. |
| **Auth & Multi-tenancy** | The API has no authentication layer; in production, patient data must be isolated per tenant with HIPAA-compliant access controls. |
| **Context Window** | Fixed top-3 reranked chunks may be insufficient for complex multi-document questions; chunk sizing is not adaptive. |

### Future Improvements

- **LangGraph migration** — replace the hand-rolled router with a LangGraph state machine for richer branching logic, memory, and observability.
- **Real-time booking integration** — connect the appointment tool to an EHR scheduling API (e.g., FHIR-compliant endpoint) for live slot availability.
- **Automated RAG evaluation** — integrate RAGAS or DeepEval in CI to gate every PR on faithfulness, answer relevancy, and context recall metrics.
- **Semantic router** — replace keyword matching with a lightweight classifier or embedding-based intent detector to improve routing accuracy on paraphrased queries.
- **Document pipeline** — add a watched ingestion queue (e.g., S3 + Lambda trigger) so the knowledge base stays current without manual `POST /ingest` calls.
- **Fine-tuned reranker** — fine-tune the BGE CrossEncoder on domain-specific healthcare Q&A pairs to improve reranking precision over the generic base model.

