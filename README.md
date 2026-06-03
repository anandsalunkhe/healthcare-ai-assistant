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

