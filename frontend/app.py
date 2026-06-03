"""Streamlit frontend — Mindbowser Enterprise Healthcare AI."""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Mindbowser Healthcare AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 Healthcare AI")
    st.caption("Mindbowser AI Engineer Hackathon")
    st.divider()

    # Health check
    st.subheader("System Status")
    if st.button("🔄 Check Health", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            h = r.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weaviate", "✅" if h.get("weaviate_ready") else "❌")
            with col2:
                st.metric("Ollama", "✅" if h.get("ollama_ready") else "⏳")
            st.caption(f"Embed: `{h.get('embed_model', '—')}`")
            st.caption(f"LLM: `{h.get('model_name', '—')}`")
        except Exception as exc:
            st.error(f"API unreachable: {exc}")

    st.divider()

    # Ingest trigger
    st.subheader("Knowledge Base")
    if st.button("📥 Ingest Documents", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                r = requests.post(f"{API_URL}/ingest", timeout=300)
                d = r.json()
                st.success(
                    f"✅ Done in {d.get('duration_ms', 0):.0f} ms\n\n"
                    f"• {d.get('txt_chunks', 0)} text chunks\n"
                    f"• {d.get('csv_rows', 0)} drug recall rows"
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()

    # Sample questions
    st.subheader("Sample Questions")
    samples = [
        "What are the signs of wound infection after surgery?",
        "Is Metformin subject to any FDA recall?",
        "How do I schedule an appointment?",
        "What is the medication refill policy for controlled substances?",
        "What are my HIPAA rights to access my medical records?",
        "What conditions are eligible for a telehealth visit?",
        "What should I do if I have chest pain after discharge?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending_question = q
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────

st.title("🏥 Mindbowser Enterprise Healthcare AI")
st.caption("Ask questions about healthcare policies, medications, appointments, and more.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Route", meta.get("route", "—"))
            c2.metric("Confidence", meta.get("confidence", "—"))
            c3.metric("Retrieval", f"{meta.get('retrieval_time_ms', 0):.0f} ms")
            c4.metric("Generation", f"{meta.get('generation_time_ms', 0):.0f} ms")
            if meta.get("sources"):
                with st.expander(f"📄 Sources ({len(meta['sources'])})"):
                    for src in meta["sources"]:
                        st.markdown(f"**{src.get('document', 'unknown')}**")
                        st.caption(src.get("chunk", "")[:300] + "…")


def _ask(question: str) -> None:
    """Send question to API and append both messages to session state."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.Timeout:
                data = {
                    "answer": "⚠️ The request timed out. The model may still be loading. Please try again shortly.",
                    "sources": [],
                    "confidence": "N/A",
                    "route": "error",
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                }
            except Exception as exc:
                data = {
                    "answer": f"⚠️ API error: {exc}",
                    "sources": [],
                    "confidence": "N/A",
                    "route": "error",
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                }

        answer = data.get("answer", "No answer returned.")
        st.markdown(answer)

        meta = {
            "route": data.get("route", "—"),
            "confidence": data.get("confidence", "—"),
            "retrieval_time_ms": data.get("retrieval_time_ms", 0),
            "generation_time_ms": data.get("generation_time_ms", 0),
            "sources": data.get("sources", []),
        }
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", meta["route"])
        c2.metric("Confidence", meta["confidence"])
        c3.metric("Retrieval", f"{meta['retrieval_time_ms']:.0f} ms")
        c4.metric("Generation", f"{meta['generation_time_ms']:.0f} ms")

        if meta["sources"]:
            with st.expander(f"📄 Sources ({len(meta['sources'])})"):
                for src in meta["sources"]:
                    st.markdown(f"**{src.get('document', 'unknown')}**")
                    st.caption(src.get("chunk", "")[:300] + "…")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


# Handle question from sidebar sample button
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    _ask(q)
    st.rerun()

# Handle question from chat input
if prompt := st.chat_input("Ask a healthcare question…"):
    _ask(prompt)
    st.rerun()
