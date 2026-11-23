"""Streamlit UI for the Production RAG System.

Runs the RAG pipeline in-process, so it can be deployed on Streamlit Community Cloud
without a separate API server:  main file path = streamlit_app.py
"""
import html
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_secrets():
    """Expose Streamlit secrets as env vars (settings.py reads the environment)."""
    try:
        for key in ("OPENROUTER_API_KEY", "LLM_MODELS", "ALLOW_UPLOADS"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets file locally - that's fine
        pass


_load_secrets()

from app.config.settings import settings  # noqa: E402
from app.evaluation.evaluator import RAGEvaluator  # noqa: E402
from app.generation.rag_pipeline import RAGPipeline  # noqa: E402
from app.ingestion.document_loader import list_document_files  # noqa: E402
from app.ingestion.indexer import ensure_index  # noqa: E402
from ui.styles import CSS, HERO, PIPELINE_DOT  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "What are the office timings?",
    "How many annual paid leaves do employees get?",
    "Is two-factor authentication mandatory?",
    "Which cloud platform is used for deployments?",
    "What is required before merging code into production?",
]


# ----------------------------------------------------------------- pipeline
@st.cache_resource(show_spinner="Loading models and building the index (first run takes about a minute)…")
def load_pipeline() -> RAGPipeline:
    ensure_index()
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not start the RAG pipeline: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_key = st.text_input(
        "OpenRouter API key (optional)",
        type="password",
        help="Used only for your session and never stored. Without a key the app "
        "answers by extracting the most relevant sentences from your documents.",
    )
    api_key = user_key.strip() or settings.OPENROUTER_API_KEY

    if api_key:
        st.markdown('<span class="badge badge-llm">● LLM answers on</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-ext">● Retrieval-only mode</span>', unsafe_allow_html=True)

    retrieve_k = st.slider("Candidates per retriever", 3, 10, settings.TOP_K_RETRIEVAL)
    context_k = st.slider("Chunks used for the answer", 1, 5, settings.TOP_K_CONTEXT)

    st.markdown("### 💡 Try asking")
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample_{i}"):
            st.session_state.pending = sample

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------ helpers
def render_result(result: dict):
    if result["blocked"]:
        st.warning(f"{result['answer']}  \n_{result['notice']}_")
        return

    st.markdown(result["answer"])

    if result["mode"] == "llm":
        st.markdown(
            f'<span class="badge badge-llm">LLM</span> `{result["model"]}`', unsafe_allow_html=True
        )
    else:
        st.markdown('<span class="badge badge-ext">Extracted from documents</span>', unsafe_allow_html=True)
    if result["notice"]:
        st.caption(f"ℹ️ {result['notice']}")

    stages = result["stages"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{result['latency']:.2f}s")
    c2.metric("Retrieval", f"{stages['retrieval']:.2f}s")
    c3.metric("Rerank", f"{stages['rerank']:.2f}s")
    c4.metric("Answer", f"{stages['generation']:.2f}s")

    with st.expander(f"📎 Sources ({len(result['sources'])})"):
        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f'<div class="source-card"><div class="meta"><b>[{i}] {html.escape(src["source"])}</b>'
                f' · chunk {src["chunk_id"]} · relevance {src["score"]:.2f}</div>'
                f'{html.escape(src["content"])}</div>',
                unsafe_allow_html=True,
            )


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)


def save_uploads(files) -> list:
    saved = []
    settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    for f in files:
        data = f.getvalue()
        if len(data) > settings.MAX_UPLOAD_BYTES:
            st.warning(f"{f.name}: larger than {settings.MAX_UPLOAD_BYTES // 1000} KB, skipped.")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            st.warning(f"{f.name}: not valid UTF-8 text, skipped.")
            continue
        (settings.DOCUMENTS_PATH / safe_filename(f.name)).write_text(text, encoding="utf-8")
        saved.append(safe_filename(f.name))
    return saved


# --------------------------------------------------------------------- page
st.markdown(HERO, unsafe_allow_html=True)
tab_ask, tab_kb, tab_eval, tab_metrics, tab_arch = st.tabs(
    ["💬 Ask", "📚 Knowledge base", "🧪 Evaluation", "📈 Metrics", "🏗️ How it works"]
)

# ---- Ask ------------------------------------------------------------------
with tab_ask:
    history = st.container()

    with st.form("ask_form", clear_on_submit=True):
        col_text, col_btn = st.columns([6, 1])
        text = col_text.text_input(
            "Question",
            placeholder="e.g. How many annual leaves are allowed?",
            label_visibility="collapsed",
        )
        submitted = col_btn.form_submit_button("Ask", type="primary")

    query = st.session_state.pop("pending", None)
    if submitted and text.strip():
        query = text.strip()

    if query:
        with st.spinner("Retrieving, reranking and answering…"):
            result = pipeline.run(query, top_k=retrieve_k, context_k=context_k, api_key=api_key)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "result": result})

    with history:
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-state"><h4>Ask anything about the knowledge base</h4>'
                "Pick a sample question from the sidebar or type your own below.</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    render_result(msg["result"])

# ---- Knowledge base ---------------------------------------------------------
with tab_kb:
    files = list_document_files()
    chunk_counts = Counter(d["source"] for d in pipeline.vector_store.documents)

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(files))
    m2.metric("Chunks indexed", len(pipeline.vector_store.documents))
    m3.metric("Chunk size", f"{settings.CHUNK_SIZE} chars")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Document": f.name,
                    "Size (KB)": round(f.stat().st_size / 1024, 1),
                    "Chunks": chunk_counts.get(f.name, 0),
                }
                for f in files
            ]
        ),
        hide_index=True,
    )

    for f in files:
        with st.expander(f"📄 {f.name}"):
            st.text(f.read_text(encoding="utf-8", errors="replace")[:4000])

    st.markdown("#### Add your own documents")
    if settings.ALLOW_UPLOADS:
        st.caption(
            "Upload .txt or .md files (UTF-8, up to 200 KB each). The index is rebuilt automatically. "
            "Note: the knowledge base is shared by everyone using this deployment."
        )
        uploaded = st.file_uploader("Documents", type=["txt", "md"], accept_multiple_files=True)
        if uploaded and st.button("➕ Add to knowledge base and re-index", type="primary"):
            saved = save_uploads(uploaded)
            if saved:
                with st.spinner("Re-indexing…"):
                    ensure_index(force=True)
                st.cache_resource.clear()
                st.toast(f"Added {len(saved)} document(s).")
                st.rerun()
    else:
        st.info("Uploads are disabled on this deployment (ALLOW_UPLOADS=false).")

# ---- Evaluation -------------------------------------------------------------
with tab_eval:
    st.markdown(
        "Runs the questions in `benchmark/questions.json` and measures **answer accuracy** "
        "(expected keywords appear in the answer), **retrieval hit-rate** (the right document was retrieved) "
        "and **latency**."
    )
    if st.button("▶ Run benchmark", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        report = RAGEvaluator(pipeline).evaluate(
            api_key=api_key,
            progress=lambda i, n: bar.progress(i / n, text=f"Question {i} of {n}"),
        )
        bar.empty()
        st.session_state.eval_report = report

    report = st.session_state.get("eval_report")
    if report:
        e1, e2, e3 = st.columns(3)
        e1.metric("Answer accuracy", f"{report['accuracy']:.0%}")
        e2.metric("Retrieval hit-rate", f"{report['retrieval_hit_rate']:.0%}")
        e3.metric("Avg latency", f"{report['average_latency']}s")

        df = pd.DataFrame(report["results"])
        st.dataframe(df, hide_index=True)
        st.download_button(
            "⬇️ Download results (CSV)", df.to_csv(index=False), "evaluation_results.csv", "text/csv"
        )

# ---- Metrics ----------------------------------------------------------------
with tab_metrics:
    metrics_file = settings.METRICS_FILE
    df_m = pd.DataFrame()
    if metrics_file.exists():
        try:
            df_m = pd.read_csv(metrics_file)
        except Exception:  # noqa: BLE001 - corrupt/legacy file
            df_m = pd.DataFrame()

    if df_m.empty or "latency" not in df_m.columns:
        st.info("No queries logged yet. Ask a question in the **Ask** tab and come back.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Queries logged", len(df_m))
        k2.metric("Average latency", f"{df_m['latency'].mean():.2f}s")
        k3.metric("95th percentile", f"{df_m['latency'].quantile(0.95):.2f}s")

        st.markdown("**Latency per query (seconds)**")
        st.line_chart(df_m["latency"].reset_index(drop=True))

        if "mode" in df_m.columns:
            st.markdown("**Answers by mode**")
            st.bar_chart(df_m["mode"].value_counts())

        st.dataframe(df_m.tail(20).iloc[::-1], hide_index=True)

# ---- How it works -------------------------------------------------------------
with tab_arch:
    st.graphviz_chart(PIPELINE_DOT)
    st.markdown(
        f"""
| Stage | What it does | Implementation |
|---|---|---|
| **Input guardrails** | Blocks empty, oversized and prompt-injection queries | `app/guardrails` |
| **Query processing** | Splits multi-part questions, expands domain terms | `app/retrieval/query_processor.py` |
| **Dense retrieval** | Semantic search with `{settings.EMBEDDING_MODEL.split('/')[-1]}` + FAISS | `app/core/vector_store.py` |
| **Sparse retrieval** | BM25 keyword matching for exact terms | `app/retrieval/bm25_retriever.py` |
| **RRF fusion** | Merges both ranked lists | `app/retrieval/rrf_fusion.py` |
| **Reranking** | `{settings.CROSS_ENCODER_MODEL.split('/')[-1]}` re-scores candidates | `app/reranking` |
| **Answer** | LLM via OpenRouter, or extractive fallback without a key | `app/generation` |
| **Grounding check** | Rejects answers not supported by the context | `app/guardrails/output_guardrails.py` |
"""
    )
