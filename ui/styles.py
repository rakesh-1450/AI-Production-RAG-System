"""Custom CSS for the Streamlit UI (works in light and dark themes)."""

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1150px; }

.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    color: #fff;
}
.hero h1 { margin: 0 0 .3rem 0; font-size: 2rem; color: #fff; }
.hero p  { margin: 0; opacity: .92; font-size: 1.02rem; }

.chips { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.chip {
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
    padding: .15rem .7rem; border-radius: 999px; font-size: .78rem; color: #fff;
}

.badge {
    display: inline-block; padding: .15rem .65rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600; margin-right: .4rem;
}
.badge-llm  { background: rgba(16,185,129,.18); color: #10b981; border: 1px solid rgba(16,185,129,.5); }
.badge-ext  { background: rgba(245,158,11,.16); color: #f59e0b; border: 1px solid rgba(245,158,11,.5); }
.badge-warn { background: rgba(239,68,68,.16);  color: #ef4444; border: 1px solid rgba(239,68,68,.5); }

.source-card {
    border: 1px solid rgba(128,128,128,.28); border-left: 4px solid #6366f1;
    border-radius: 10px; padding: .6rem .9rem; margin-bottom: .6rem;
}
.source-card .meta { font-size: .8rem; opacity: .75; margin-bottom: .25rem; }

.empty-state {
    text-align: center; padding: 2.5rem 1rem; opacity: .8;
    border: 1px dashed rgba(128,128,128,.4); border-radius: 14px;
}

div[data-testid="stMetric"] {
    background: rgba(128,128,128,.08); border-radius: 12px; padding: .6rem .9rem;
}
.stButton > button { width: 100%; border-radius: 10px; }
footer { visibility: hidden; }
</style>
"""

HERO = """
<div class="hero">
  <h1>🔎 Production RAG Assistant</h1>
  <p>Ask questions about your documents — answers are grounded in retrieved sources, with citations.</p>
  <div class="chips">
    <span class="chip">FAISS dense search</span>
    <span class="chip">BM25 keyword search</span>
    <span class="chip">Reciprocal Rank Fusion</span>
    <span class="chip">Cross-encoder reranking</span>
    <span class="chip">Guardrails</span>
    <span class="chip">Evaluation &amp; metrics</span>
  </div>
</div>
"""

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1",
        fontname="Helvetica", fontsize=11, fontcolor="#1e1b4b"];
  edge [color="#6366f1"];
  Q   [label="User query"];
  IG  [label="Input\\nguardrails"];
  QP  [label="Query\\nprocessing"];
  D   [label="Dense search\\n(FAISS)"];
  B   [label="Keyword search\\n(BM25)"];
  R   [label="RRF fusion"];
  CE  [label="Cross-encoder\\nrerank"];
  G   [label="Answer\\n(LLM or extractive)"];
  OG  [label="Grounding\\ncheck"];
  A   [label="Answer + sources", fillcolor="#dcfce7", color="#16a34a"];
  Q -> IG -> QP;
  QP -> D; QP -> B;
  D -> R; B -> R;
  R -> CE -> G -> OG -> A;
}
"""
