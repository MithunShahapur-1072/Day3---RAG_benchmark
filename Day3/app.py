"""Streamlit UI – 5 tabs. Run:  streamlit run app.py"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pwc_rag.config import PRACTICE_AREAS, Hyperparams, demo_hyperparams
from pwc_rag.evaluation import ENGINES, run_benchmark, summarise
from pwc_rag.kb import KnowledgeBase, load_bytes, load_sample_corpus
from pwc_rag.llm import LLM
from pwc_rag.metrics import faithfulness
from pwc_rag.pipelines import BRANCHES, WORKFLOW_NODES, Pipelines, RunConfig

SAMPLE = Path(__file__).parent / "data" / "sample"
COLORS = {"Standard RAG": "#4C78A8", "HyDE RAG": "#F58518", "Graph RAG": "#54A24B"}
TYPE_COLORS = {"Authority": "#4C78A8", "Statute": "#E45756", "Provision": "#F58518", "Concept": "#54A24B"}

st.set_page_config(page_title="PwC India – RAG Benchmark", layout="wide")
S = st.session_state
S.setdefault("hp", demo_hyperparams())
S.setdefault("uploads", [])
S.setdefault("history", [])
S.setdefault("nodes_on", set(WORKFLOW_NODES))
S.setdefault("branches_on", set(BRANCHES))
S.setdefault("off_entities", [])
S.setdefault("off_relations", [])
S.setdefault("chat", [])


def build_kb():
    llm = LLM()
    docs = load_sample_corpus(SAMPLE) + S["uploads"]
    with st.spinner("Building vector + graph indexes…"):
        kb = KnowledgeBase(S["hp"], llm).build(docs)
    S["llm"], S["kb"], S["pipes"] = llm, kb, Pipelines(kb, llm)


if "kb" not in S:
    build_kb()
kb: KnowledgeBase = S["kb"]


def cfg(area: str) -> RunConfig:
    return RunConfig(hp=S["hp"], area=area, nodes_on=set(S["nodes_on"]), branches_on=set(S["branches_on"]),
                     disabled_entities=set(S["off_entities"]), disabled_relations=set(S["off_relations"]))


st.title("Standard vs HyDE vs Graph RAG – PwC India Advisory")
st.caption(f"LLM: **{S['llm'].name}** · Index: {kb.stats()} · Sample corpus is synthetic/illustrative.")
t1, t2, t3, t4, t5 = st.tabs(["1 · Ingestion & Chat", "2 · Knowledge Graph (3D)", "3 · LangGraph Workflow",
                              "4 · Hyperparameters & KPIs", "5 · Comparative Benchmark"])

# ------------------------------------------------------------------ Tab 1
with t1:
    c1, c2 = st.columns([1, 2])
    with c1:
        area = st.selectbox("Practice Area", list(PRACTICE_AREAS), key="area")
        st.info(f"**Guardrail:** {PRACTICE_AREAS[area]['guardrail']}")
        engine = st.radio("RAG engine", ENGINES, index=2, key="engine")
        files = st.file_uploader("Upload PDF / DOCX / JSON / TXT", type=["pdf", "docx", "json", "txt"],
                                 accept_multiple_files=True)
        if st.button("Index uploaded documents (tagged with selected practice area)") and files:
            for f in files:
                S["uploads"] += load_bytes(f.name, f.getvalue(), area)
            build_kb(); st.rerun()
    with c2:
        for m in S["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                for s in m.get("sources", []):
                    with st.expander(f"[{s['tag']}] {s['source']} p.{s['page']}"):
                        st.write(s["text"])
        q = st.chat_input("Ask a question about the indexed corpus")
        if q:
            S["chat"].append({"role": "user", "content": q})
            r = S["pipes"].run(engine, q, cfg(area))
            f = faithfulness(r["answer"], r["sources"])
            S["history"].append({"engine": engine, "faithfulness": f, **r["latency"]})
            lat = r["latency"]
            S["chat"].append({"role": "assistant", "sources": r["sources"], "content":
                              f"{r['answer']}\n\n*{engine} · {lat['total_ms']:.0f} ms · path: "
                              f"{' → '.join(r['trace'])}*"})
            st.rerun()

# ------------------------------------------------------------------ Tab 2
with t2:
    st.write("Entities sit on a sphere (coloured by type). Plotly cannot click-select 3D nodes, so use the pickers "
             "below to switch entities/relations **off**; switched-off items are excluded from graph retrieval.")
    g = kb.graph
    nodes = sorted(g.nodes)
    S["off_entities"] = st.multiselect("Deselect entities", nodes, default=[e for e in S["off_entities"] if e in g])
    all_edges = sorted({kb.edge_id(u, k, v) for u, v, k in g.edges(keys=True)})
    S["off_relations"] = st.multiselect("Deselect relations", all_edges,
                                        default=[e for e in S["off_relations"] if e in all_edges])
    n = max(len(nodes), 1)
    pos = {}
    for i, nd in enumerate(nodes):  # Fibonacci sphere
        y = 1 - 2 * (i + .5) / n
        r = math.sqrt(1 - y * y); th = math.pi * (3 - math.sqrt(5)) * i
        pos[nd] = (r * math.cos(th), y, r * math.sin(th))
    fig = go.Figure()
    ex, ey, ez, dx, dy, dz = [], [], [], [], [], []
    for u, v, k in g.edges(keys=True):
        off = kb.edge_id(u, k, v) in S["off_relations"] or u in S["off_entities"] or v in S["off_entities"]
        tgt = (dx, dy, dz) if off else (ex, ey, ez)
        for a, b in zip(tgt, zip(pos[u], pos[v])):
            a += [b[0], b[1], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#888", width=2),
                               name="active relation", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode="lines", line=dict(color="#ddd", width=1),
                               name="deselected relation", hoverinfo="skip"))
    for t, col in TYPE_COLORS.items():
        ns = [x for x in nodes if g.nodes[x]["type"] == t]
        fig.add_trace(go.Scatter3d(
            x=[pos[x][0] for x in ns], y=[pos[x][1] for x in ns], z=[pos[x][2] for x in ns], mode="markers+text",
            text=[x for x in ns], textfont=dict(size=9), name=t,
            marker=dict(size=[4 if x in S["off_entities"] else 7 for x in ns], color=col,
                        opacity=0.9, line=dict(width=0)),
            hovertext=[f"{x} ({len(g.nodes[x]['chunks'])} chunks)" for x in ns], hoverinfo="text"))
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Tab 3
with t3:
    st.write("Toggle LangGraph nodes and retrieval branches to test pipeline variations; "
             "Graph RAG runs (Tab 1 / Tab 5) use these settings. `hybrid_search` cannot be removed.")
    c1, c2 = st.columns(2)
    S["nodes_on"] = {"hybrid_search"} | {n for n in WORKFLOW_NODES if n != "hybrid_search" and
                                         c1.checkbox(n, value=n in S["nodes_on"], key=f"n_{n}")}
    S["branches_on"] = {b for b in BRANCHES if c2.checkbox(f"hybrid branch: {b}", value=b in S["branches_on"],
                                                           key=f"b_{b}")}
    xs = {n: i for i, n in enumerate(WORKFLOW_NODES)}
    fig = go.Figure()
    for a, b in Pipelines.workflow_edges(S["pipes"]):
        loop = (a, b) == ("rerank", "hybrid_search")
        y0 = -0.6 if loop else 0
        fig.add_trace(go.Scatter(x=[xs[a], (xs[a] + xs[b]) / 2 if loop else xs[b], xs[b]],
                                 y=[0, y0, 0] if loop else [0, 0, 0], mode="lines",
                                 line=dict(color="#c44" if loop else "#888", dash="dot" if loop else "solid"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=list(xs.values()), y=[0] * 5, mode="markers+text", text=WORKFLOW_NODES, textposition="top center",
        marker=dict(size=34, color=["#54A24B" if n in S["nodes_on"] else "#ccc" for n in WORKFLOW_NODES]),
        showlegend=False))
    fig.add_annotation(x=2.5, y=-0.6, text="sufficiency loop: widen hop radius (max 1×)", showarrow=False,
                       yshift=-12, font=dict(color="#c44"))
    fig.update_layout(height=320, xaxis_visible=False, yaxis_visible=False, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Active branches: " + ", ".join(sorted(S["branches_on"])) + " · fused with RRF when rrf_fusion is on.")

# ------------------------------------------------------------------ Tab 4
with t4:
    hp: Hyperparams = S["hp"]
    st.subheader("Hyperparameters")
    c = st.columns(4)
    cs = c[0].number_input("Chunk size (tokens)", 40, 1200, hp.chunk_size, 10)
    co = c[0].number_input("Chunk overlap (tokens)", 0, 200, hp.chunk_overlap, 5)
    hn = c[1].slider("Multi-HyDE passages N", 1, 4, hp.hyde_n)
    hr = c[2].slider("Graph hop radius k", 1, 4, hp.hop_radius)
    ra = c[3].slider("RRF constant α", 1, 120, hp.rrf_alpha)
    reindex = (cs, co) != (hp.chunk_size, hp.chunk_overlap)
    S["hp"] = Hyperparams(int(cs), int(co), hn, hr, ra)
    if reindex:
        build_kb(); st.rerun()
    st.caption("Spec values: 600/60, N=3, k=2, α=60. The demo defaults to 90/10 because the sample corpus is tiny "
               "(set 600/60 to see the spec configuration). Chunk changes re-index automatically.")
    st.subheader("KPIs (live chat runs)")
    h = pd.DataFrame(S["history"])
    if h.empty:
        st.info("Ask questions in Tab 1 to populate live KPIs; run Tab 5 for RCR and ER accuracy.")
    else:
        agg = h.groupby("engine")[["faithfulness", "retrieval_ms", "graph_ms", "llm_ms", "total_ms"]].mean().round(2)
        st.dataframe(agg, width="stretch")
    if "bench" in S:
        st.dataframe(summarise(S["bench"]), width="stretch")

# ------------------------------------------------------------------ Tab 5
with t5:
    if st.button("Run benchmark on evaluation set (data/eval_set.json)"):
        with st.spinner("Running 3 engines × eval questions…"):
            S["bench"] = run_benchmark(S["pipes"], cfg(area if "area" in S else list(PRACTICE_AREAS)[0]))
    if "bench" in S:
        df = S["bench"]
        sm = summarise(df)
        st.dataframe(sm, width="stretch")
        m = st.selectbox("Metric", ["RCR", "faithfulness", "total_ms"])
        by = df.groupby(["type", "engine"])[m].mean().unstack().reindex(columns=ENGINES)
        fig = go.Figure([go.Bar(name=e, x=by.index, y=by[e], marker_color=COLORS[e]) for e in ENGINES])
        fig.update_layout(barmode="group", height=360, yaxis_title=m, legend_title="Engine")
        st.plotly_chart(fig, width="stretch")
        st.caption("Offline mode uses a hashing embedder and extractive answers on a ~11-question set – "
                   "numbers show harness behaviour, not production quality. Use Claude + BGE for real runs.")
    st.subheader("Reference matrix from the architecture brief")
    st.table(pd.DataFrame({
        "Metric": ["Simple keyword queries", "Multi-hop relational queries", "Domain vocabulary mismatch",
                   "Average latency (target)", "Hallucination risk", "Best enterprise use case"],
        "Standard RAG": ["High precision / low latency", "Poor", "Poor", "~450 ms", "Moderate", "Single-document lookup (SOPs)"],
        "HyDE RAG": ["Moderate (HyDE overhead)", "Moderate", "Superior", "~1,200 ms", "Moderate (passage drift)", "Vague regulatory inquiries"],
        "Graph RAG": ["Moderate (graph overhead)", "Superior (k-hop)", "High (entity resolution)", "~850 ms", "Lowest",
                      "Complex Tax / M&A due diligence (completed – source text was truncated)"]}))
