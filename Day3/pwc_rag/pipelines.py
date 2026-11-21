"""Standard, HyDE and Graph RAG as LangGraph state graphs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import TOP_K, Hyperparams, system_prompt
from .embeddings import cosine, get_embeddings, tokenize_words
from .kb import KnowledgeBase
from .llm import LLM

WORKFLOW_NODES = ["query_expansion", "hybrid_search", "rrf_fusion", "rerank", "synthesis"]
BRANCHES = ["vector", "bm25", "graph"]
MAX_LOOPS = 1  # sufficiency check may widen the hop radius once (cyclic edge)


@dataclass
class RunConfig:
    hp: Hyperparams = field(default_factory=Hyperparams)
    area: str = "Tax & Regulatory Services"
    nodes_on: set[str] = field(default_factory=lambda: set(WORKFLOW_NODES))
    branches_on: set[str] = field(default_factory=lambda: set(BRANCHES))
    disabled_entities: set[str] = field(default_factory=set)
    disabled_relations: set[str] = field(default_factory=set)


class State(TypedDict, total=False):
    question: str
    queries: list[str]
    lists: dict[str, list[str]]
    fused: list[str]
    final: list[str]
    hops: int
    loops: int
    edges: list
    answer: str
    sources: list[dict]
    trace: list[str]
    ms: dict[str, float]


def rrf(lists: list[list[str]], alpha: int) -> list[str]:
    s: dict[str, float] = {}
    for lst in lists:
        for r, cid in enumerate(lst, 1):
            s[cid] = s.get(cid, 0.0) + 1.0 / (alpha + r)
    return sorted(s, key=lambda c: -s[c])


def _step(name: str, bucket: str, fn: Callable[[State], dict]):
    """Wrap a node: records trace + latency bucket (retrieval | graph | llm)."""
    def run(st: State) -> dict:
        t = time.perf_counter()
        upd = fn(st)
        ms = dict(st.get("ms", {}))
        ms[bucket] = ms.get(bucket, 0.0) + (time.perf_counter() - t) * 1000
        return {**upd, "ms": ms, "trace": st.get("trace", []) + [name]}
    return run


class Pipelines:
    def __init__(self, kb: KnowledgeBase, llm: LLM):
        self.kb, self.llm = kb, llm

    # ------------------------------------------------ shared synthesis
    def _synth(self, cfg: RunConfig, st: State, ids: list[str]) -> dict:
        ids = ids[:TOP_K]
        blocks = [(f"S{i}", self.kb.chunks[c].page_content) for i, c in enumerate(ids, 1)]
        ans = self.llm.answer(system_prompt(cfg.area), st["question"], blocks)
        src = [{"tag": f"S{i}", "id": c, "source": self.kb.chunks[c].metadata["source"],
                "page": self.kb.chunks[c].metadata.get("page"), "text": self.kb.chunks[c].page_content}
               for i, c in enumerate(ids, 1)]
        return {"answer": ans, "sources": src, "final": ids}

    def _finish(self, out: State, t0: float) -> dict:
        ms = out.get("ms", {})
        ms["total"] = (time.perf_counter() - t0) * 1000
        return {"answer": out["answer"], "sources": out["sources"], "trace": out["trace"],
                "latency": {"retrieval_ms": ms.get("retrieval", 0), "graph_ms": ms.get("graph", 0),
                            "llm_ms": ms.get("llm", 0), "total_ms": ms["total"]},
                "retrieved_ids": out.get("final", []), "edges": out.get("edges", [])}

    # ------------------------------------------------ Standard RAG
    def standard(self, cfg: RunConfig):
        def retrieve(st):
            return {"final": self.kb.vector_search([st["question"]], TOP_K, cfg.area)}
        g = StateGraph(State)
        g.add_node("retrieve", _step("retrieve", "retrieval", retrieve))
        g.add_node("synthesis", _step("synthesis", "llm", lambda st: self._synth(cfg, st, st["final"])))
        g.add_edge(START, "retrieve"); g.add_edge("retrieve", "synthesis"); g.add_edge("synthesis", END)
        return g.compile()

    # ------------------------------------------------ Multi-hypothesis HyDE RAG
    def hyde(self, cfg: RunConfig):
        def hypothesize(st):
            return {"queries": [st["question"]] + self.llm.hypotheses(st["question"], cfg.hp.hyde_n, cfg.area)}

        def retrieve(st):
            lists = self.kb.vector_search_multi(st["queries"], TOP_K * 2, cfg.area)
            return {"final": rrf(lists, cfg.hp.rrf_alpha)[:TOP_K]}
        g = StateGraph(State)
        g.add_node("hypothesize", _step("hypothesize", "llm", hypothesize))
        g.add_node("retrieve", _step("retrieve", "retrieval", retrieve))
        g.add_node("synthesis", _step("synthesis", "llm", lambda st: self._synth(cfg, st, st["final"])))
        g.add_edge(START, "hypothesize"); g.add_edge("hypothesize", "retrieve")
        g.add_edge("retrieve", "synthesis"); g.add_edge("synthesis", END)
        return g.compile()

    # ------------------------------------------------ Graph RAG (cyclic)
    def graph_rag(self, cfg: RunConfig):
        kb, on = self.kb, cfg.nodes_on
        lbl = lambda n: n if n in on else f"{n} (off)"

        def expand(st):
            if "query_expansion" not in on:
                return {"queries": [st["question"]]}
            return {"queries": self.llm.expand(st["question"])}

        def search_node(st):
            t0 = time.perf_counter()
            lists: dict[str, list[str]] = {}
            qs, k = st["queries"], TOP_K * 3
            if "vector" in cfg.branches_on:
                lists["vector"] = kb.vector_search_multi(qs[:1], k, cfg.area)[0]
            if "bm25" in cfg.branches_on:
                lists["bm25"] = kb.bm25_search(" ".join(qs), k, cfg.area)
            edges, g_ms = st.get("edges", []), 0.0
            if "graph" in cfg.branches_on:
                tg = time.perf_counter()
                lists["graph"], edges = kb.graph_search(
                    " ".join(qs), st["hops"], k, cfg.area, cfg.disabled_entities, cfg.disabled_relations,
                    fallback_chunks=lists.get("vector", []))
                g_ms = (time.perf_counter() - tg) * 1000
            total = (time.perf_counter() - t0) * 1000
            ms = dict(st.get("ms", {}))
            ms["graph"] = ms.get("graph", 0) + g_ms
            ms["retrieval"] = ms.get("retrieval", 0) + total - g_ms
            return {"lists": lists, "edges": edges, "ms": ms, "trace": st.get("trace", []) + ["hybrid_search"]}

        def fuse(st):
            ls = [v for v in st["lists"].values() if v]
            if "rrf_fusion" in on:
                return {"fused": rrf(ls, cfg.hp.rrf_alpha)}
            seen = list(dict.fromkeys(c for l in ls for c in l))  # plain concatenation, no fusion
            return {"fused": seen}

        def rerank(st):
            if "rerank" not in on:
                return {"final": st["fused"][:TOP_K]}
            emb = get_embeddings()
            q = emb.embed_query(st["question"])
            qt = set(tokenize_words(st["question"]))
            pool = st["fused"][:TOP_K * 3]
            vecs = emb.embed_documents([kb.chunks[c].page_content for c in pool])
            graph_set = set(st["lists"].get("graph", []))
            def score(c, v):
                lex = len(qt & set(tokenize_words(kb.chunks[c].page_content))) / (len(qt) or 1)
                return 0.5 * cosine(q, v) + 0.3 * lex + (0.2 if c in graph_set else 0)
            order = sorted(zip(pool, vecs), key=lambda x: -score(*x))
            return {"final": [c for c, _ in order][:TOP_K]}

        def sufficiency(st):  # cyclic edge: widen graph radius when no graph evidence made the cut
            if ("graph" in cfg.branches_on and st.get("loops", 0) < MAX_LOOPS
                    and not set(st["final"]) & set(st["lists"].get("graph", []))):
                return "widen"
            return "done"

        def widen(st):
            return {"hops": st["hops"] + 1, "loops": st.get("loops", 0) + 1}

        def synth(st):
            if "synthesis" not in on:
                ids = st["final"][:TOP_K]
                upd = self._synth(cfg, st, ids)
                upd["answer"] = "*Synthesis node disabled – raw retrieved passages only.*"
                return upd
            return self._synth(cfg, st, st["final"])

        g = StateGraph(State)
        g.add_node("query_expansion", _step(lbl("query_expansion"), "llm", expand))
        g.add_node("hybrid_search", search_node)
        g.add_node("rrf_fusion", _step(lbl("rrf_fusion"), "retrieval", fuse))
        g.add_node("rerank", _step(lbl("rerank"), "retrieval", rerank))
        g.add_node("widen_hops", _step("widen_hops", "graph", widen))
        g.add_node("synthesis", _step(lbl("synthesis"), "llm", synth))
        g.add_edge(START, "query_expansion"); g.add_edge("query_expansion", "hybrid_search")
        g.add_edge("hybrid_search", "rrf_fusion"); g.add_edge("rrf_fusion", "rerank")
        g.add_conditional_edges("rerank", sufficiency, {"widen": "widen_hops", "done": "synthesis"})
        g.add_edge("widen_hops", "hybrid_search"); g.add_edge("synthesis", END)
        return g.compile()

    # ------------------------------------------------ entry point
    def run(self, engine: str, question: str, cfg: RunConfig) -> dict:
        app = {"Standard RAG": self.standard, "HyDE RAG": self.hyde, "Graph RAG": self.graph_rag}[engine](cfg)
        t0 = time.perf_counter()
        init: State = {"question": question, "hops": cfg.hp.hop_radius, "loops": 0, "trace": [], "ms": {},
                       "edges": []}
        out = app.invoke(init)
        return self._finish(out, t0)

    def workflow_edges(self) -> list[tuple[str, str]]:
        return [("query_expansion", "hybrid_search"), ("hybrid_search", "rrf_fusion"),
                ("rrf_fusion", "rerank"), ("rerank", "synthesis"), ("rerank", "hybrid_search")]
