"""Steps 1-6: ingestion, entity-aware chunking, normalisation, embedding, dual index, hybrid stores."""
from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .config import Hyperparams
from .embeddings import get_embeddings, get_token_counter, tokenize_words
from .llm import LLM

# ---------------------------------------------------------------- Step 1: ingestion
def load_bytes(name: str, data: bytes, area: str) -> list[Document]:
    ext = Path(name).suffix.lower()
    meta = {"source": name, "practice_area": area}
    if ext == ".pdf":
        from pypdf import PdfReader
        return [Document(page_content=p.extract_text() or "", metadata={**meta, "page": i + 1})
                for i, p in enumerate(PdfReader(io.BytesIO(data)).pages)]
    if ext == ".docx":
        import docx
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        return [Document(page_content=text, metadata={**meta, "page": 1})]
    text = data.decode("utf-8", errors="ignore")
    if ext == ".json":
        text = json.dumps(json.loads(text), indent=1)  # tabular/JSON tax tables -> readable lines
    return [Document(page_content=text, metadata={**meta, "page": 1})]


AREA_BY_PREFIX = {"tax": "Tax & Regulatory Services", "grc": "Risk Advisory & Governance (GRC)",
                  "deal": "Deal Advisory & Corporate Finance"}


def load_sample_corpus(root: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(root).glob("*.txt")):
        docs += load_bytes(p.name, p.read_bytes(), AREA_BY_PREFIX[p.name.split("_")[0]])
    return docs


# ---------------------------------------------------------------- Step 3: entity normalisation
ALIASES: dict[str, list[str]] = {
    "CBDT": ["cbdt", "central board of direct taxes", "income tax dept", "income tax department",
             "tax authority"],
    "CBIC": ["cbic", "central board of indirect taxes and customs"],
    "GST Council": ["gst council"],
    "RBI": ["rbi", "reserve bank of india", "the reserve bank", "the bank"],
    "SEBI": ["sebi", "securities and exchange board of india"],
    "FIU-IND": ["fiu-ind", "fiu", "financial intelligence unit"],
    "Income Tax Act": ["income tax act", "income-tax act", "it act"],
    "Companies Act 2013": ["companies act 2013", "companies act"],
    "PMLA": ["pmla", "prevention of money laundering act", "pml act"],
    "FEMA": ["fema", "foreign exchange management act"],
    "FCRA": ["fcra", "foreign contribution regulation act"],
    "KYC": ["kyc", "know your customer"],
    "Input Service Distributor": ["input service distributor", "isd"],
    "Input Tax Credit": ["input tax credit", "itc"],
    "E-Invoicing": ["e-invoicing", "e-invoice", "einvoicing", "invoice registration portal", "irp"],
    "GST": ["gst", "goods and services tax"],
    "Suspicious Transaction Report": ["suspicious transaction report", "str"],
    "Beneficial Owner": ["beneficial owner", "beneficial ownership"],
    "Material Adverse Change": ["material adverse change", "mac", "material adverse effect"],
    "Indemnity": ["indemnity", "indemnities", "indemnify", "indemnification"],
    "Escrow": ["escrow", "escrow account"],
    "Share Purchase Agreement": ["share purchase agreement", "spa"],
    "BRSR": ["brsr", "business responsibility and sustainability report"],
    "ISO 31000": ["iso 31000"],
    "Internal Audit": ["internal audit", "internal audit controls"],
    "Valuation Report": ["valuation report", "registered valuer report"],
    "Registered Valuer": ["registered valuer"],
    "Foreign Direct Investment": ["foreign direct investment", "fdi"],
    "Master Direction": ["master direction", "master directions"],
    "TDS": ["tds", "tax deducted at source"],
}
ENTITY_TYPE = {**{k: "Authority" for k in ["CBDT", "CBIC", "GST Council", "RBI", "SEBI", "FIU-IND"]},
               **{k: "Statute" for k in ["Income Tax Act", "Companies Act 2013", "PMLA", "FEMA", "FCRA",
                                          "ISO 31000", "Master Direction", "BRSR"]}}
SURFACE2CANON = {s: c for c, ss in ALIASES.items() for s in ss}
_ALIAS_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SURFACE2CANON), key=len, reverse=True)) + r")\b",
                       re.I)
_PROV_RE = re.compile(r"\b(?:Section|Rule|Clause|Regulation|Paragraph)\s+\d+[A-Z]{0,3}(?:\.\d+)*(?:\(\w+\))?")
_REL = [(r"amend", "amends"), (r"issu|notif", "issues"), (r"penal|fine\b|prosecut", "penalises"),
        (r"exempt", "exempts"), (r"indemnif", "indemnifies"), (r"refer|subject to|read with", "references"),
        (r"mandat|require|shall|must|obligat", "requires"), (r"govern|under|pursuant", "governed_by"),
        (r"terminat|condition", "conditions")]


def normalise(surface: str) -> str:
    s = surface.strip()
    return SURFACE2CANON.get(s.lower(), re.sub(r"\s+", " ", s.title() if s.islower() else s))


def rule_entities(text: str) -> list[str]:
    spans = [(m.start(), normalise(m.group(0))) for m in _ALIAS_RE.finditer(text)]
    spans += [(m.start(), re.sub(r"\s+", " ", m.group(0))) for m in _PROV_RE.finditer(text)]
    return [e for _, e in sorted(spans)]


def rule_triples(text: str) -> list[tuple[str, str, str]]:
    out = []
    for sent in re.split(r"(?<=[.;])\s+|\n+", text):
        ents = list(dict.fromkeys(rule_entities(sent)))
        rel = next((r for pat, r in _REL if re.search(pat, sent, re.I)), "related_to")
        out += [(a, rel, b) for a, b in zip(ents, ents[1:])]
    return out


def entity_type(name: str) -> str:
    return ENTITY_TYPE.get(name) or ("Provision" if _PROV_RE.fullmatch(name) else "Concept")


# ---------------------------------------------------------------- KnowledgeBase
class KnowledgeBase:
    def __init__(self, hp: Hyperparams, llm: LLM, use_llm_extraction: bool = True):
        self.hp, self.llm, self.use_llm_extraction = hp, llm, use_llm_extraction
        self.chunks: dict[str, Document] = {}
        self.graph = nx.MultiDiGraph()
        self.node_vec: dict[str, np.ndarray] = {}
        self.vs: InMemoryVectorStore | None = None
        self.bm25: BM25Okapi | None = None
        self._ids: list[str] = []

    # Step 2: chunking (token-aware; keeps legal clauses together by preferring clause boundaries)
    def chunk(self, docs: list[Document]) -> list[Document]:
        sp = RecursiveCharacterTextSplitter(
            chunk_size=self.hp.chunk_size, chunk_overlap=self.hp.chunk_overlap,
            length_function=get_token_counter(),
            separators=["\n\n", "\nSection ", "\nClause ", "\nRule ", "\n", ". ", " ", ""])
        out = []
        for d in docs:
            for i, c in enumerate(sp.split_documents([d])):
                cid = f"{c.metadata['source']}#p{c.metadata.get('page', 1)}c{i}"
                out.append(Document(id=cid, page_content=c.page_content, metadata={**c.metadata, "chunk_id": cid}))
        return out

    def build(self, docs: list[Document]) -> "KnowledgeBase":
        chunks = self.chunk(docs)
        self.chunks = {c.id: c for c in chunks}
        self._ids = list(self.chunks)
        # Step 4/5a: dense embeddings + vector index; lexical index for the BM25 branch
        emb = get_embeddings()
        self.vs = InMemoryVectorStore(emb)  # brute-force cosine; swap for Qdrant/Chroma HNSW in production
        self.vs.add_documents(chunks, ids=self._ids)
        self.bm25 = BM25Okapi([tokenize_words(c.page_content) for c in chunks])
        # Step 5b: entity-relation graph
        g = nx.MultiDiGraph()
        for c in chunks:
            trip = rule_triples(c.page_content)
            if self.use_llm_extraction and self.llm.online:
                trip += [(normalise(s), r, normalise(o)) for s, r, o in self.llm.triples(c.page_content)]
            for ent in set(rule_entities(c.page_content)) | {x for t in trip for x in (t[0], t[2])}:
                if ent not in g:
                    g.add_node(ent, type=entity_type(ent), chunks=set())
                g.nodes[ent]["chunks"].add(c.id)
            for s, r, o in trip:
                if s == o:
                    continue
                if g.has_edge(s, o, key=r):
                    g[s][o][r]["chunks"].add(c.id)
                else:
                    g.add_edge(s, o, key=r, chunks={c.id})
        self.graph = g
        self._node2vec()
        return self

    def _node2vec(self, dim: int = 16, walks: int = 10, length: int = 8, seed: int = 7):
        """Node2Vec-style structural embeddings (uniform walks, p=q=1 -> DeepWalk) via PPMI + SVD."""
        g = self.graph.to_undirected()
        nodes = list(g)
        if len(nodes) < 3:
            self.node_vec = {n: np.zeros(1) for n in nodes}
            return
        idx, rng = {n: i for i, n in enumerate(nodes)}, np.random.default_rng(seed)
        M = np.zeros((len(nodes), len(nodes)))
        for n in nodes:
            for _ in range(walks):
                cur, path = n, [n]
                for _ in range(length):
                    nb = list(g[cur])
                    if not nb:
                        break
                    cur = nb[rng.integers(len(nb))]
                    path.append(cur)
                for a in range(len(path)):
                    for b in range(max(0, a - 2), min(len(path), a + 3)):
                        if a != b:
                            M[idx[path[a]], idx[path[b]]] += 1
        tot, r, c = M.sum(), M.sum(1, keepdims=True), M.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi = np.maximum(np.log(M * tot / (r @ c)), 0)
        ppmi[~np.isfinite(ppmi)] = 0
        u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        k = min(dim, len(s))
        emb = u[:, :k] * np.sqrt(s[:k])
        self.node_vec = {n: emb[idx[n]] for n in nodes}

    # ------------------------------------------------------------ retrieval primitives
    def _area_ok(self, area: str | None):
        return (lambda d: d.metadata.get("practice_area") == area) if area else (lambda d: True)

    def vector_search(self, queries: list[str], k: int, area: str | None) -> list[str]:
        """Rank fusion across several query vectors (original + HyDE passages) is done by caller."""
        emb = get_embeddings()
        res = self.vs.similarity_search_with_score_by_vector(emb.embed_query(queries[0]), k=k,
                                                             filter=self._area_ok(area))
        return [d.id for d, _ in res]

    def vector_search_multi(self, queries: list[str], k: int, area: str | None) -> list[list[str]]:
        emb = get_embeddings()
        return [[d.id for d, _ in self.vs.similarity_search_with_score_by_vector(
            emb.embed_query(q), k=k, filter=self._area_ok(area))] for q in queries]

    def bm25_search(self, query: str, k: int, area: str | None) -> list[str]:
        sc = self.bm25.get_scores(tokenize_words(query))
        ok = self._area_ok(area)
        order = [i for i in np.argsort(-sc) if sc[i] > 0 and ok(self.chunks[self._ids[i]])]
        return [self._ids[i] for i in order[:k]]

    def find_seeds(self, text: str) -> list[str]:
        found = [e for e in rule_entities(text) if e in self.graph]
        return list(dict.fromkeys(found))

    @staticmethod
    def edge_id(u: str, rel: str, v: str) -> str:
        return f"{u} -[{rel}]-> {v}"

    def graph_search(self, query: str, hops: int, k: int, area: str | None,
                     disabled_nodes: set[str] = frozenset(), disabled_edges: set[str] = frozenset(),
                     fallback_chunks: list[str] = ()) -> tuple[list[str], list[tuple[str, str, str]]]:
        """k-hop entity-relation expansion; returns ranked chunk ids and traversed edges."""
        seeds = self.find_seeds(query)
        if not seeds:  # no entity in the query: seed from entities of top vector hits
            for cid in fallback_chunks[:3]:
                seeds += [e for e in rule_entities(self.chunks[cid].page_content) if e in self.graph]
            seeds = list(dict.fromkeys(seeds))[:6]
        seeds = [s for s in seeds if s not in disabled_nodes]
        dist = {s: 0 for s in seeds}
        frontier, used = list(seeds), []
        for h in range(1, hops + 1):
            nxt = []
            for u in frontier:
                for v, key in [(v, key) for _, v, key in self.graph.out_edges(u, keys=True)] + \
                              [(v, key) for v, _, key in self.graph.in_edges(u, keys=True)]:
                    a, b = (u, v) if self.graph.has_edge(u, v, key=key) else (v, u)
                    eid = self.edge_id(a, key, b)
                    if v in disabled_nodes or eid in disabled_edges:
                        continue
                    used.append((a, key, b))
                    if v not in dist:
                        dist[v] = h
                        nxt.append(v)
            frontier = nxt
        score: dict[str, float] = defaultdict(float)
        for n, d in dist.items():
            w = 1.0 / (1 + d)
            for cid in self.graph.nodes[n]["chunks"]:
                score[cid] += w
        for a, r, b in set(used):
            w = 1.0 / (1 + min(dist.get(a, hops), dist.get(b, hops)))
            for cid in self.graph[a][b][r]["chunks"]:
                score[cid] += w
        ok = self._area_ok(area)
        ranked = [c for c, _ in sorted(score.items(), key=lambda x: -x[1]) if ok(self.chunks[c])]
        return ranked[:k], list(dict.fromkeys(used))

    # ------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "entities": self.graph.number_of_nodes(),
                "relations": self.graph.number_of_edges()}
