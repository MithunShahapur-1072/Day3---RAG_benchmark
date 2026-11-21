# Standard vs HyDE vs Graph RAG – PwC India Advisory prototype

LangChain + LangGraph implementation of the architecture brief.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; without it an offline extractive LLM is used
.venv/bin/streamlit run app.py      # 5-tab UI
.venv/bin/python benchmark.py       # CLI KPI table
.venv/bin/python -m pytest tests
```

| Brief section | Where |
|---|---|
| CO-STAR prompt + 3 practice-area guardrails | `pwc_rag/config.py` |
| Steps 1-6 (ingest, chunk, normalise, embed, dual index, stores) | `pwc_rag/kb.py` |
| Standard / HyDE / Graph RAG LangGraph graphs | `pwc_rag/pipelines.py` |
| 4 hyperparameters / 4 KPIs | `config.Hyperparams`, `pwc_rag/metrics.py` |
| 5 UI tabs | `app.py` |

## Deliberate prototype simplifications
- **Stores:** in-memory (`InMemoryVectorStore`, NetworkX) instead of Qdrant/Chroma HNSW and Neo4j. Interfaces are isolated in `KnowledgeBase` for swapping.
- **Embeddings:** offline 1024-d hashing embedder by default; `PWC_EMBEDDINGS=bge` (+ `pip install sentence-transformers`) switches to BAAI/bge-large-en-v1.5 and its sub-word tokenizer.
- **Loaders:** `pypdf` / `python-docx` rather than `UnstructuredPDFLoader` (lighter install).
- **Entity extraction:** glossary + regex rules always; Claude triple extraction added when a key is set.
- **Node2Vec:** uniform random walks + PPMI/SVD (DeepWalk-style, p=q=1); used in `kb.node_vec`, not yet in scoring.
- **Rerank:** cosine + lexical + graph-evidence blend, not a cross-encoder.
- **Faithfulness:** lexical-overlap heuristic; swap in an LLM judge for real evaluation.
- **3D graph:** Plotly cannot click-select 3D nodes, so entity/relation deselection is via pickers.
- **Corpus/eval:** sample documents are synthetic. The demo uses 90-token chunks because 600-token chunks make every sample file one chunk (set 600/60 in Tab 4 for spec values).
- The brief's last table row was cut off; "Complex Tax/M&A Due Diligence" is my completion.
