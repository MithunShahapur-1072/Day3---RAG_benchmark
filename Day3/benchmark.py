"""CLI: python benchmark.py  -> prints KPI table for Standard / HyDE / Graph RAG."""
from pathlib import Path

import pandas as pd

from pwc_rag.config import demo_hyperparams
from pwc_rag.evaluation import run_benchmark, summarise
from pwc_rag.kb import KnowledgeBase, load_sample_corpus
from pwc_rag.llm import LLM
from pwc_rag.pipelines import Pipelines, RunConfig

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    llm, hp = LLM(), demo_hyperparams()
    kb = KnowledgeBase(hp, llm).build(load_sample_corpus(Path(__file__).parent / "data" / "sample"))
    print("LLM:", llm.name, "| KB:", kb.stats())
    df = run_benchmark(Pipelines(kb, llm), RunConfig(hp=hp))
    print(summarise(df))
    print(df.groupby(["type", "engine"])["RCR"].mean().unstack().round(2))
