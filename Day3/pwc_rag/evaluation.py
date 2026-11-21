"""Benchmark harness: runs the eval set across all three engines."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import entity_resolution_accuracy, faithfulness, retrieval_context_recall
from .pipelines import Pipelines, RunConfig

ENGINES = ["Standard RAG", "HyDE RAG", "Graph RAG"]
EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


def load_eval() -> dict:
    return json.loads(EVAL_PATH.read_text())


def run_benchmark(pipes: Pipelines, cfg: RunConfig) -> pd.DataFrame:
    ev, rows = load_eval(), []
    er = entity_resolution_accuracy(ev["entity_gold"])
    for item in ev["questions"]:
        c = RunConfig(hp=cfg.hp, area=item["area"], nodes_on=cfg.nodes_on, branches_on=cfg.branches_on,
                      disabled_entities=cfg.disabled_entities, disabled_relations=cfg.disabled_relations)
        for eng in ENGINES:
            r = pipes.run(eng, item["q"], c)
            texts = [s["text"] for s in r["sources"]]
            rows.append({"engine": eng, "type": item["type"], "question": item["q"],
                         "RCR": retrieval_context_recall(item["facts"], texts),
                         "faithfulness": faithfulness(r["answer"], r["sources"]),
                         "ER_accuracy": er if eng == "Graph RAG" else float("nan"),
                         **r["latency"]})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["RCR", "faithfulness", "ER_accuracy", "retrieval_ms", "graph_ms", "llm_ms", "total_ms"]
    return df.groupby("engine")[cols].mean().reindex(ENGINES).round(3)
