from pathlib import Path

from pwc_rag.config import PRACTICE_AREAS, demo_hyperparams, system_prompt
from pwc_rag.evaluation import ENGINES, run_benchmark
from pwc_rag.kb import KnowledgeBase, load_sample_corpus, normalise
from pwc_rag.llm import LLM
from pwc_rag.pipelines import Pipelines, RunConfig, rrf

ROOT = Path(__file__).parent.parent


def _kb():
    llm, hp = LLM(force_offline=True), demo_hyperparams()
    return KnowledgeBase(hp, llm).build(load_sample_corpus(ROOT / "data" / "sample")), llm, hp


def test_alias_resolution():
    assert normalise("Income Tax Dept") == "CBDT" and normalise("Tax Authority") == "CBDT"


def test_rrf_alpha():
    assert rrf([["a", "b"], ["b", "a"]], 60)[0] in {"a", "b"}
    assert rrf([["a"], ["a"], ["b"]], 60)[0] == "a"


def test_prompt_has_guardrail():
    for a in PRACTICE_AREAS:
        assert PRACTICE_AREAS[a]["guardrail"] in system_prompt(a)


def test_all_engines_cite_and_respect_area():
    kb, llm, hp = _kb()
    p = Pipelines(kb, llm)
    for e in ENGINES:
        r = p.run(e, "What is the sellers' indemnity cap?", RunConfig(hp=hp, area=list(PRACTICE_AREAS)[2]))
        assert r["sources"] and all(kb.chunks[s["id"]].metadata["practice_area"] == list(PRACTICE_AREAS)[2]
                                    for s in r["sources"])


def test_graph_toggles_and_disabled_entity():
    kb, llm, hp = _kb()
    p = Pipelines(kb, llm)
    cfg = RunConfig(hp=hp, area=list(PRACTICE_AREAS)[1], disabled_entities={"PMLA"})
    r = p.run("Graph RAG", "penalty under PMLA", cfg)
    assert all("PMLA" not in (a, b) for a, _, b in r["edges"])
    cfg2 = RunConfig(hp=hp, area=list(PRACTICE_AREAS)[1], nodes_on={"hybrid_search"})
    assert "rerank (off)" in p.run("Graph RAG", "penalty under PMLA", cfg2)["trace"]


def test_benchmark_runs():
    kb, llm, hp = _kb()
    df = run_benchmark(Pipelines(kb, llm), RunConfig(hp=hp))
    assert set(df.engine) == set(ENGINES) and df.RCR.between(0, 1).all()
