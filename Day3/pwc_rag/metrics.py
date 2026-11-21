"""The 4 KPIs: Retrieval Context Recall, Faithfulness, Entity-Relation Resolution Accuracy, Latency."""
from __future__ import annotations

import re

from .embeddings import tokenize_words
from .kb import normalise
from .llm import cited_tags


def retrieval_context_recall(facts: list[str], retrieved_texts: list[str]) -> float:
    """Share of ground-truth fact phrases present (case-insensitive, whitespace-normalised) in retrieved chunks."""
    blob = re.sub(r"\s+", " ", " ".join(retrieved_texts).lower())
    return sum(f.lower() in blob for f in facts) / len(facts) if facts else 0.0


def faithfulness(answer: str, sources: list[dict]) -> float:
    """Heuristic: fraction of claim sentences that cite a valid [S#] AND lexically overlap (>=50% of content
    words) with the cited source. Replace with an LLM judge for production use."""
    by_tag = {s["tag"]: set(tokenize_words(s["text"])) for s in sources}
    claims = [s for s in re.split(r"(?<=[.;])\s+(?!\[S\d+\])|\n+", answer) if cited_tags(s)]
    if not claims:
        return 0.0
    ok = 0
    for c in claims:
        words = set(tokenize_words(re.sub(r"\[S\d+\]", "", c))) - {"the", "a", "of", "to", "and", "in", "is"}
        tags = [t for t in cited_tags(c) if t in by_tag]
        if tags and words and max(len(words & by_tag[t]) / len(words) for t in tags) >= 0.5:
            ok += 1
    return ok / len(claims)


def entity_resolution_accuracy(gold: dict[str, str]) -> float:
    return sum(normalise(k) == v for k, v in gold.items()) / len(gold) if gold else 0.0
