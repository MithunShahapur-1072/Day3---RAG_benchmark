"""LLM facade: Claude via langchain-anthropic when ANTHROPIC_API_KEY is set, else a deterministic offline
fallback (extractive answers, templated HyDE passages) so the whole pipeline is runnable without a key."""
from __future__ import annotations

import json
import os
import re

from .embeddings import tokenize_words

MODEL = os.getenv("PWC_MODEL", "claude-sonnet-5-5")
_CITE = re.compile(r"\[S\d+\]")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+|\n+", text) if len(s.strip()) > 25]


class LLM:
    def __init__(self, force_offline: bool = False):
        self.online = bool(os.getenv("ANTHROPIC_API_KEY")) and not force_offline
        self._chat = None
        if self.online:
            from langchain_anthropic import ChatAnthropic
            self._chat = ChatAnthropic(model=MODEL, temperature=0, max_tokens=1500)

    @property
    def name(self) -> str:
        return MODEL if self.online else "offline-extractive (set ANTHROPIC_API_KEY for Claude)"

    def _call(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        return self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)]).text

    # ---- synthesis -------------------------------------------------------------------------------
    def answer(self, system: str, question: str, blocks: list[tuple[str, str]]) -> str:
        """blocks = [(tag, text)], tag like 'S1'."""
        if not blocks:
            return "**Gap disclosure:** no indexed source supports this question for the selected practice area."
        if self.online:
            ctx = "\n\n".join(f"[{t}] {x}" for t, x in blocks)
            return self._call(system, f"SOURCES:\n{ctx}\n\nQUESTION: {question}")
        q = set(tokenize_words(question))
        scored = []
        for tag, text in blocks:
            for s in _sentences(text):
                w = set(tokenize_words(s))
                scored.append((len(q & w) / (len(q) or 1), tag, s))
        scored.sort(key=lambda x: -x[0])
        picked = [x for x in scored[:5] if x[0] > 0.08]
        if not picked:
            return "**Gap disclosure:** retrieved sources do not directly address the question."
        body = "\n".join(f"- {s} [{tag}]" for _, tag, s in picked)
        return f"**Findings (extractive, offline mode)**\n{body}"

    # ---- HyDE ------------------------------------------------------------------------------------
    def hypotheses(self, question: str, n: int, area: str) -> list[str]:
        angles = ["the statutory/regulatory provision text and its section or rule number",
                  "the compliance obligations, thresholds, penalties and consequences",
                  "how the provision interacts with related rules, authorities and definitions",
                  "practical application, exceptions and prospective vs retrospective effect"]
        if self.online:
            outs = []
            for a in angles[:n]:
                outs.append(self._call(
                    "You write short hypothetical passages (max 120 words) as they would appear in an Indian "
                    f"regulatory or contractual document ({area}). No preamble.",
                    f"Question: {question}\nWrite the passage focusing on {a}."))
            return outs
        return [f"{question} This provision addresses {a}. Section Rule Clause obligation penalty authority "
                f"notification {area}." for a in angles[:n]]

    # ---- query expansion (Graph RAG) -------------------------------------------------------------
    def expand(self, question: str) -> list[str]:
        if self.online:
            try:
                out = self._call("Return a JSON list of up to 3 alternative search queries using formal "
                                 "Indian regulatory vocabulary. JSON only.", question)
                lst = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
                return [question] + [str(x) for x in lst][:3]
            except Exception:
                pass
        return [question]

    # ---- entity/relation extraction --------------------------------------------------------------
    def triples(self, text: str) -> list[tuple[str, str, str]]:
        if not self.online:
            return []
        try:
            out = self._call("Extract regulatory entities and relations as a JSON list of "
                             "[subject, relation, object] triples (max 12). Use short canonical names "
                             "(e.g. 'CBDT', 'Section 12 PMLA'). JSON only.", text[:4000])
            lst = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
            return [tuple(map(str, t)) for t in lst if len(t) == 3]
        except Exception:
            return []


def cited_tags(answer: str) -> set[str]:
    return {m.strip("[]") for m in _CITE.findall(answer)}
