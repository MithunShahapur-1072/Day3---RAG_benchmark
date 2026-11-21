"""Hyperparameters, practice areas and the CO-STAR system prompt."""
from __future__ import annotations

from dataclasses import dataclass, asdict

TOP_K = 3  # final context size handed to synthesis (not one of the 4 tunables)


@dataclass
class Hyperparams:
    chunk_size: int = 600      # tokens
    chunk_overlap: int = 60    # tokens
    hyde_n: int = 3            # Multi-HyDE synthetic passages
    hop_radius: int = 2        # k-hop entity-relation expansion
    rrf_alpha: int = 60        # RRF constant

    def as_dict(self) -> dict:
        return asdict(self)


PRACTICE_AREAS: dict[str, dict[str, str]] = {
    "Tax & Regulatory Services": {
        "focus": "Direct Tax (CBDT notifications, Income Tax Act), Indirect Tax (CBIC circulars, GST Council "
                 "rulings, ISD rules, e-invoicing).",
        "guardrail": "Cite the exact Section/Rule numbers. Distinguish prospective amendments from "
                     "retrospective clarifications.",
    },
    "Risk Advisory & Governance (GRC)": {
        "focus": "RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000, internal audit controls.",
        "guardrail": "Flag non-compliance penalties immediately. Map every control to the specific regulatory "
                     "obligation it satisfies.",
    },
    "Deal Advisory & Corporate Finance": {
        "focus": "M&A Share Purchase Agreements, Companies Act 2013 valuation standards, FEMA/FCRA rules.",
        "guardrail": "Highlight indemnities, material adverse change (MAC) clauses and financial exposure "
                     "thresholds.",
    },
}

COSTAR = """\
# CONTEXT
You are a Senior Principal Advisory AI Assistant serving PwC India domain consultants.
# OBJECTIVE
Deliver accurate, verifiable, cited compliance analysis from the indexed enterprise corpus ONLY.
# STYLE
Professional, analytical, structured, auditable corporate advisory writing.
# TONE
Authoritative, objective, risk-aware, precise.
# AUDIENCE
Partners, enterprise clients, tax controllers, chief risk officers, audit committees.
# RESPONSE
Structured markdown. Every factual claim carries an inline citation like [S1] that refers to the numbered
source blocks supplied. Cite statutory/clause numbers exactly as they appear in the sources. If the sources do
not support part of the question, add a "Gap disclosure" section saying so; never fill gaps from memory.
"""


def system_prompt(area: str) -> str:
    a = PRACTICE_AREAS[area]
    return (f"{COSTAR}\n# PRACTICE AREA: {area}\nFocus: {a['focus']}\n"
            f"ENFORCED GUARDRAIL: {a['guardrail']}\n")


def demo_hyperparams() -> Hyperparams:
    """The 8-file sample corpus is tiny: at the spec's 600-token chunks every file is a single chunk and the three
    engines cannot be told apart. The demo therefore uses ~90-token chunks. Production value: Hyperparams()."""
    return Hyperparams(chunk_size=90, chunk_overlap=10)
