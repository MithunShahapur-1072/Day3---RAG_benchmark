"""Embeddings + token counting.

Default is an offline 1024-dim hashing embedder so the prototype runs anywhere.
Set PWC_EMBEDDINGS=bge to use BAAI/bge-large-en-v1.5 (1024-dim) via sentence-transformers; the same
HuggingFace sub-word tokenizer is then used for chunk-size accounting (Step 3).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

DIM = 1024
_WORD = re.compile(r"\w+|[^\w\s]")


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = DIM):
        self._v = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False,
                                    norm="l2", lowercase=True, stop_words="english")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._v.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class BGEEmbeddings(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed_documents(self, texts):
        return self.m.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        q = "Represent this sentence for searching relevant passages: " + text
        return self.m.encode([q], normalize_embeddings=True)[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return BGEEmbeddings() if os.getenv("PWC_EMBEDDINGS") == "bge" else HashingEmbeddings()


@lru_cache(maxsize=1)
def get_token_counter():
    if os.getenv("PWC_EMBEDDINGS") == "bge":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        return lambda t: len(tok.tokenize(t))
    return lambda t: len(_WORD.findall(t))


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0
