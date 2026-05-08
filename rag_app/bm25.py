from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag_app.documents import ChunkDocument


TOKEN_RE = re.compile(r"[\wёЁ]+", re.UNICODE)


@dataclass(frozen=True)
class SparseHit:
    doc: ChunkDocument
    score: float


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class BM25Index:
    def __init__(self, docs: list[ChunkDocument]):
        self._docs = docs
        corpus = [tokenize(doc.searchable_text) for doc in docs]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, limit: int) -> list[SparseHit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            SparseHit(doc=self._docs[index], score=float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]

