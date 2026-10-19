"""In-memory search indexes over a list of chunks.

VectorIndex  brute-force cosine similarity with numpy. Rebuilt on every start; fine for a few hundred chunks.
             Stage 7 replaces it with a real vector DB.
HybridIndex  vectorIndex + BM25 keyword scores, merged with reciprocal rank fusion.

Every index has scores(query) -> one number per chunk, and search(query, k, allow) -> top-k hits. `allow` is a
metadata filter applied *before* taking the top k (pre-filtering). Filtering after would return fewer than k
hits whenever filtered chunks ranked high.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ssa.data import load_knowledge
from ssa.rag.bm25 import BM25Index
from ssa.rag.chunking import Chunk, fixed_size_chunks
from ssa.rag.embeddings import embed_documents, embed_query

Allow = Callable[[Chunk], bool] | None
RRF_K = 60   # the usual constant: damps the gap between rank 1 and rank 2 so neither ranking dominates


@dataclass
class Hit:
    chunk: Chunk
    score: float


def top_k(chunks: list[Chunk], scores: np.ndarray, k: int, allow: Allow = None) -> list[Hit]:
    hits = []
    for i in np.argsort(-scores):
        if allow is None or allow(chunks[i]):
            hits.append(Hit(chunks[i], float(scores[i])))
            if len(hits) == k:
                break
    return hits


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        started = time.perf_counter()
        self.chunks = chunks
        self.vectors = embed_documents([c.text for c in chunks])
        self.build_s = round(time.perf_counter() - started, 2)

    def scores(self, query: str) -> np.ndarray:
        return self.vectors @ embed_query(query)

    def search(self, query: str, k: int = 4, allow: Allow = None) -> list[Hit]:
        return top_k(self.chunks, self.scores(query), k, allow)


class HybridIndex:
    """Reciprocal rank fusion: score = sum over both rankings of 1 / (RRF_K + rank).

    It fuses ranks, not raw scores, because cosine (about 0.3 to 0.8 here) and BM25 (0 to about 20) are on
    different scales. A chunk with no keyword overlap gets nothing from the BM25 side.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vector = VectorIndex(chunks)
        self.bm25 = BM25Index(chunks)
        self.build_s = self.vector.build_s

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.chunks))
        for scores in (self.vector.scores(query), self.bm25.scores(query)):
            ranks = np.empty(len(scores))
            ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
            fused += np.where(scores > 0, 1 / (RRF_K + ranks), 0)
        return fused

    def search(self, query: str, k: int = 4, allow: Allow = None) -> list[Hit]:
        return top_k(self.chunks, self.scores(query), k, allow)


def build_index(size: int = 400, overlap: int = 50) -> VectorIndex:
    chunks = [c for doc_id, text in load_knowledge().items() for c in fixed_size_chunks(doc_id, text, size, overlap)]
    return VectorIndex(chunks)


def ranked_docs(hits: list[Hit]) -> list[str]:
    """Distinct doc ids in the order their best chunk was ranked."""
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))
