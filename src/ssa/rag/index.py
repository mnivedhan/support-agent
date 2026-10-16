"""In-memory vector index: brute-force cosine similarity over every chunk with numpy.

Rebuilt from scratch on every start. Fine for a few hundred chunks; Stage 7 replaces it with a real vector DB.
"""

import time
from dataclasses import dataclass

import numpy as np

from ssa.data import load_knowledge
from ssa.rag.chunking import Chunk, fixed_size_chunks
from ssa.rag.embeddings import embed_documents, embed_query


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        started = time.perf_counter()
        self.chunks = chunks
        self.vectors = embed_documents([c.text for c in chunks])
        self.build_s = round(time.perf_counter() - started, 2)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = self.vectors @ embed_query(query)
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]


def build_index(size: int = 400, overlap: int = 50) -> VectorIndex:
    chunks = [c for doc_id, text in load_knowledge().items() for c in fixed_size_chunks(doc_id, text, size, overlap)]
    return VectorIndex(chunks)


def ranked_docs(hits: list[Hit]) -> list[str]:
    """Distinct doc ids in the order their best chunk was ranked."""
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))
