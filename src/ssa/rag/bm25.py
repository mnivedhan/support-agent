"""BM25 keyword search, written out in full so you can see there's no magic.

Embeddings match meaning ("parcel never came" ~ "not received") but blur exact terms: "Gold", "express",
"photos" and "USD 500" are just a few dimensions out of 768. BM25 scores exact word overlap instead:

  score(chunk) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

  tf      how often t appears in the chunk (with diminishing returns, controlled by k1)
  idf(t)  rarer terms count more: "gold" matters, "the" barely does
  b       penalises long chunks, which match more words by chance
"""

import math
import re
from collections import Counter

import numpy as np

from ssa.rag.chunking import Chunk

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.term_counts = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = np.array([sum(tc.values()) for tc in self.term_counts], dtype=np.float32)
        self.avg_len = float(self.lengths.mean())
        n = len(chunks)
        doc_freq = Counter(term for tc in self.term_counts for term in tc)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = [t for t in tokenize(query) if t in self.idf]
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, (tc, length) in enumerate(zip(self.term_counts, self.lengths)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            for t in terms:
                if tf := tc.get(t):
                    scores[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return scores
