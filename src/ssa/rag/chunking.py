"""Naive fixed-size chunking: cut every document into equal character windows.

Deliberately ignores headings, sentences and front matter, so a rule can be split in half and a chunk can lose
the title that says which policy it belongs to.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str      # "<doc_id>#<n>"
    doc_id: str
    text: str


def fixed_size_chunks(doc_id: str, text: str, size: int = 400, overlap: int = 50) -> list[Chunk]:
    step = size - overlap
    return [Chunk(f"{doc_id}#{n}", doc_id, text[start:start + size])
            for n, start in enumerate(range(0, max(len(text) - overlap, 1), step))]
