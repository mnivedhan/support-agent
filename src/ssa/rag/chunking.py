"""Two ways to cut documents into chunks.

fixed_size_chunks: equal character windows (Stage 1). Ignores headings, sentences and front matter, so a rule can
be split in half and a chunk can lose the title that says which policy it belongs to.

heading_chunks: one chunk per markdown section (Stage 1b). Each chunk starts with "<doc title> > <heading>" so
both the embedding and the LLM know where it came from. Works here because every section is short. A real corpus
also needs to split long sections and merge tiny ones.
"""

from dataclasses import dataclass

from ssa.data import split_front_matter


@dataclass(frozen=True)
class Chunk:
    id: str      # "<doc_id>#<n>"
    doc_id: str
    text: str


def fixed_size_chunks(doc_id: str, text: str, size: int = 400, overlap: int = 50) -> list[Chunk]:
    step = size - overlap
    return [Chunk(f"{doc_id}#{n}", doc_id, text[start:start + size])
            for n, start in enumerate(range(0, max(len(text) - overlap, 1), step))]


def heading_chunks(doc_id: str, text: str) -> list[Chunk]:
    meta, body = split_front_matter(text)
    title = meta.get("title", doc_id)
    sections: list[tuple[str, list[str]]] = [("", [])]   # (heading, lines); text before any heading has none
    for line in body.splitlines():
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        content = "\n".join(lines).strip()
        if content:
            path = f"{title} > {heading}" if heading else title
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, f"{path}\n{content}"))
    return chunks
