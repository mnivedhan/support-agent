"""Retrieval strategies, so each Stage 1b fix can be switched on separately and measured.

LADDER adds one fix per rung, so the eval shows what each fix contributed (an ablation). Rungs are cumulative,
which means a fix gets credit only for what the earlier ones hadn't already solved.
"""

from dataclasses import dataclass
from functools import cache

from ssa.case_context import CaseContext, build_case_context
from ssa.data import load_knowledge, split_front_matter
from ssa.rag.chunking import Chunk, fixed_size_chunks, heading_chunks
from ssa.rag.index import Hit, HybridIndex, VectorIndex, ranked_docs

MESSAGE_K = 4   # chunks for the customer's message
FACT_K = 2      # chunks for each case fact query


@dataclass(frozen=True)
class Strategy:
    name: str
    chunker: str = "fixed"        # "fixed" | "heading"
    search: str = "vector"        # "vector" | "hybrid"
    case_context: bool = False    # add one query per database fact
    scope_filter: bool = False    # drop docs whose applies_to doesn't match the order
    pin_sop: bool = False         # always put the SOP in the prompt instead of hoping retrieval finds it


LADDER = [
    Strategy("stage1"),
    Strategy("+heading", "heading"),
    Strategy("+hybrid", "heading", "hybrid"),
    Strategy("+context", "heading", "hybrid", case_context=True),
    Strategy("+filter", "heading", "hybrid", case_context=True, scope_filter=True),
    Strategy("+pin-sop", "heading", "hybrid", case_context=True, scope_filter=True, pin_sop=True),
]
BEST = LADDER[-1]


@dataclass
class Retrieval:
    hits: list[Hit]
    pinned: list[str]     # doc ids put in the prompt whole, without retrieval
    queries: list[str]
    context: CaseContext

    @property
    def docs(self) -> list[str]:
        return list(dict.fromkeys(self.pinned + ranked_docs(self.hits)))

    @property
    def context_chars(self) -> int:
        knowledge = load_knowledge()
        return sum(len(h.chunk.text) for h in self.hits) + sum(len(knowledge[d]) for d in self.pinned)


@cache
def doc_meta() -> dict[str, dict[str, str]]:
    return {doc_id: split_front_matter(text)[0] for doc_id, text in load_knowledge().items()}


@cache
def get_index(chunker: str, search: str) -> VectorIndex | HybridIndex:
    cut = heading_chunks if chunker == "heading" else fixed_size_chunks
    chunks = [c for doc_id, text in load_knowledge().items() for c in cut(doc_id, text)]
    return HybridIndex(chunks) if search == "hybrid" else VectorIndex(chunks)


def retrieve(strategy: Strategy, message: dict, store: dict) -> Retrieval:
    index = get_index(strategy.chunker, strategy.search)
    ctx = build_case_context(message, store)
    pinned = [d for d, meta in doc_meta().items() if meta.get("doc_type") == "sop"] if strategy.pin_sop else []

    def allow(chunk: Chunk) -> bool:
        if chunk.doc_id in pinned:
            return False   # already in the prompt whole
        scope = doc_meta()[chunk.doc_id].get("applies_to")
        # Only filter when there's an order to check against. A general question ("do you do pickup?") needs
        # the scoped articles.
        return not (strategy.scope_filter and ctx.has_order and scope and scope not in ctx.tags)

    queries = [message["text"]] + (ctx.facts if strategy.case_context else [])
    hits: dict[str, Hit] = {}
    for n, query in enumerate(queries):
        for hit in index.search(query, MESSAGE_K if n == 0 else FACT_K, allow):
            hits.setdefault(hit.chunk.id, hit)
    return Retrieval(list(hits.values()), pinned, queries, ctx)
