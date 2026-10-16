"""Retrieval eval: does search put the right policy in front of the model? No LLM involved, so it's free and fast.

hit@k    share of queries where at least one expected doc is in the top-k chunks
recall@k share of all expected docs that made it into the top-k (matters when a question needs two policies)
MRR      mean of 1/rank of the first expected doc (1.0 = always ranked first)

Run:  python -m ssa.rag_eval            # k=4
      python -m ssa.rag_eval 8
"""

import json
import sys

from ssa.data import DATA_DIR
from ssa.rag.index import build_index, ranked_docs


def main(k: int) -> None:
    index = build_index()
    print(f"{len(index.chunks)} chunks embedded in {index.build_s}s\n")
    queries = [json.loads(line) for line in (DATA_DIR / "evals" / "retrieval.jsonl").read_text().splitlines()]

    hits = recall = rr = 0.0
    for q in queries:
        docs = ranked_docs(index.search(q["query"], k))
        expected = set(q["expected_docs"])
        found = expected & set(docs)
        first = next((rank for rank, d in enumerate(docs, 1) if d in expected), None)
        hits += bool(found)
        recall += len(found) / len(expected)
        rr += 1 / first if first else 0
        mark = "✓" if found == expected else ("~" if found else "✗")
        print(f"{mark} {q['query']}\n    expected {sorted(expected)}  got {docs}")

    n = len(queries)
    print(f"\nhit@{k} {hits / n:.2f}   recall@{k} {recall / n:.2f}   MRR {rr / n:.2f}   ({n} queries)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
