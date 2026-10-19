"""Retrieval eval: does search put the right policies in front of the model? No LLM, so it's free and fast.

Two query sets:
  cases      real-looking customer messages: the 7 scenarios + data/evals/retrieval_cases.jsonl. They carry
             customer and order ids, so the case-context strategies can use the database. This is production.
  questions  data/evals/retrieval.jsonl: policy questions written by someone who knew the answer (Stage 1's eval).
             No customer or order, so case context adds nothing there.

Metrics per strategy:
  covered      share of cases where *every* required doc was in context. The one that matters: one missing rule
               is enough for a wrong answer.
  recall       share of required docs found, averaged over cases
  precision    share of docs in context that were required. Extra docs cost tokens and can mislead.
  wrong-scope  docs whose applies_to doesn't match the order (marketplace rules for a Brightcart order), per case
  tokens       context size (chars / 4), pinned docs included
  ms           retrieval time per case, query embeddings included

There's no MRR any more: once a strategy runs several queries and merges them, there's no single ranking.

Run:  python -m ssa.rag_eval            # summary for every strategy in the ladder
      python -m ssa.rag_eval +filter    # plus per-case detail for one strategy
"""

import json
import sys
import time

from ssa.data import DATA_DIR, load_scenarios, load_store
from ssa.rag.retriever import LADDER, Strategy, doc_meta, get_index, retrieve


def load_cases() -> list[dict]:
    scenarios = [{"name": s["name"], "message": s["message"], "required_docs": s["expect"]["required_docs"]}
                 for s in load_scenarios()]
    extra = [json.loads(line) for line in (DATA_DIR / "evals" / "retrieval_cases.jsonl").read_text().splitlines()]
    return scenarios + extra


def load_questions() -> list[dict]:
    rows = [json.loads(line) for line in (DATA_DIR / "evals" / "retrieval.jsonl").read_text().splitlines()]
    return [{"name": q["query"], "required_docs": q["expected_docs"],
             "message": {"text": q["query"], "customer_id": None, "order_id": None, "attachments": []}}
            for q in rows]


def evaluate(strategy: Strategy, cases: list[dict], store: dict) -> tuple[dict, list[dict]]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        r = retrieve(strategy, case["message"], store)
        ms = (time.perf_counter() - started) * 1000
        required = set(case["required_docs"])
        found = required & set(r.docs)
        wrong = [d for d in r.docs if r.context.has_order
                 and (scope := doc_meta()[d].get("applies_to")) and scope not in r.context.tags]
        rows.append({"name": case["name"], "retrieval": r, "missing": sorted(required - found), "wrong": wrong,
                     "covered": found == required, "recall": len(found) / len(required),
                     "precision": len(found) / len(r.docs), "tokens": r.context_chars / 4, "ms": ms})
    summary = {key: sum(row[key] if key != "wrong" else len(row[key]) for row in rows) / len(rows)
               for key in ("covered", "recall", "precision", "wrong", "tokens", "ms")}
    return summary, rows


def main(detail: str | None) -> None:
    store = load_store()
    cases, questions = load_cases(), load_questions()
    for chunker, search in dict.fromkeys((s.chunker, s.search) for s in LADDER):
        index = get_index(chunker, search)   # build outside the timed loop
        print(f"index {chunker}/{search}: {len(index.chunks)} chunks, built in {index.build_s}s")

    print(f"\n{'':<10}{'---------- customer cases (' + str(len(cases)) + ') ----------':^58}"
          f"  {'-- questions (' + str(len(questions)) + ') --':^20}")
    print(f"{'strategy':<10}{'covered':>9}{'recall':>8}{'precision':>11}{'wrong-scope':>13}{'tokens':>8}{'ms':>7}"
          f"  {'covered':>9}{'recall':>8}")
    details = {}
    for s in LADDER:
        c, rows = evaluate(s, cases, store)
        q, _ = evaluate(s, questions, store)
        details[s.name] = rows
        print(f"{s.name:<10}{c['covered']:>9.2f}{c['recall']:>8.2f}{c['precision']:>11.2f}{c['wrong']:>13.2f}"
              f"{c['tokens']:>8.0f}{c['ms']:>7.0f}  {q['covered']:>9.2f}{q['recall']:>8.2f}")

    if detail:
        print(f"\nper-case detail for {detail}:")
        for row in details[detail]:
            r = row["retrieval"]
            print(f"\n{'✓' if row['covered'] else '✗'} {row['name']}   docs {r.docs}")
            if row["missing"]:
                print(f"    MISSING {row['missing']}")
            if row["wrong"]:
                print(f"    WRONG SCOPE {row['wrong']}")
            for query in r.queries[1:]:
                print(f"    fact query: {query}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
