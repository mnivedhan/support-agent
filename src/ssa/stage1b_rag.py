"""Stage 1b: Stage 1 with the retrieval fixes (the last rung of rag_eval's ladder).

- heading-aware chunks, titled "<doc> > <section>"
- hybrid search: vector + BM25, fused by rank
- one extra query per database fact (tier, refunds in 90 days, days late, card mismatch, ...)
- scoped help articles filtered out when the order doesn't match their applies_to
- the SOP pinned in the prompt whole, never retrieved

The content is otherwise the same as Stage 1, so any change in answers comes from retrieval. The layout differs:
retrieved sections go in the user message so the system prompt stays identical and cacheable. The facts only drive
retrieval; they are not shown to the model. The store database still goes in whole (Stage 2 replaces that with
tools).

Run:  python -m ssa.stage1b_rag                     # all 7 scenarios
      python -m ssa.stage1b_rag gold-late-delivery  # just one
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime

from ssa.data import load_knowledge, load_scenarios, load_store
from ssa.llm import complete
from ssa.rag.retriever import BEST, Retrieval, retrieve
from ssa.stage0_baseline import INSTRUCTIONS, RUNS_DIR, format_message


def build_system_prompt(pinned: list[str], store: dict) -> str:
    """Identical on every call, so the prompt cache can reuse all of it. The cache only hits at breakpoints (the
    CLI sets one at the end of the system prompt), so per-request content anywhere in here defeats it."""
    knowledge = load_knowledge()
    docs = "\n\n".join(f'<doc id="{d}">\n{knowledge[d]}\n</doc>' for d in pinned)
    return (f"{INSTRUCTIONS}\n\n<procedures>\n{docs}\n</procedures>\n\n"
            f"<store_database>\n{json.dumps(store, indent=2)}\n</store_database>")


def build_user_message(retrieval: Retrieval, message: dict) -> str:
    """The per-request part: retrieved policy sections, then the customer message."""
    chunks = "\n\n".join(f'<chunk doc="{h.chunk.doc_id}">\n{h.chunk.text}\n</chunk>' for h in retrieval.hits)
    return f"<policies>\n{chunks}\n</policies>\n\n{format_message(message)}"


def main(names: list[str]) -> None:
    store = load_store()
    scenarios = [s for s in load_scenarios() if not names or s["name"] in names]

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"stage1b-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    results = []
    with out_path.open("w") as out:
        for scenario in scenarios:
            retrieval = retrieve(BEST, scenario["message"], store)
            missing = [d for d in scenario["expect"]["required_docs"] if d not in retrieval.docs]
            print(f"\n{'=' * 100}\n{scenario['name']}: {scenario['message']['text']}")
            print(f"context {retrieval.docs}" + (f"   MISSING {missing}" if missing else ""))
            print("-" * 100)

            result = complete(build_system_prompt(retrieval.pinned, store),
                              build_user_message(retrieval, scenario["message"]))
            print(result.text)
            row = {"name": scenario["name"], "expect": scenario["expect"], "retrieved": retrieval.docs,
                   "queries": retrieval.queries, "missing": missing, "answer": result.text,
                   **{k: v for k, v in asdict(result).items() if k != "text"}}
            out.write(json.dumps(row) + "\n")
            results.append(row)

    print(f"\n{'scenario':<24}{'in tok':>8}{'cached':>8}{'out tok':>9}{'latency':>9}  {'missing docs':<34}expected")
    for r in results:
        e = r["expect"]
        print(f"{r['name']:<24}{r['input_tokens']:>8}{r['cache_read_tokens']:>8}{r['output_tokens']:>9}"
              f"{r['latency_s']:>8}s  {', '.join(r['missing']) or '-':<34}{e['final_status']}, "
              f"${e['case_value']:.2f}, approval={e['needs_approval']}")
    print(f"\ntotal ${sum(r['cost_usd'] for r in results):.4f} (list-price equivalent)   saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
