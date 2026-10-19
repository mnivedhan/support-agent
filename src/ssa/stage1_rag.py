"""Stage 1: simple RAG. Same as Stage 0, except the prompt holds only the top-k policy chunks for the message.

Retrieval query = the customer's message text, nothing else. The store database still goes in whole; looking
up only the relevant order and customer is Stage 2's job (tools).

Run:  python -m ssa.stage1_rag                     # all 7 scenarios
      python -m ssa.stage1_rag gold-late-delivery  # just one
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime

from ssa.data import load_scenarios, load_store
from ssa.llm import complete
from ssa.rag.index import build_index, ranked_docs
from ssa.stage0_baseline import INSTRUCTIONS, RUNS_DIR, format_message

TOP_K = 4


def build_system_prompt(hits) -> str:
    chunks = "\n\n".join(f'<chunk doc="{h.chunk.doc_id}">\n{h.chunk.text}\n</chunk>' for h in hits)
    store = json.dumps(load_store(), indent=2)
    return f"{INSTRUCTIONS}\n\n<policies>\n{chunks}\n</policies>\n\n<store_database>\n{store}\n</store_database>"


def main(names: list[str]) -> None:
    index = build_index()
    print(f"index: {len(index.chunks)} chunks embedded in {index.build_s}s")
    scenarios = [s for s in load_scenarios() if not names or s["name"] in names]

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"stage1-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    results = []
    with out_path.open("w") as out:
        for scenario in scenarios:
            hits = index.search(scenario["message"]["text"], TOP_K)
            docs = ranked_docs(hits)
            missing = [d for d in scenario["expect"]["required_docs"] if d not in docs]
            print(f"\n{'=' * 100}\n{scenario['name']}: {scenario['message']['text']}")
            print(f"retrieved {docs}" + (f"   MISSING {missing}" if missing else ""))
            print("-" * 100)

            result = complete(build_system_prompt(hits), format_message(scenario["message"]))
            print(result.text)
            row = {"name": scenario["name"], "expect": scenario["expect"], "retrieved": docs, "missing": missing,
                   "answer": result.text, **{k: v for k, v in asdict(result).items() if k != "text"}}
            out.write(json.dumps(row) + "\n")
            results.append(row)

    print(f"\n{'scenario':<24}{'in tok':>8}{'cached':>8}{'out tok':>9}{'latency':>9}  {'missing docs':<34}expected")
    for r in results:
        e = r["expect"]
        print(f"{r['name']:<24}{r['input_tokens']:>8}{r['cache_read_tokens']:>8}{r['output_tokens']:>9}{r['latency_s']:>8}s  "
              f"{', '.join(r['missing']) or '-':<34}{e['final_status']}, ${e['case_value']:.2f}, "
              f"approval={e['needs_approval']}")
    print(f"\ntotal ${sum(r['cost_usd'] for r in results):.4f} (list-price equivalent)   saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
