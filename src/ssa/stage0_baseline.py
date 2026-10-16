"""Stage 0: the simplest thing that could possibly work.

Every policy document and the whole store database go into one prompt with the customer message, and one LLM
call returns a resolution as free text. No retrieval, no tools, no structured output and no guardrails, on
purpose. This stage exists to be measured: tokens, cost, latency, and how often the answer is wrong.

Run:  python -m ssa.stage0_baseline                    # all 7 scenarios
      python -m ssa.stage0_baseline damaged-and-late   # just one
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ssa.data import load_knowledge, load_scenarios, load_store
from ssa.llm import complete

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

INSTRUCTIONS = """You are a customer support agent for Brightcart, an online store. A customer message has
arrived. Using only the policies and store database below:
1. Decide what kind of case this is, and whether it needs any action.
2. Work out the resolution the policies allow, with every amount and how you calculated it.
3. State the total case value and whether it needs Support Manager approval, and why.
4. Draft the reply to the customer.
5. Cite the policy documents you relied on by their id."""


def build_system_prompt() -> str:
    docs = "\n\n".join(f'<document id="{doc_id}">\n{text}\n</document>' for doc_id, text in load_knowledge().items())
    store = json.dumps(load_store(), indent=2)
    return f"{INSTRUCTIONS}\n\n<policies>\n{docs}\n</policies>\n\n<store_database>\n{store}\n</store_database>"


def format_message(message: dict) -> str:
    attachments = ", ".join(message["attachments"]) or "none"
    return (f"Channel: {message['channel']}\nCustomer: {message['customer_id']}\nOrder: {message['order_id']}\n"
            f"Attachments: {attachments}\n\n{message['text']}")


def respond(system: str, message: dict) -> dict:
    result = complete(system, format_message(message))
    return {"answer": result.text, **{k: v for k, v in asdict(result).items() if k != "text"}}


def main(names: list[str]) -> None:
    system = build_system_prompt()
    scenarios = [s for s in load_scenarios() if not names or s["name"] in names]

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"stage0-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    results = []
    with out_path.open("w") as out:
        for scenario in scenarios:
            print(f"\n{'=' * 100}\n{scenario['name']}: {scenario['message']['text']}\n{'-' * 100}")
            result = {"name": scenario["name"], "expect": scenario["expect"], **respond(system, scenario["message"])}
            print(result["answer"])
            out.write(json.dumps(result) + "\n")
            results.append(result)

    print(f"\n{'scenario':<24}{'in tok':>8}{'cached':>8}{'out tok':>9}{'latency':>9}{'cost':>9}  expected")
    for r in results:
        e = r["expect"]
        print(f"{r['name']:<24}{r['input_tokens']:>8}{r['cache_read_tokens']:>8}{r['output_tokens']:>9}"
              f"{r['latency_s']:>8}s{'$' + format(r['cost_usd'], '.4f'):>9}"
              f"  {e['final_status']}, ${e['case_value']:.2f}, approval={e['needs_approval']}")
    total = sum(r["cost_usd"] for r in results)
    print(f"\ntotal ${total:.4f} (list-price equivalent)   saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
