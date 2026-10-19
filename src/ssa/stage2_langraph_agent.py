"""Shared by both Stage 2 agents (cli/ and LangGraph): the run summary, grading against the answer key, and the
runner that prints the trace and the results table.

Stage 2 on LangGraph + Gemini. Same tools, calculator, gates and grading as the CLI version.

Run:  python -m ssa.stage2_langgraph                     # all 7 scenarios
      python -m ssa.stage2_langgraph gold-late-delivery  # just one
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

from ssa.agent_graph import AgentRun, run_graph
from ssa.data import load_scenarios, load_store
from ssa.stage0_baseline import RUNS_DIR

FIELDS = ("kind", "status", "approval", "value", "lines")


def grade(expect: dict, run: AgentRun) -> dict[str, bool]:
    """Compare a run to the answer key, field by field. Possible at last, because the output is structured."""
    if run.resolution is None:
        return dict.fromkeys(FIELDS, False)
    d, r = run.final.decision, run.resolution
    kind_ok = d.suspicious if expect["kind"] is None else (
        not d.suspicious and set(d.kinds) == set(expect["kind"].split("+")))
    return {"kind": kind_ok, "status": r.status == expect["final_status"],
            "approval": r.needs_approval == expect["needs_approval"],
            "value": abs(r.case_value - expect["case_value"]) < 0.005,
            "lines": sorted(r.lines) == sorted(expect["resolution"])}


def describe(step: dict) -> str:
    if "tool_calls" in step:
        return ", ".join(f"{c['tool']}({', '.join(map(str, c['args'].values()))[:60]})" for c in step["tool_calls"])
    if "final" in step:
        return "final " + ("REJECTED: " + "; ".join(step["rejected"]) if step["rejected"] else "accepted")
    return "schema error"


def main(names: list[str], runner: Callable[[dict, dict], AgentRun], tag: str) -> None:
    store = load_store()
    scenarios = [s for s in load_scenarios() if not names or s["name"] in names]

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"{tag}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    results = []
    with out_path.open("w") as out:
        for scenario in scenarios:
            print(f"\n{'=' * 100}\n{scenario['name']}: {scenario['message']['text']}\n{'-' * 100}", flush=True)
            run = runner(scenario["message"], store)
            for step in run.trace:
                print(f"  step {step['step']}: {describe(step)}")
            if run.resolution:
                r = run.resolution
                print(f"  -> {r.status}, ${r.case_value:.2f}, approval={r.needs_approval}, lines={r.lines}")
                print(f"     kinds={run.final.decision.kinds} sections={run.final.decision.policy_sections}")
                print("\n" + run.final.reply)
            else:
                print(f"  -> {run.status}: no accepted answer within the step budget")
            marks = grade(scenario["expect"], run)
            row = {"name": scenario["name"], "expect": scenario["expect"], "grade": marks, "status": run.status,
                   "final": run.final.model_dump() if run.final else None,
                   "resolution": asdict(run.resolution) if run.resolution else None,
                   **{k: v for k, v in asdict(run).items() if k not in ("final", "resolution", "status")}}
            out.write(json.dumps(row) + "\n")
            out.flush()
            results.append(row)

    print(f"\n{'scenario':<24}{'kind':>5}{'stat':>5}{'appr':>5}{'value':>6}{'lines':>6}{'steps':>7}{'calls':>6}"
          f"{'in tok':>8}{'cached':>8}{'out tok':>8}{'latency':>9}")
    for r in results:
        marks = "".join(f"{'✓' if r['grade'][f] else '✗':>{6 if f in ('value', 'lines') else 5}}" for f in FIELDS)
        print(f"{r['name']:<24}{marks}{len(r['trace']):>7}{r['llm_calls']:>6}{r['input_tokens']:>8}"
              f"{r['cache_read_tokens']:>8}{r['output_tokens']:>8}{r['latency_s']:>8}s")
    passed = sum(all(r["grade"].values()) for r in results)
    print(f"\n{passed}/{len(results)} fully correct   total ${sum(r['cost_usd'] for r in results):.4f} "
          f"(list-price equivalent)   saved to {out_path}")


if __name__ == "__main__":
    # main(sys.argv[1:], runner=run_graph, tag="stage2-langgraph")
    main("laptop-cracked-screen", runner=run_graph, tag="stage2-langgraph")