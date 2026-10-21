"""Stage 4's approval queue on the command line. Each command is a separate process, like separate workers and a
reviewer's browser: nothing survives between them except var/checkpoints.db and var/ledger.db.

Run:  python -m ssa.stage4_queue run all                      # the agent handles the 7 scenarios; some wait
      python -m ssa.stage4_queue queue                        # what's waiting, and why
      python -m ssa.stage4_queue approve laptop-cracked-screen --by ana
      python -m ssa.stage4_queue reject repeat-not-received --by ana --note "4th claim"
      python -m ssa.stage4_queue close prompt-injection --by sec-team
      python -m ssa.stage4_queue ledger                       # money moved and emails sent

Break it:
      python -m ssa.stage4_queue run laptop-cracked-screen --memory     # then `queue --memory`: gone
      python -m ssa.stage4_queue approve laptop-cracked-screen --by ana --set products.P-LAP.stock=0   # stale
      python -m ssa.stage4_queue approve laptop-cracked-screen --by ana --crash-at execute
      python -m ssa.stage4_queue recover [case ...]           # finish what a dead worker left
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ssa.agent_graph import gemini
from ssa.data import load_scenarios
from ssa.ledger import Ledger
from ssa.stage3_eval import patched_store
from ssa.stage4_graph import Stage4Context, build_graph, decide, pending, recover, stuck, submit

VAR = Path(__file__).resolve().parents[2] / "var"


@contextmanager
def open_graph(memory: bool):
    if memory:
        cp = InMemorySaver()
        yield build_graph(cp), cp
        return
    with SqliteSaver.from_conn_string(str(VAR / "checkpoints.db")) as cp:
        yield build_graph(cp), cp


def store_with(sets: list[str]) -> dict:
    """--set products.P-LAP.stock=0: the database as it is when the reviewer clicks, not when the agent ran."""
    return patched_store({k: json.loads(v) for k, v in (s.split("=", 1) for s in sets)})


def show(state: dict) -> None:
    for entry in state.get("log", []):
        print("  " + json.dumps(entry, default=str)[:200])
    print(f"  -> {state.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "queue", "approve", "reject", "close", "recover", "ledger"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--by", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--set", action="append", default=[], help="change the store before acting: path=json")
    ap.add_argument("--crash-at", choices=["execute"])
    ap.add_argument("--memory", action="store_true", help="in-memory checkpoints: lost when the process exits")
    args = ap.parse_args()
    VAR.mkdir(exist_ok=True)
    ledger = Ledger(":memory:" if args.memory else VAR / "ledger.db")
    ctx = Stage4Context(store_with(args.set), ledger, crash_at=args.crash_at)

    with open_graph(args.memory) as (graph, cp):
        if args.command == "run":
            ctx.model = gemini()
            scenarios = [s for s in load_scenarios() if args.names in ([], ["all"]) or s["name"] in args.names]
            for s in scenarios:
                print(f"\n[{s['name']}] {s['message']['text'][:100]}")
                try:
                    show(submit(graph, s["name"], s["message"], ctx))
                except Exception as e:  # noqa: BLE001 -- the case keeps its checkpoint; `recover` finishes it
                    print(f"  ERROR {type(e).__name__}: {str(e)[:150]}\n  -> stuck; `recover` finishes it")
        elif args.command in ("approve", "reject", "close"):
            for case_id in args.names:
                print(f"\n[{case_id}] {args.command} by {args.by}")
                try:
                    show(decide(graph, case_id, args.command, args.by, ctx, args.note))
                except ValueError as e:
                    print(f"  refused: {e}")
        elif args.command == "recover":
            cases = [t for t in stuck(graph, cp) if not args.names or t in args.names]
            if any(graph.get_state({"configurable": {"thread_id": t}}).next == ("handle",) for t in cases):
                ctx.model = gemini()   # a case that died inside the agent loop runs the loop again
            for case_id in cases:
                print(f"\n[{case_id}] recovering")
                show(recover(graph, case_id, ctx))
        elif args.command == "ledger":
            for a in ledger.actions():
                print(f"  {a['at']}  {a['case_id']:<24} {a['line']:<28} approved by {a['approved_by']}")
            for e in ledger.emails():
                print(f"  {e['at']}  {e['case_id']:<24} email {e['kind']:<8} {e['body'][:80]!r}")
        if args.command in ("run", "queue", "approve", "reject", "close", "recover"):
            waiting = pending(graph, cp)
            print(f"\nqueue: {len(waiting)} waiting" + ("" if waiting else " (empty)"))
            for q in waiting:
                print(f"  {q['case_id']:<24} {q['customer_id']}  actions={q['actions']}")
                print(f"    proposal: {q['proposal']['lines'] if q['proposal'] else None}, "
                      f"USD {q['proposal']['case_value'] if q['proposal'] else 0:.2f}")
                for why in q["why"]:
                    print(f"    why: {why}"[:160])
                if q.get("error"):
                    print(f"    last answer refused: {q['error']}")


if __name__ == "__main__":
    main()
