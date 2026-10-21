"""Stage 4: a case can wait days for a manager, survive a crash, and resume on any worker.

    START → handle ──auto-executable──→ execute ──→ confirm → END
              │                           │ └ data changed since the decision ──→ review
              └→ acknowledge ──needs a human──→ review ──approve──→ execute
                     │                             └─reject or close──→ close → END
                     └ otherwise (denied, needs info, question) → END

- handle: runs the whole Stage 3 graph (agent, gates, critic) as one step and keeps only its outcome.
- acknowledge: sends the first email. The agent's reply, or a holding reply when the case was escalated.
- review: `interrupt()` saves the case and stops. The process can exit. Days later, a manager's
  `Command(resume=...)` continues it from here, in any process, with no LLM call.
- execute: re-checks the approved resolution against today's data, then moves the money through the idempotent
  ledger. If the data changed (the laptop went out of stock), it goes back to review instead.
- confirm / close: the last email. After a human decision it's written by code from the executed lines, so it
  can't promise anything that wasn't approved.

Three design choices, each from something measured or broken (notes/stage-04-approvals.md):
1. The checkpoint holds the case, not the conversation. Putting a checkpointer on the Stage 3 graph saved 11
   checkpoints and 129 KB for one simple case, because every checkpoint copies the full message history.
   Here the agent loop runs inside one node. A crash during the loop re-runs it, which is safe because its tools
   are read-only and costs a few LLM calls. A crash after a payment is what must never repeat the work.
2. The store and the model come in through the runtime context, not the state. They aren't copied into every
   checkpoint, and a resume always sees today's data, which is what the stale-approval check needs.
3. Any side effect can run twice. The node after an interrupt re-runs from its first line, and a crash
   after paying re-runs `execute`. So every side effect goes through the ledger's idempotency keys.
"""

import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ssa import stage3_graph
from ssa.ledger import Ledger
from ssa.resolution import resolve
from ssa.schemas import CaseDecision
from ssa.stage3_eval import to_run

HOLDING_REPLY = stage3_graph.QUARANTINE_REPLY
REJECT_REPLY = ("We've reviewed your request and aren't able to approve it. A member of our team will follow up "
                "with the details.")


class SimulatedCrash(RuntimeError):
    """Fault injection: the worker dies at a chosen point, as a deploy or an OOM kill would."""


@dataclass
class Stage4Context:
    store: dict                         # read fresh on every run and every resume
    ledger: Ledger
    model: BaseChatModel | None = None  # only `handle` calls it; a resume needs none
    crash_at: str | None = None         # "execute": die after paying, before the checkpoint is saved


class Stage4State(TypedDict, total=False):
    case_id: str
    message: dict
    outcome: dict           # what Stage 3 decided: status, reasons, flags, LLM calls, trace. No message history
    proposal: dict | None   # {decision, resolution, reply}: what a human approves, or None when there's nothing
    review: dict | None     # the human's answer
    stale: dict | None      # the approved lines and today's lines, when the data changed in between
    status: str             # waiting_for_review | stale | executed | resolved | replied | rejected | closed
    log: Annotated[list[dict], operator.add]   # the audit trail: who decided what, when, and what ran


def summarize(state: dict) -> tuple[dict, dict | None]:
    """Stage 3's final state → (outcome, proposal). Drops the messages: the reviewer needs the decision, not 20 KB
    of conversation."""
    run = to_run(state, 0.0)
    r, final, esc = state.get("resolution"), state.get("final"), state.get("escalation") or {}
    outcome = {"status": state["status"], "resolution_status": r["status"] if r else None,
               "reasons": r["reasons"] if r else [], "escalation": esc.get("reason"),
               "critic_issues": esc.get("issues", []), "risk_flags": state.get("risk_flags", []),
               "llm_calls": run.llm_calls, "trace": state.get("trace", [])}
    if r and final and r["status"] != "quarantined":
        proposal = {"decision": final["decision"], "resolution": r, "reply": final["reply"]}
    elif "proposed" in esc:
        p = esc["proposed"]
        proposal = {"decision": p["final"]["decision"], "resolution": p["resolution"], "reply": p["final"]["reply"]}
    else:
        proposal = None
    return outcome, proposal


def needs_human(outcome: dict) -> bool:
    return outcome["status"] == "escalated" or outcome["resolution_status"] in ("awaiting_approval", "quarantined")


def confirmation(lines: list[str], store: dict) -> str:
    """The email after a human approved: written by code from what was executed, never by the model."""
    name = {pid: p["name"] for pid, p in store["products"].items()}
    parts = []
    for line in lines:
        match line.split():
            case ["refund", "order", amount]:
                parts.append(f"refunded the full order, USD {amount}")
            case ["refund", "express", "fee", amount]:
                parts.append(f"refunded the USD {amount} express shipping fee")
            case ["refund", "shipping", amount]:
                parts.append(f"refunded the USD {amount} shipping fee")
            case ["refund", pid, amount]:
                parts.append(f"refunded USD {amount} for the {name[pid]}")
            case ["late", "credit", amount]:
                parts.append(f"added a USD {amount} late-delivery credit")
            case ["replace", pid]:
                parts.append(f"sent a replacement {name[pid]}")
            case ["warranty", "replace", pid]:
                parts.append(f"sent a warranty replacement {name[pid]}")
            case ["warranty", "repair", pid]:
                parts.append(f"arranged a warranty repair of your {name[pid]}")
            case ["carrier", "claim"]:
                parts.append("opened a claim with the carrier")
            case _:
                raise ValueError(f"no wording for resolution line {line!r}")
    done = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Thanks for waiting while we reviewed your case. We've {done}. Refunds go to your original payment method."


def review_request(state: Stage4State) -> dict:
    """What the reviewer sees in the queue."""
    o, p = state["outcome"], state.get("proposal")
    why = [o["escalation"]] if o["escalation"] else []
    why += o["critic_issues"] + [r for r in o["reasons"] if "approval" in r or "flagged" in r]
    if state.get("stale"):
        why.insert(0, f"data changed since the last decision: {state['stale']['was']} -> {state['stale']['now']}")
    return {"case_id": state["case_id"], "customer_id": state["message"].get("customer_id"),
            "message": state["message"]["text"], "why": why, "risk_flags": o["risk_flags"],
            "proposal": {"lines": p["resolution"]["lines"], "case_value": p["resolution"]["case_value"],
                         "reasons": p["resolution"]["reasons"]} if p else None,
            # Nothing to approve (quarantined, budget used up, nothing left to do): the human handles it and closes.
            "actions": ["approve", "reject"] if p and p["resolution"]["lines"] else ["close"]}


def check_answer(answer: Any, request: dict) -> str | None:
    if not isinstance(answer, dict) or answer.get("action") not in request["actions"]:
        return f"action must be one of {request['actions']}"
    if not str(answer.get("reviewer", "")).strip():
        return "say who is deciding: reviewer is required for the audit log"
    return None


def build_graph(checkpointer=None):
    def handle(state: Stage4State, runtime: Runtime[Stage4Context]) -> dict:
        ctx = runtime.context
        # checkpointer=False: a graph run inside a node inherits the parent's checkpointer, and None means
        # "inherit", not "none". Without it the agent loop wrote 7 checkpoints (64 KB) under this one.
        result = stage3_graph.invoke(state["message"], ctx.store, ctx.model, checkpointer=False)
        outcome, proposal = summarize(result)
        return {"outcome": outcome, "proposal": proposal,
                "log": [{"event": "handled", "status": outcome["status"], "llm_calls": outcome["llm_calls"],
                         "resolution": proposal["resolution"]["lines"] if proposal else None, "at": _now()}]}

    def acknowledge(state: Stage4State, runtime: Runtime[Stage4Context]) -> dict:
        o, p = state["outcome"], state.get("proposal")
        escalated = o["status"] == "escalated"
        reply = HOLDING_REPLY if escalated or not p else p["reply"]
        runtime.context.ledger.send(state["case_id"], "ack", reply)
        status = "waiting_for_review" if needs_human(o) else "replied"
        return {"status": status, "log": [{"event": "sent ack", "at": _now()}]}

    def review(state: Stage4State) -> dict:
        # Everything above `interrupt` runs again on resume, so it must be free of side effects.
        request = review_request(state)
        while (problem := check_answer(answer := interrupt(request), request)) is not None:
            request = {**request, "error": problem}   # ask again: a bad answer never moves the case
        entry = {"event": "reviewed", "action": answer["action"], "reviewer": answer["reviewer"],
                 "note": answer.get("note", ""), "approved": request["proposal"], "at": _now()}
        return {"review": answer, "stale": None, "log": [entry]}

    def execute(state: Stage4State, runtime: Runtime[Stage4Context]) -> dict:
        ctx, p = runtime.context, state["proposal"]
        today = resolve(CaseDecision.model_validate(p["decision"]), state["message"], ctx.store)
        if today.lines != p["resolution"]["lines"] or today.case_value != p["resolution"]["case_value"]:
            stale = {"was": p["resolution"]["lines"], "now": today.lines}
            return {"proposal": {**p, "resolution": today.__dict__}, "review": None, "stale": stale,
                    "status": "stale", "log": [{"event": "stale", **stale, "at": _now()}]}
        by = (state.get("review") or {}).get("reviewer", "auto")
        ran = ctx.ledger.execute(state["case_id"], today.lines, by)
        if ctx.crash_at == "execute":
            raise SimulatedCrash("worker died after paying, before the checkpoint was saved")
        return {"status": "executed", "log": [{"event": "executed", "lines": today.lines, "ran_now": ran,
                                               "approved_by": by, "at": _now()}]}

    def confirm(state: Stage4State, runtime: Runtime[Stage4Context]) -> dict:
        ctx = runtime.context
        reply = (confirmation(state["proposal"]["resolution"]["lines"], ctx.store) if state.get("review")
                 else state["proposal"]["reply"])   # no human involved: the agent's validated reply
        ctx.ledger.send(state["case_id"], "confirm", reply)
        return {"status": "resolved", "log": [{"event": "sent confirmation", "at": _now()}]}

    def close(state: Stage4State, runtime: Runtime[Stage4Context]) -> dict:
        if state["review"]["action"] == "reject":   # "close": the human handled it themselves, no email from us
            runtime.context.ledger.send(state["case_id"], "reject", REJECT_REPLY)
            return {"status": "rejected", "log": [{"event": "sent rejection", "at": _now()}]}
        return {"status": "closed", "log": [{"event": "closed by reviewer", "at": _now()}]}

    def after_handle(state: Stage4State) -> str:
        auto = not needs_human(state["outcome"]) and state.get("proposal") \
            and state["proposal"]["resolution"]["status"] == "completed"
        return "execute" if auto else "acknowledge"

    def after_acknowledge(state: Stage4State) -> str:
        return "review" if state["status"] == "waiting_for_review" else END

    def after_review(state: Stage4State) -> str:
        return "execute" if state["review"]["action"] == "approve" else "close"

    def after_execute(state: Stage4State) -> str:
        return "review" if state["status"] == "stale" else "confirm"

    graph = StateGraph(Stage4State, context_schema=Stage4Context)
    for name, node in [("handle", handle), ("acknowledge", acknowledge), ("review", review),
                       ("execute", execute), ("confirm", confirm), ("close", close)]:
        graph.add_node(name, node)
    graph.add_edge(START, "handle")
    graph.add_conditional_edges("handle", after_handle, ["execute", "acknowledge"])
    graph.add_conditional_edges("acknowledge", after_acknowledge, ["review", END])
    graph.add_conditional_edges("review", after_review, ["execute", "close"])
    graph.add_conditional_edges("execute", after_execute, ["review", "confirm"])
    graph.add_edge("confirm", END)
    graph.add_edge("close", END)
    return graph.compile(checkpointer=checkpointer)


# The approval queue: a thin layer over the checkpointer. Any process with the same database can run it.

def _config(case_id: str) -> dict:
    return {"configurable": {"thread_id": case_id}}


def submit(graph, case_id: str, message: dict, ctx: Stage4Context) -> dict:
    """Start a case. A second submit of the same case id (a retried webhook, a double-click) doesn't run it again."""
    existing = graph.get_state(_config(case_id))
    if existing.values:
        return existing.values
    return graph.invoke({"case_id": case_id, "message": message, "log": []}, _config(case_id), context=ctx)


def pending(graph, checkpointer) -> list[dict]:
    """Cases waiting for a human, oldest first. Scans every thread: fine for a demo, a full table scan in
    production. Stage 5 keeps a cases table with a status column and an index."""
    threads = list(dict.fromkeys(c.config["configurable"]["thread_id"] for c in checkpointer.list(None)))
    waiting = []
    for case_id in reversed(threads):
        state = graph.get_state(_config(case_id))
        if state.interrupts:
            waiting.append(state.interrupts[0].value)
    return waiting


def stuck(graph, checkpointer) -> list[str]:
    """Cases that stopped mid-run with no interrupt: a worker died. `recover` continues them."""
    threads = dict.fromkeys(c.config["configurable"]["thread_id"] for c in checkpointer.list(None))
    return [t for t in threads if (s := graph.get_state(_config(t))).next and not s.interrupts]


def decide(graph, case_id: str, action: str, reviewer: str, ctx: Stage4Context, note: str = "") -> dict:
    """A human's answer. Refused when nothing is waiting: a second click on "approve" doesn't approve twice.
    That check is for the reviewer; the ledger's keys are what stop a double payment if two clicks race."""
    if not graph.get_state(_config(case_id)).interrupts:
        raise ValueError(f"case {case_id} isn't waiting for a review")
    answer = {"action": action, "reviewer": reviewer, "note": note}
    return graph.invoke(Command(resume=answer), _config(case_id), context=ctx)


def recover(graph, case_id: str, ctx: Stage4Context) -> dict:
    """Continue a case from its last checkpoint after a crash: the unfinished node runs again from its start."""
    return graph.invoke(None, _config(case_id), context=ctx)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
