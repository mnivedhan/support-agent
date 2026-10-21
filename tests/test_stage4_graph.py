"""Stage 4 with a scripted model: waiting for a human, crashes, restarts, stale data and double clicks, for free.

"A new process" is simulated by closing the database and opening a new checkpointer and graph on the same file:
nothing from the first run survives in memory, as after a deploy or a crash.
"""

import copy
from contextlib import contextmanager

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ssa import stage4_graph
from ssa.data import load_scenarios, load_store
from ssa.ledger import Ledger
from ssa.resolution import Resolution
from ssa.stage4_graph import (
    SimulatedCrash,
    Stage4Context,
    confirmation,
    decide,
    pending,
    recover,
    stuck,
    submit,
)
from ssa.validation import check_reply
from tests.test_agent_graph import ScriptedModel, call
from tests.test_stage3_graph import C101, GOOD_REPLY, MUGS, approve

SCENARIOS = {s["name"]: s for s in load_scenarios()}
LAPTOP = SCENARIOS["laptop-cracked-screen"]["message"]
LAPTOP_DECISION = {"kinds": ["damaged_item"], "order_id": "1077", "suspicious": False, "suspicious_reason": None,
                   "items": [{"product_id": "P-LAP", "issue": "damaged", "wants": "replacement"}],
                   "policy_sections": ["refund-policy#damaged-or-wrong-items"]}
LAPTOP_REPLY = "Your replacement laptop is awaiting a manager's approval."


def laptop_model():
    return ScriptedModel(turns=[call("submit_final", {"decision": LAPTOP_DECISION, "reply": LAPTOP_REPLY}, 1)])


@contextmanager
def process(db):
    """One worker process: its own connection, checkpointer and compiled graph."""
    with SqliteSaver.from_conn_string(str(db)) as cp:
        yield stage4_graph.build_graph(cp), cp


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "checkpoints.db", tmp_path / "ledger.db"


def test_auto_completed_case_pays_and_replies_without_a_human(paths):
    db, ledger_db = paths
    ledger = Ledger(ledger_db)
    model = ScriptedModel(turns=[call("submit_final", {"decision": MUGS, "reply": GOOD_REPLY}, 1), approve(1)])
    with process(db) as (graph, _):
        state = submit(graph, "mugs", {**C101, "text": "My mugs arrived smashed."}, Stage4Context(load_store(),
                                                                                                    ledger, model))
    assert state["status"] == "resolved" and state.get("review") is None
    assert [a["line"] for a in ledger.actions()] == ["refund P-MUG 40.00"]
    assert [(e["kind"], e["body"]) for e in ledger.emails()] == [("confirm", GOOD_REPLY)]


def test_approval_survives_a_restart_and_resumes_with_no_llm_call(paths):
    db, ledger_db = paths
    model = laptop_model()
    with process(db) as (graph, _):
        state = submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(ledger_db), model))
    assert state["status"] == "waiting_for_review" and Ledger(ledger_db).actions() == []
    assert [e["kind"] for e in Ledger(ledger_db).emails()] == ["ack"]   # "awaiting a manager's approval"

    with process(db) as (graph, cp):   # days later, another worker, no model at all
        queue = pending(graph, cp)
        assert [q["case_id"] for q in queue] == ["laptop"] and queue[0]["proposal"]["lines"] == ["replace P-LAP"]
        assert any("Support Manager approval" in w for w in queue[0]["why"])
        state = decide(graph, "laptop", "approve", "ana", Stage4Context(load_store(), Ledger(ledger_db)))
    assert state["status"] == "resolved" and len(model.seen) == 1
    actions = Ledger(ledger_db).actions()
    assert [(a["line"], a["approved_by"]) for a in actions] == [("replace P-LAP", "ana")]
    assert "sent a replacement Laptop 14in 16GB" in Ledger(ledger_db).emails("laptop")[-1]["body"]


def test_in_memory_checkpoints_lose_the_case_on_restart(paths):
    """The break that Stage 4 fixes: the default saver lives in the process, so a restart forgets the case."""
    graph = stage4_graph.build_graph(InMemorySaver())
    submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(), laptop_model()))
    restarted = stage4_graph.build_graph(InMemorySaver())
    assert restarted.get_state({"configurable": {"thread_id": "laptop"}}).values == {}


def test_reject_executes_nothing(paths):
    db, ledger_db = paths
    with process(db) as (graph, _):
        submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(ledger_db), laptop_model()))
        state = decide(graph, "laptop", "reject", "ana", Stage4Context(load_store(), Ledger(ledger_db)),
                       note="photos look edited")
    assert state["status"] == "rejected" and Ledger(ledger_db).actions() == []
    assert [e["kind"] for e in Ledger(ledger_db).emails()] == ["ack", "reject"]
    assert state["log"][-2]["note"] == "photos look edited"


def test_data_changed_while_waiting_goes_back_to_review(paths):
    """Approved "replace", but the last laptop sold while the case waited. Same USD 899 value, different action."""
    db, ledger_db = paths
    with process(db) as (graph, cp):
        submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(ledger_db), laptop_model()))
        sold_out = copy.deepcopy(load_store())
        sold_out["products"]["P-LAP"]["stock"] = 0
        state = decide(graph, "laptop", "approve", "ana", Stage4Context(sold_out, Ledger(ledger_db)))
        assert state["status"] == "stale" and Ledger(ledger_db).actions() == []
        again = pending(graph, cp)[0]
        assert again["proposal"]["lines"] == ["refund P-LAP 899.00"] and "data changed" in again["why"][0]
        state = decide(graph, "laptop", "approve", "ana", Stage4Context(sold_out, Ledger(ledger_db)))
    assert state["status"] == "resolved"
    assert [a["line"] for a in Ledger(ledger_db).actions()] == ["refund P-LAP 899.00"]


def test_crash_after_paying_does_not_pay_twice(paths):
    db, ledger_db = paths
    with process(db) as (graph, _):
        submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(ledger_db), laptop_model()))
        with pytest.raises(SimulatedCrash):
            decide(graph, "laptop", "approve", "ana", Stage4Context(load_store(), Ledger(ledger_db),
                                                                     crash_at="execute"))
    with process(db) as (graph, cp):
        assert stuck(graph, cp) == ["laptop"]
        state = recover(graph, "laptop", Stage4Context(load_store(), Ledger(ledger_db)))
    assert state["status"] == "resolved"
    assert len(Ledger(ledger_db).actions()) == 1                # execute ran twice, the ledger paid once
    assert state["log"][-2]["ran_now"] == []                    # the second run found it already done


def test_second_approval_is_refused(paths):
    db, ledger_db = paths
    with process(db) as (graph, _):
        ctx = Stage4Context(load_store(), Ledger(ledger_db), laptop_model())
        submit(graph, "laptop", LAPTOP, ctx)
        decide(graph, "laptop", "approve", "ana", ctx)
        with pytest.raises(ValueError, match="isn't waiting"):
            decide(graph, "laptop", "approve", "ben", ctx)
    assert len(Ledger(ledger_db).actions()) == 1


def test_a_bad_answer_is_asked_again_and_moves_nothing(paths):
    db, ledger_db = paths
    with process(db) as (graph, cp):
        ctx = Stage4Context(load_store(), Ledger(ledger_db), laptop_model())
        submit(graph, "laptop", LAPTOP, ctx)
        config = {"configurable": {"thread_id": "laptop"}}
        graph.invoke(Command(resume={"action": "approve"}), config, context=ctx)   # no reviewer
        assert pending(graph, cp)[0]["error"].startswith("say who is deciding")
        assert Ledger(ledger_db).actions() == []
        assert decide(graph, "laptop", "approve", "ana", ctx)["status"] == "resolved"


def test_duplicate_submit_does_not_run_the_agent_again(paths):
    db, ledger_db = paths
    model = laptop_model()
    with process(db) as (graph, _):
        ctx = Stage4Context(load_store(), Ledger(ledger_db), model)
        submit(graph, "laptop", LAPTOP, ctx)
        submit(graph, "laptop", LAPTOP, ctx)
    assert len(model.seen) == 1 and len(Ledger(ledger_db).emails()) == 1


def test_quarantined_case_can_only_be_closed(paths):
    db, ledger_db = paths
    with process(db) as (graph, cp):
        ctx = Stage4Context(load_store(), Ledger(ledger_db), ScriptedModel(turns=[]))
        submit(graph, "inj", SCENARIOS["prompt-injection"]["message"], ctx)
        assert pending(graph, cp)[0]["actions"] == ["close"]
        decide(graph, "inj", "approve", "ana", ctx)   # not offered: the node asks again
        assert pending(graph, cp)[0]["error"] == "action must be one of ['close']"
        state = decide(graph, "inj", "close", "sec-team", ctx)
    assert state["status"] == "closed" and Ledger(ledger_db).actions() == []
    assert [e["kind"] for e in Ledger(ledger_db).emails()] == ["ack"]


def test_confirmation_email_is_grounded_by_construction():
    store = load_store()
    lines = ["carrier claim", "refund order 70.99", "late credit 54.00", "refund express fee 14.99"]
    text = confirmation(lines, store)
    assert check_reply(text, Resolution("completed", lines, 139.98), seen=[]) == []


def test_interrupted_node_reruns_from_its_first_line():
    """LangGraph's rule that shapes this design: on resume, the node that called interrupt() starts over.
    Anything before the interrupt runs again, so it must have no side effects."""
    calls = []

    def ask(state):
        calls.append("before interrupt")
        return {"answer": interrupt("ok?")}

    g = StateGraph(dict)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    graph = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({}, config)
    graph.invoke(Command(resume="yes"), config)
    assert calls == ["before interrupt", "before interrupt"]


def test_the_agent_loop_writes_no_checkpoints_of_its_own(paths):
    """A graph run inside a node inherits the parent's checkpointer unless compiled with checkpointer=False.
    It did: 7 extra checkpoints (64 KB) of message history per case, under a "handle:<id>" namespace."""
    db, ledger_db = paths
    with process(db) as (graph, cp):
        submit(graph, "laptop", LAPTOP, Stage4Context(load_store(), Ledger(ledger_db), laptop_model()))
        namespaces = {c.config["configurable"]["checkpoint_ns"] for c in cp.list(None)}
    assert namespaces == {""}
