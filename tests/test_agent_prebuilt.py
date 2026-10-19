"""The same scripted runs as test_agent_graph.py, against the create_agent version, so the two behave alike.

The final answer here is the structured response, a tool call named after the schema (FinalAnswer), instead of
submit_final.
"""

from langchain_core.messages import AIMessage, ToolMessage

from ssa import agent_prebuilt
from ssa.agent_prebuilt import run_prebuilt
from ssa.data import load_store
from tests.test_agent_graph import DECISION, GOLD, ScriptedModel, call


def test_quote_reject_repair_accept():
    model = ScriptedModel(turns=[
        call("quote_resolution", {"decision": DECISION}, 1),
        call("FinalAnswer", {"decision": DECISION, "reply": "Your credit is being checked."}, 2),
        call("FinalAnswer", {"decision": DECISION, "reply": "Refunded USD 54.00 credit + USD 14.99 fee."}, 3),
    ])
    run = run_prebuilt(GOLD["message"], load_store(), model)

    assert run.status == "done" and run.llm_calls == 3
    assert run.resolution.case_value == 68.99 and run.resolution.status == "completed"
    assert run.trace[1]["rejected"] and not run.trace[2]["rejected"]
    assert model.seen[2][-1].content.startswith("Your final answer was rejected.")


def test_prefetch_puts_order_and_customer_in_the_first_prompt():
    model = ScriptedModel(turns=[call("FinalAnswer", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 1)])
    run = run_prebuilt(GOLD["message"], load_store(), model)
    first_prompt = model.seen[0][1].content
    assert "get_order(1101)" in first_prompt and '"tier": "gold"' in first_prompt
    assert run.status == "done" and run.llm_calls == 1


def test_call_limit_escalates_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(agent_prebuilt, "MAX_STEPS", 3)
    model = ScriptedModel(turns=[call("get_policy_sections", {"section_ids": ["support-sop#purpose"]}, n)
                                 for n in range(10)])
    run = run_prebuilt(GOLD["message"], load_store(), model)
    assert run.status == "escalated" and run.resolution is None and run.llm_calls == 3


def test_tool_errors_go_back_to_the_model():
    model = ScriptedModel(turns=[
        call("get_order", {"wrong_arg": "1101"}, 1),
        call("FinalAnswer", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 2),
    ])
    run_prebuilt(GOLD["message"], load_store(), model)
    error = model.seen[1][-1]
    assert isinstance(error, ToolMessage) and "error" in error.content.lower()


def test_a_text_only_turn_is_nudged():
    model = ScriptedModel(turns=[AIMessage("Let me think about this."),
                                 call("FinalAnswer", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 1)])
    run = run_prebuilt(GOLD["message"], load_store(), model)
    assert run.status == "done" and "FinalAnswer" in model.seen[1][-1].content
