"""The graph's routing, tested with a scripted model: no API key, no cost, same result every time.

The model is a stand-in that returns pre-written turns. That isolates what we wrote (nodes, edges, gates) from what
the LLM does. Whether Gemini makes good decisions is the eval's job (stage2_langgraph); whether the graph sends a
rejected answer back, stops at the budget and grades correctly is this file's job.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from ssa import agent_graph
from ssa.agent_graph import build_graph, run_graph
from ssa.data import load_scenarios, load_store

GOLD = next(s for s in load_scenarios() if s["name"] == "gold-late-delivery")
DECISION = {"kinds": ["late_delivery"], "order_id": "1101", "items": [], "suspicious": False,
            "suspicious_reason": None, "policy_sections": ["shipping-policy#late-delivery-credit"]}


class ScriptedModel(BaseChatModel):
    turns: list[AIMessage]
    seen: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.turns.pop(0))])


def call(name, args, n):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call-{n}"}])


def test_quote_reject_repair_accept():
    model = ScriptedModel(turns=[
        call("quote_resolution", {"decision": DECISION}, 1),
        call("submit_final", {"decision": DECISION, "reply": "Your credit is being checked."}, 2),
        call("submit_final", {"decision": DECISION, "reply": "Refunded USD 54.00 credit + USD 14.99 fee."}, 3),
    ])
    run = run_graph(GOLD["message"], load_store(), model)

    assert run.status == "done" and run.llm_calls == 3
    assert run.resolution.case_value == 68.99 and run.resolution.status == "completed"
    assert run.trace[1]["rejected"] == ["the reply doesn't mention the amount for 'late credit 54.00'",
                                        "the reply doesn't mention the amount for 'refund express fee 14.99'"]
    rejection = model.seen[2][-1]   # what the model saw before its third turn
    assert isinstance(rejection, ToolMessage) and rejection.content.startswith("Rejected.")


def test_prefetch_puts_order_and_customer_in_the_first_prompt():
    model = ScriptedModel(turns=[call("submit_final", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 1)])
    run = run_graph(GOLD["message"], load_store(), model)
    first_prompt = model.seen[0][1].content
    assert "get_order(1101)" in first_prompt and '"tier": "gold"' in first_prompt
    assert run.status == "done" and run.llm_calls == 1


def test_step_budget_escalates_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(agent_graph, "MAX_STEPS", 3)
    model = ScriptedModel(turns=[call("get_policy_sections", {"section_ids": ["support-sop#purpose"]}, n)
                                 for n in range(10)])
    run = run_graph(GOLD["message"], load_store(), model)
    assert run.status == "escalated" and run.resolution is None and run.llm_calls == 3


def test_tool_errors_go_back_to_the_model():
    model = ScriptedModel(turns=[
        call("get_order", {"wrong_arg": "1101"}, 1),
        call("submit_final", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 2),
    ])
    run_graph(GOLD["message"], load_store(), model)
    error = model.seen[1][-1]
    assert isinstance(error, ToolMessage) and "error" in error.content.lower()


def test_a_text_only_turn_is_nudged():
    model = ScriptedModel(turns=[AIMessage("Let me think about this."),
                                 call("submit_final", {"decision": DECISION, "reply": "USD 54.00 and USD 14.99"}, 1)])
    run = run_graph(GOLD["message"], load_store(), model)
    assert run.status == "done" and "submit_final" in model.seen[1][-1].content


def test_graph_shape():
    graph = build_graph(ScriptedModel(turns=[])).get_graph()
    assert {"prefetch", "agent", "tools", "validate", "nudge", "escalate"} <= set(graph.nodes)
