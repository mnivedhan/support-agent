"""Stage 2 with LangChain's prebuilt agent (`create_agent`) instead of a hand-drawn graph.

Same tools, calculator, gates, grading and Gemini model as agent_graph.py. The difference is where each piece lives:

    agent_graph.py (LangGraph)             here (create_agent + middleware)
    prefetch node                       →  before_agent hook
    agent ⇄ tools edges                 →  the built-in loop
    submit_final tool → validate node   →  response_format=ToolStrategy(FinalAnswer), checked in an after_model hook
    rejected → edge back to agent       →  after_model returns jump_to="model"
    step budget → escalate node         →  ModelCallLimitMiddleware(run_limit=...), then "no answer" = escalated
    text-only turn → nudge node         →  after_model returns jump_to="model" (otherwise the loop just ends)

The loop itself is LangChain's; we only hook into it. That's less code, but the repair route is now a hook that
jumps, not an edge you can see on the graph.
"""

import json
import time
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, hook_config
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ssa import tools as base
from ssa.agent_graph import INSTRUCTIONS, MAX_STEPS, TOOLS, AgentRun, build_system_prompt, gemini
from ssa.resolution import resolve
from ssa.schemas import FinalAnswer
from ssa.stage0_baseline import format_message
from ssa.validation import check_decision, check_reply

# The prebuilt agent has no submit_final tool: the final answer is the structured response.
PROMPT_INSTRUCTIONS = INSTRUCTIONS.replace("submit_final", "your final answer (FinalAnswer)")


class CaseState(AgentState):
    message: NotRequired[dict]
    store: NotRequired[dict]
    case_data: NotRequired[str]
    trace: NotRequired[list[dict]]


class SupportCase(AgentMiddleware[CaseState]):
    """Prefetch before the loop, and the three gates on the final answer after each model call."""

    state_schema = CaseState

    def before_agent(self, state: CaseState, runtime) -> dict[str, Any]:
        ctx, msg = base.ToolContext(state["message"], state["store"]), state["message"]
        lookups = []
        if msg.get("order_id"):
            lookups.append(f"get_order({msg['order_id']}) -> {json.dumps(base.get_order(ctx, msg['order_id']))}")
        if msg.get("customer_id"):
            lookups.append(f"get_customer({msg['customer_id']}) -> "
                           f"{json.dumps(base.get_customer(ctx, msg['customer_id']))}")
        case_data = "\n".join(lookups) or "(no order or customer on this message)"
        user = (f"Today is {state['store']['today']}.\n\n<customer_message>\n{format_message(msg)}\n"
                f"</customer_message>\n\n<case_data>\n{case_data}\n</case_data>")
        return {"messages": [HumanMessage(user)], "case_data": case_data, "trace": []}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: CaseState, runtime) -> dict[str, Any] | None:
        ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        n = sum(isinstance(m, AIMessage) for m in state["messages"])
        trace = list(state.get("trace", []))
        final: FinalAnswer | None = state.get("structured_response")
        if final is None:
            calls = [{"tool": c["name"], "args": c["args"]} for c in ai.tool_calls]
            if calls:
                return {"trace": trace + [{"step": n, "tool_calls": calls}]}
            # The prebuilt loop treats a text-only turn as "done", which would end the case with no answer.
            return {"jump_to": "model",
                    "messages": [HumanMessage("Call a tool, or give your final answer (FinalAnswer).")]}

        errors = check_decision(final.decision, state["message"], state["store"], set(base.policy_sections()))
        if not errors:
            resolution = resolve(final.decision, state["message"], state["store"])
            seen = [state["case_data"]] + [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
            errors = check_reply(final.reply, resolution, seen)
        trace.append({"step": n, "final": final.model_dump(), "rejected": errors})
        if not errors:
            return {"trace": trace}
        # Throw the answer away and send the model back with the reasons. Without jump_to, the agent would end here.
        return {"trace": trace, "structured_response": None, "jump_to": "model",
                "messages": [HumanMessage("Your final answer was rejected. Fix these and answer again:\n- "
                                          + "\n- ".join(errors))]}


def build_agent(model: BaseChatModel):
    system = build_system_prompt().replace(INSTRUCTIONS, PROMPT_INSTRUCTIONS)
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=system,
        response_format=ToolStrategy(FinalAnswer),
        middleware=[SupportCase(), ModelCallLimitMiddleware(run_limit=MAX_STEPS, exit_behavior="end")],
        state_schema=CaseState,
    )


def run_prebuilt(message: dict, store: dict, model: BaseChatModel | None = None) -> AgentRun:
    agent = build_agent(model or gemini())
    started = time.perf_counter()
    state = agent.invoke({"messages": [], "message": message, "store": store},
                         {"recursion_limit": 4 * MAX_STEPS + 10})
    run = AgentRun(status="escalated", trace=state.get("trace", []), latency_s=round(time.perf_counter() - started, 2))
    if final := state.get("structured_response"):
        run.status, run.final = "done", final
        run.resolution = resolve(final.decision, message, store)
    for m in state["messages"]:
        # Skip the "Model call limits exceeded" message ModelCallLimitMiddleware adds: it isn't a model call.
        if isinstance(m, AIMessage) and (m.usage_metadata or m.tool_calls):
            run.llm_calls += 1
            usage = m.usage_metadata or {}
            run.input_tokens += usage.get("input_tokens", 0)
            run.output_tokens += usage.get("output_tokens", 0)
            run.cache_read_tokens += (usage.get("input_token_details") or {}).get("cache_read", 0)
    return run
