"""The tool-calling protocol we wrote by hand for `claude -p`, which has no native tool use for our own functions.

Each step the model returns an AgentStep as JSON (forced with --json-schema): tool calls to run, or a final answer.
run_tool executes a call and records the result for the grounding check. With an API that supports tools natively
(Gemini in agent_graph.py), all of this is replaced by function declarations and ToolNode.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ssa.schemas import FinalAnswer
from ssa.tools import TOOLS, ToolContext

TOOL_DOCS = """\
get_order(order_id) -> the order: items with name, category, price paid and stock; shipping method and fee;
    ordered, promised and delivered dates; tracking status; payment method
get_customer(customer_id) -> name, loyalty tier and refund history
get_policy_sections(section_ids: list of ids) -> the full text of those sections
search_policies(query) -> the 4 best-matching sections, for when the table of contents doesn't show where a rule is
quote_resolution(decision) -> what code computes for a decision: resolution lines, case value, whether approval is
    needed, the status, and the reasons. `decision` has the same shape as final.decision."""


class ToolCall(BaseModel):
    tool: Literal["get_order", "get_customer", "get_policy_sections", "search_policies", "quote_resolution"]
    args: dict[str, Any]


class AgentStep(BaseModel):
    """One turn of the loop: either call tools or give the final answer."""
    tool_calls: list[ToolCall] = Field(description="Tools to run now. Several are allowed. Empty when final is set.")
    final: FinalAnswer | None = Field(description="Set only when the investigation is done.")


def run_tool(call: ToolCall, ctx: ToolContext) -> dict:
    try:
        result = TOOLS[call.tool](ctx, **call.args)
    except TypeError as e:
        result = {"error": f"bad arguments for {call.tool}: {e}"}
    except ValidationError as e:
        result = {"error": f"invalid decision: {e.errors(include_url=False)}"}
    ctx.seen.append(json.dumps(result))
    return result
