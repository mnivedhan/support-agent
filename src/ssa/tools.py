"""Read-only tools the agent can call.

A tool failure goes back to the model as {"error": ...}, not as an exception: a wrong id or a bad argument is
something the model can fix on its next step. Nothing here changes state. Executing refunds is code's job, after
approval (Stage 4).

Not yet least-privilege: get_order and get_customer will return *any* order or customer, not just the ones in this
case. A prompt injection could use that to read other customers' data. Stage 3 scopes tools to the case.
"""

from dataclasses import asdict, dataclass, field
from functools import cache

from ssa.data import load_knowledge
from ssa.rag.chunking import heading_chunks
from ssa.rag.retriever import get_index
from ssa.resolution import resolve
from ssa.schemas import CaseDecision
from ssa.validation import check_decision


@cache
def policy_sections() -> dict[str, str]:
    """Section id -> "<doc title> > <heading>\\n<text>", for every section of every doc."""
    return {c.id: c.text for doc_id, text in load_knowledge().items() for c in heading_chunks(doc_id, text)}


def table_of_contents() -> str:
    return "\n".join(f"{sid}: {text.splitlines()[0]}" for sid, text in policy_sections().items())


@dataclass
class ToolContext:
    message: dict
    store: dict
    seen: list[str] = field(default_factory=list)   # CLI agent: every tool result, for the grounding check


def get_order(ctx: ToolContext, order_id: str) -> dict:
    order = ctx.store["orders"].get(order_id)
    if order is None:
        return {"error": f"no order {order_id}"}
    products = ctx.store["products"]
    items = [{**i, "name": products[i["product_id"]]["name"], "category": products[i["product_id"]]["category"],
              "stock": products[i["product_id"]]["stock"]} for i in order["items"]]
    return {"order_id": order_id, **order, "items": items}


def get_customer(ctx: ToolContext, customer_id: str) -> dict:
    customer = ctx.store["customers"].get(customer_id)
    return {"customer_id": customer_id, **customer} if customer else {"error": f"no customer {customer_id}"}


def get_policy_sections(ctx: ToolContext, section_ids: list[str]) -> dict:
    sections = policy_sections()
    return {sid: sections.get(sid, "error: no such section id") for sid in section_ids}


def search_policies(ctx: ToolContext, query: str) -> dict:
    return {h.chunk.id: h.chunk.text for h in get_index("heading", "hybrid").search(query, 4)}


def quote_resolution(ctx: ToolContext, decision: dict) -> dict:
    parsed = CaseDecision.model_validate(decision)
    if errors := check_decision(parsed, ctx.message, ctx.store, set(policy_sections())):
        return {"error": errors}
    return asdict(resolve(parsed, ctx.message, ctx.store))


TOOLS = {f.__name__: f for f in (get_order, get_customer, get_policy_sections, search_policies, quote_resolution)}
