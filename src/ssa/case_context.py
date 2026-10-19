"""Case context: what the database knows about a case, as retrieval queries and filter tags.

Stage 1 searched with the message text only. But whether the Gold rule or the repeat-claims rule applies depends on
facts that only exist in the database (tier, refund history, delivery dates), so no search over the message could
find them. Here code looks those facts up and writes each notable one as a plain sentence. Each sentence becomes its
own retrieval query, so one loud topic in the message ("smashed mugs") can't drown out a quiet one ("late").

The weakness is that the fact templates are hand-written: every new kind of fact needs new code. Stage 2 lets the
model ask for what it needs through tools instead.
"""

import re
from dataclasses import dataclass, field
from datetime import date

HOLIDAY_WINDOW = ((11, 15), (12, 24))   # (month, day) range for the holiday return extension


@dataclass
class CaseContext:
    facts: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)   # order attributes, matched against a doc's applies_to
    has_order: bool = False


def build_case_context(message: dict, store: dict) -> CaseContext:
    ctx = CaseContext()
    today = date.fromisoformat(store["today"])

    customer = store["customers"].get(message.get("customer_id") or "")
    if customer:
        if customer["tier"] != "standard":   # standard has no benefits; saying so only pulls in noise
            ctx.facts.append(f"The customer is a {customer['tier'].title()} loyalty tier member.")
        recent = [r for r in customer["refund_history"] if (today - date.fromisoformat(r["date"])).days <= 90]
        if recent:
            ctx.facts.append(f"The customer has had {len(recent)} refunds or replacements in the last 90 days.")

    order = store["orders"].get(message.get("order_id") or "")
    if order:
        ctx.has_order = True
        ctx.tags = order_tags(order, store)
        if order.get("delivered"):
            delivered = date.fromisoformat(order["delivered"])
            ctx.facts.append(f"Tracking shows the order was delivered {(today - delivered).days} days ago.")
            days_late = (delivered - date.fromisoformat(order["promised"])).days
            if days_late > 0:
                ctx.facts.append(f"The order was delivered {days_late} days after the promised date, "
                                 f"with {order['shipping']['method']} shipping.")
        total = sum(i["price"] * i["qty"] for i in order["items"]) + order["shipping"]["fee"]
        for card in re.findall(r"card ending (\d{4})", message["text"]):
            if card not in order["payment"]:
                ctx.facts.append(f"The message asks for money to go to card ending {card}, which is not the card "
                                 f"used to pay for the order.")
        for amount in re.findall(r"\$\s?([\d,]+(?:\.\d+)?)", message["text"]):
            if float(amount.replace(",", "")) > total:
                ctx.facts.append(f"The message asks for USD {amount}, more than the order total of USD {total:.2f}.")

    if message.get("attachments"):
        ctx.facts.append(f"The customer attached {len(message['attachments'])} photos as evidence.")
    return ctx


def order_tags(order: dict, store: dict) -> set[str]:
    """Attributes that decide which scoped help articles can apply. In production these are order columns."""
    products = [store["products"][i["product_id"]] for i in order["items"]]
    ordered = date.fromisoformat(order["ordered"])
    checks = {
        "marketplace_item": any(p.get("seller", "brightcart") != "brightcart" for p in products),
        "partner_card": "partner" in order["payment"].lower(),
        "subscription": any(p["category"] == "subscription" for p in products),
        "bulk_order": any(i["qty"] >= 20 for i in order["items"]),
        "pickup": order["shipping"]["method"] == "pickup",
        "holiday_purchase": HOLIDAY_WINDOW[0] <= (ordered.month, ordered.day) <= HOLIDAY_WINDOW[1],
        "preorder": bool(order.get("preorder")),
        "promo_code": bool(order.get("promo_code")),
        "student_discount": bool(order.get("student_discount")),
    }
    return {tag for tag, applies in checks.items() if applies}
