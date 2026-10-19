"""Code does the math: turn the model's CaseDecision into money, an approval decision and a status.

The model says what happened; this file applies the rules. Every constant cites the policy section it comes from.
That's the tradeoff of Stage 2: amounts become exact, repeatable and unit-testable (tests/test_resolution.py),
but the rules now live in two places. When a policy changes, the doc and this file must change together.

It also checks the model's claims against the database instead of trusting them: a "late" order that tracking
shows on time gets no credit, and the photo rule reads the real attachments, not what the model says.
"""

from dataclasses import dataclass, field
from datetime import date

from ssa.case_context import order_tags
from ssa.schemas import CaseDecision

# shipping-policy#late-delivery-credit and loyalty-tiers#gold-usd-1-500-or-more-per-year: (rate per day, cap)
LATE_CREDIT = {"gold": (0.10, 0.30)}
STANDARD_LATE_CREDIT = (0.05, 0.20)
NO_LATE_CREDIT_TAGS = {"pickup", "subscription", "bulk_order"}  # store-pickup, subscription-coffee, bulk-orders
DAMAGE_REPORT_DAYS = 7               # refund-policy#damaged-or-wrong-items
RETURN_REQUIRED_FROM = 100.00        # refund-policy#damaged-or-wrong-items
CHANGE_OF_MIND_DAYS = 30             # refund-policy#change-of-mind
RESTOCKING_FEE = 0.15                # refund-policy#change-of-mind
PHOTOS_REQUIRED_ABOVE = 500.00       # fraud-and-abuse#high-value-damage-claims
APPROVAL_ABOVE = 200.00              # support-sop#approval-thresholds
REPEAT_WINDOW_DAYS = 90              # fraud-and-abuse#repeat-claims
REPEAT_CLAIM_LIMIT = 3               # the 3rd claim in the window needs approval
WARRANTY_DAYS = {"electronics": 365, "small_appliance": 365, "kitchenware": 90, "accessories": 90}  # warranty-policy


@dataclass
class Resolution:
    status: str   # completed | awaiting_approval | denied | needs_info | closed_no_action | quarantined
    lines: list[str] = field(default_factory=list)     # "refund P-MUG 40.00", "replace P-LAP", "carrier claim"
    case_value: float = 0.0
    needs_approval: bool = False
    reasons: list[str] = field(default_factory=list)   # every check that fired, for the reply and the audit log


def resolve(decision: CaseDecision, message: dict, store: dict) -> Resolution:
    if decision.suspicious:
        return Resolution("quarantined", reasons=[f"flagged for review: {decision.suspicious_reason}"])
    order = store["orders"].get(decision.order_id or "")
    if order is None:
        if set(decision.kinds) <= {"question"}:
            return Resolution("closed_no_action", reasons=["question only, nothing to execute"])
        return Resolution("needs_info", reasons=["no order found for this case; ask the customer for the order number"])

    today = date.fromisoformat(store["today"])
    products = store["products"]
    customer = store["customers"][order["customer_id"]]
    delivered = date.fromisoformat(order["delivered"]) if order.get("delivered") else None
    days_since_delivery = (today - delivered).days if delivered else None
    paid = {i["product_id"]: i["price"] * i["qty"] for i in order["items"]}
    subtotal = sum(paid.values())

    lines: list[str] = []
    reasons: list[str] = []
    denied: list[str] = []
    missing_info: list[str] = []
    value = 0.0
    # Items refunded because the store got it wrong (damaged or wrong). The policy says standard shipping is
    # refunded "when every item in the order is refunded", which read literally includes change-of-mind returns.
    # We read it as store-fault refunds only. Writing the rule as code forced that choice; confirm it with the
    # policy owner. An LLM would make it silently, maybe differently each run.
    refunded: set[str] = set()

    not_received = "not_received" in decision.kinds or any(c.issue == "not_received" for c in decision.items)
    if not_received:
        total = subtotal + order["shipping"]["fee"]
        if order["tracking"] == "delivered":
            lines.append("carrier claim")
            reasons.append("tracking shows delivered, so a carrier claim is opened before refunding")
        lines.append(f"refund order {total:.2f}")
        value += total

    for claim in decision.items:
        pid = claim.product_id
        price = paid[pid]
        product = products[pid]
        if claim.issue in ("damaged", "wrong"):
            if days_since_delivery is None or days_since_delivery > DAMAGE_REPORT_DAYS:
                denied.append(f"{pid}: damage must be reported within {DAMAGE_REPORT_DAYS} days of delivery "
                              f"(reported {days_since_delivery} days after)")
                continue
            if price > PHOTOS_REQUIRED_ABOVE and not message.get("attachments"):
                missing_info.append(f"{pid}: photos of the damage are required for items over "
                                    f"USD {PHOTOS_REQUIRED_ABOVE:.0f}")
                continue
            in_stock = product["stock"] > 0
            if claim.wants == "replacement" and not in_stock:
                reasons.append(f"{pid} is out of stock, so it is refunded instead of replaced")
            if in_stock and claim.wants in ("replacement", "no_preference"):
                lines.append(f"replace {pid}")
                value += product["price"]
            else:
                lines.append(f"refund {pid} {price:.2f}")
                value += price
                refunded.add(pid)
            reasons.append(f"{pid}: send a prepaid return label (store cost USD 6)" if price >= RETURN_REQUIRED_FROM
                           else f"{pid}: under USD {RETURN_REQUIRED_FROM:.0f}, no return needed")
        elif claim.issue == "change_of_mind":
            if days_since_delivery is None or days_since_delivery > CHANGE_OF_MIND_DAYS:
                denied.append(f"{pid}: change-of-mind returns close {CHANGE_OF_MIND_DAYS} days after delivery "
                              f"(delivered {days_since_delivery} days ago)")
                continue
            amount = round(price * (1 - RESTOCKING_FEE), 2)
            lines.append(f"refund {pid} {amount:.2f}")
            value += amount
            free_return = customer["tier"] in ("silver", "gold")
            reasons.append(f"{pid}: {RESTOCKING_FEE:.0%} restocking fee; return shipping paid by "
                           f"{'the store (loyalty tier)' if free_return else 'the customer'}")
        elif claim.issue == "defect":
            limit = WARRANTY_DAYS.get(product["category"], 0)
            if days_since_delivery is None or days_since_delivery > limit:
                denied.append(f"{pid}: outside the {limit}-day warranty for {product['category']}")
                continue
            if product["stock"] > 0:
                lines.append(f"warranty replace {pid}")
                value += product["price"]
            else:
                lines.append(f"warranty repair {pid}")

    if "late_delivery" in decision.kinds and not not_received:
        days_late = (delivered - date.fromisoformat(order["promised"])).days if delivered else 0
        if days_late <= 0:
            reasons.append("not late: tracking shows delivery by the promised date, so no late credit")
        elif excluded := order_tags(order, store) & NO_LATE_CREDIT_TAGS:
            reasons.append(f"no late credit for {', '.join(sorted(excluded))} orders")
        else:
            rate, cap = LATE_CREDIT.get(customer["tier"], STANDARD_LATE_CREDIT)
            credit = round(subtotal * min(rate * days_late, cap), 2)
            lines.append(f"late credit {credit:.2f}")
            value += credit
            reasons.append(f"{days_late} days late x {rate:.0%} per day, capped at {cap:.0%}, "
                           f"of the USD {subtotal:.2f} subtotal")
            if order["shipping"]["method"] == "express":
                lines.append(f"refund express fee {order['shipping']['fee']:.2f}")
                value += order["shipping"]["fee"]

    if refunded == set(paid) and order["shipping"]["method"] == "standard" and not not_received:
        lines.append(f"refund shipping {order['shipping']['fee']:.2f}")   # refund-policy#shipping-fees-on-refunds
        value += order["shipping"]["fee"]

    value = round(value, 2)
    reasons += denied + missing_info
    if missing_info:
        return Resolution("needs_info", [], 0.0, False, reasons)
    if not lines:
        status = "denied" if denied else "closed_no_action"
        return Resolution(status, [], 0.0, False, reasons)

    needs_approval = False
    if value > APPROVAL_ABOVE:
        needs_approval = True
        reasons.append(f"case value USD {value:.2f} is above USD {APPROVAL_ABOVE:.0f}: Support Manager approval")
    recent = [r for r in customer["refund_history"]
              if (today - date.fromisoformat(r["date"])).days <= REPEAT_WINDOW_DAYS]
    if len(recent) + 1 >= REPEAT_CLAIM_LIMIT:
        needs_approval = True
        reasons.append(f"this would be claim {len(recent) + 1} in {REPEAT_WINDOW_DAYS} days: approval for "
                       f"fraud review")
    return Resolution("awaiting_approval" if needs_approval else "completed", lines, value, needs_approval, reasons)
