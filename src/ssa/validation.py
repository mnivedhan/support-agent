"""Checks the schema can't express. A failed check goes back to the model as an error to fix (a repair loop).

check_decision: the ids are real and belong together (the product is in the order, the order is the customer's).
check_reply:    every money amount in the reply comes from data the agent actually looked up or was quoted.
                That's a grounding check: an amount the model calculated itself, or copied from the customer's
                message, is rejected.
"""

import re

from ssa.resolution import Resolution
from ssa.schemas import CaseDecision

MONEY = re.compile(r"(?:\$|USD\s?)\s?(\d[\d,]*(?:\.\d{1,2})?)")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def check_decision(decision: CaseDecision, message: dict, store: dict, section_ids: set[str]) -> list[str]:
    errors = []
    if not decision.kinds and not decision.suspicious:
        errors.append("kinds is empty: give at least one case kind")
    if message.get("order_id") and decision.order_id != message["order_id"]:
        errors.append(f"order_id must be the order the message is about ({message['order_id']}), "
                      f"not {decision.order_id}")
    order = store["orders"].get(decision.order_id or "")
    if decision.order_id and order is None:
        errors.append(f"order {decision.order_id} does not exist")
    if order and message.get("customer_id") and order["customer_id"] != message["customer_id"]:
        errors.append(f"order {decision.order_id} belongs to a different customer")
    in_order = {i["product_id"] for i in order["items"]} if order else set()
    for item in decision.items:
        if item.product_id not in in_order:
            errors.append(f"{item.product_id} is not in order {decision.order_id}; items are {sorted(in_order)}")
    if unknown := [s for s in decision.policy_sections if s not in section_ids]:
        errors.append(f"unknown policy section ids {unknown}; use ids from the table of contents")
    return errors


def _amount(text: str) -> float:
    return round(float(text.replace(",", "")), 2)


def check_reply(reply: str, resolution: Resolution, seen: list[str]) -> list[str]:
    grounded = {_amount(n) for text in seen for n in NUMBER.findall(text)}
    grounded |= {_amount(n) for line in resolution.lines for n in NUMBER.findall(line)} | {resolution.case_value}
    errors = [f"the reply mentions {m.group(0)}, which isn't in the quoted resolution or any looked-up data; "
              f"use quote_resolution's amounts" for m in MONEY.finditer(reply) if _amount(m.group(1)) not in grounded]

    in_reply = {_amount(m.group(1)) for m in MONEY.finditer(reply)}
    for line in resolution.lines:
        amounts = [_amount(n) for n in re.findall(r"\d+\.\d{2}", line)]
        if amounts and amounts[0] not in in_reply:
            errors.append(f"the reply doesn't mention the amount for '{line}'")
    return errors
