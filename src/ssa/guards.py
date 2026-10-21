"""Guardrails in code. They don't depend on the model noticing anything.

Three places where untrusted text meets the agent, and a guard for each:

  intake_check        the customer's message, before any LLM call. Hard violations quarantine the case
                      immediately: no model to fool, no tokens spent. Soft signals are passed on as flags.
  sanitize            data coming back from tools (product names, notes). Text that reads like instructions to
                      an agent is cut out before the model sees it. That's indirect prompt injection: the attack
                      arrives through the database, not the message.
  check_reply_safety  the reply, before it reaches the customer: no card numbers except the order's own, no other
                      customers' names.

Pattern lists like these are a floor, not a wall: an attacker can always rephrase. They catch the cheap attacks
deterministically, while the real protection is structural. Code computes every amount, refunds only go to the
original payment method, and tools only return this customer's data.
"""

import re

INSTRUCTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"system (override|note|prompt|message)",
    r"note to (the )?(agent|assistant|ai)",
    r"\bpre-?approved\b",
    r"you (must|should) (now )?(issue|refund|approve|include)",
    r"as an ai\b",
]
_INSTRUCTIONS = re.compile("|".join(INSTRUCTION_PATTERNS), re.IGNORECASE)
CARD = re.compile(r"card (?:ending|ending in|number ending)?\s*(?:in\s*)?(\d{4})\b", re.IGNORECASE)
MONEY = re.compile(r"(?:\$|USD\s?)\s?(\d[\d,]*(?:\.\d{1,2})?)")


def instruction_like(text: str) -> list[str]:
    return [m.group(0) for m in _INSTRUCTIONS.finditer(text)]


def _order_card(order: dict) -> str | None:
    m = re.search(r"(\d{4})\s*$", order.get("payment", ""))
    return m.group(1) if m else None


def intake_check(message: dict, store: dict) -> tuple[list[str], list[str]]:
    """(hard, soft) flags for an incoming message. Any hard flag quarantines the case."""
    hard, soft = [], []
    text = message["text"]
    if phrases := instruction_like(text):
        hard.append(f"instruction-like text in the message: {phrases}")

    customer_id, order_id = message.get("customer_id"), message.get("order_id")
    order = store["orders"].get(order_id or "")
    if order_id and order is None:
        soft.append(f"order {order_id} does not exist")
    if order and customer_id and order["customer_id"] != customer_id:
        hard.append(f"order {order_id} belongs to another customer")
    if order:
        own_card = _order_card(order)
        for card in CARD.findall(text):
            if card != own_card:
                hard.append(f"asks for money to go to card ending {card}, not the order's card")
        total = sum(i["price"] * i["qty"] for i in order["items"]) + order["shipping"]["fee"]
        for amount in MONEY.findall(text):
            if float(amount.replace(",", "")) > total:
                hard.append(f"asks for USD {amount}, more than the order total of USD {total:.2f}")
    if len(text) > 4000:
        soft.append("message is unusually long")
    return hard, soft


def sanitize(value, flags: list[str], path: str = ""):
    """Copy of a tool result with instruction-like text cut out of every string. Findings are added to `flags`."""
    if isinstance(value, dict):
        return {k: sanitize(v, flags, f"{path}.{k}" if path else k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, flags, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str) and (phrases := instruction_like(value)):
        flags.append(f"instruction-like text removed from tool data at {path}: {phrases}")
        first = _INSTRUCTIONS.search(value).start()
        return value[:first].rstrip(" .:-") + " [removed: instruction-like text]"
    return value


def check_reply_safety(reply: str, message: dict, store: dict) -> list[str]:
    errors = []
    order = store["orders"].get(message.get("order_id") or "")
    own_card = _order_card(order) if order else None
    for card in set(re.findall(r"\b(\d{4})\b", reply)):
        if card != own_card and any(card in c for c in CARD.findall(message["text"])):
            errors.append(f"the reply repeats card ending {card} from the message; never echo payment details")
    if order:
        for cid, customer in store["customers"].items():
            if cid != order["customer_id"] and customer["name"] in reply:
                errors.append("the reply names another customer")
    return errors
