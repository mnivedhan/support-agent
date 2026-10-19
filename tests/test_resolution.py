"""The answer key as unit tests. The calculator is plain code, so these run in milliseconds with no LLM.

Each test hands resolve() the decision a correct model would make and checks the money against the answer key in
notes/stage-00-baseline.md. If a test fails, the bug is in the rules, not the model.
"""

import copy

import pytest

from ssa.data import load_scenarios, load_store
from ssa.resolution import resolve
from ssa.schemas import CaseDecision, ItemClaim

SCENARIOS = {s["name"]: s for s in load_scenarios()}


def decide(kinds, order_id, items=(), suspicious=False):
    return CaseDecision(kinds=kinds, order_id=order_id, items=[ItemClaim(**i) for i in items],
                        suspicious=suspicious, suspicious_reason="test" if suspicious else None, policy_sections=[])


CORRECT_DECISIONS = {
    "damaged-and-late": decide(["damaged_item", "late_delivery"], "1042",
                               [{"product_id": "P-MUG", "issue": "damaged", "wants": "refund"}]),
    "laptop-cracked-screen": decide(["damaged_item"], "1077",
                                    [{"product_id": "P-LAP", "issue": "damaged", "wants": "replacement"}]),
    "repeat-not-received": decide(["not_received"], "1090",
                                  [{"product_id": "P-HDP", "issue": "not_received", "wants": "refund"}]),
    "gold-late-delivery": decide(["late_delivery"], "1101"),
    "change-of-mind-expired": decide(["change_of_mind"], "1033",
                                     [{"product_id": "P-BLD", "issue": "change_of_mind", "wants": "refund"}]),
    "shipping-question": decide(["question"], None),
    "prompt-injection": decide(["question"], "1042", suspicious=True),
}


@pytest.mark.parametrize("name", CORRECT_DECISIONS)
def test_answer_key(name):
    expect = SCENARIOS[name]["expect"]
    r = resolve(CORRECT_DECISIONS[name], SCENARIOS[name]["message"], load_store())
    assert r.status == expect["final_status"]
    assert r.needs_approval == expect["needs_approval"]
    assert r.case_value == pytest.approx(expect["case_value"])
    assert sorted(r.lines) == sorted(expect["resolution"])


def test_laptop_without_photos_asks_for_them():
    message = {**SCENARIOS["laptop-cracked-screen"]["message"], "attachments": []}
    r = resolve(CORRECT_DECISIONS["laptop-cracked-screen"], message, load_store())
    assert r.status == "needs_info" and r.lines == []


def test_late_claim_on_an_on_time_order_gets_no_credit():
    r = resolve(decide(["late_delivery"], "1077"), {"attachments": []}, load_store())
    assert r.status == "closed_no_action" and r.case_value == 0


def test_refund_one_day_outside_the_window_does_not_count():
    store = load_store()
    store = copy.deepcopy(store)
    store["customers"]["C-150"]["refund_history"].append(
        {"order_id": "0999", "date": "2026-07-17", "type": "refund", "amount": 10.0})
    r = resolve(CORRECT_DECISIONS["repeat-not-received"], {"attachments": []}, store)
    assert "claim 3 in 90 days" in " ".join(r.reasons)   # still 3, not 4


def test_refunding_every_item_also_refunds_standard_shipping():
    decision = decide(["damaged_item"], "1042", [{"product_id": "P-MUG", "issue": "damaged", "wants": "refund"},
                                                 {"product_id": "P-FRP", "issue": "damaged", "wants": "refund"}])
    r = resolve(decision, {"attachments": []}, load_store())
    assert sorted(r.lines) == ["refund P-FRP 35.00", "refund P-MUG 40.00", "refund shipping 5.99"]


def test_change_of_mind_inside_the_window_charges_restocking():
    store = copy.deepcopy(load_store())
    store["today"] = "2026-09-20"   # 19 days after the blender was delivered
    r = resolve(CORRECT_DECISIONS["change-of-mind-expired"], {"attachments": []}, store)
    assert r.lines == ["refund P-BLD 109.65"] and r.status == "completed"
