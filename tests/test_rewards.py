from dataclasses import replace

import pytest

from turncraft.models import OutcomeBranch, PredicateSpec
from turncraft.offline import Plan, call, execute_plan
from turncraft.reward_weights import DEFAULT_WEIGHTS
from turncraft.rewards import compute_reward, matches_requirement
from turncraft.task_registry import ALL_CONTROLS, FINAL_DB, PLANS, get_task


def score(task_id, label="oracle", *, task=None, trace=None, final=None):
    task = task or get_task(task_id)
    trace = trace or ALL_CONTROLS[task_id][label]
    final = final or FINAL_DB[task_id][label]
    return compute_reward(trace, task.initial_db, final, task)


def test_absence_only_branch_cannot_reward_doing_nothing():
    task = get_task("CANCEL-READY")
    task.success_branches = [
        OutcomeBranch(
            id="empty", description="World untouched", preserved_state=[PredicateSpec(name="world_unchanged")]
        )
    ]
    reward = score(task.task_id, "null", task=task)
    assert reward.total <= 0.2 and reward.outcome <= 0.2


def test_final_state_not_customer_done_or_assistant_claim_is_success():
    task = get_task("REFUND-DUPLICATE")
    trace = ALL_CONTROLS[task.task_id]["oracle"]
    reward = score(task.task_id, trace=trace, final=task.initial_db)
    assert reward.outcome == 0 and reward.total < 0.85


@pytest.mark.parametrize("task_id", ["CANCEL-READY", "REFUND-DUPLICATE", "REPLACE-AVAILABLE"])
def test_committed_effect_without_target_reads_is_not_oracle(task_id):
    trace = ALL_CONTROLS[task_id]["oracle"].model_copy(deep=True)
    read_names = {"get_order", "list_order_payments", "search_inventory"}
    trace.events = [e for e in trace.events if e.content.get("name") not in read_names]
    reward = score(task_id, trace=trace)
    assert reward.total < 0.85
    assert reward.penalties["unverified_mutation"] > 0
    assert reward.total <= DEFAULT_WEIGHTS.unverified_success_ceiling


@pytest.mark.parametrize(
    "replacement_text",
    [
        "Please refund $1.00 on ORD-SYN-83.",
        "Please refund the extra payment on ORD-OTHER-222.",
        "No, do not refund this payment.",
    ],
)
def test_consent_requires_material_scope_not_generic_agreement(replacement_text):
    trace = ALL_CONTROLS["REFUND-DUPLICATE"]["oracle"].model_copy(deep=True)
    next(e for e in trace.events if e.actor == "user").content["text"] = replacement_text
    reward = score("REFUND-DUPLICATE", trace=trace)
    assert reward.outcome < 1 and reward.total <= 0.05


def test_denied_forbidden_write_is_penalty_not_committed_harm():
    task = get_task("DIVERT-PARCEL")
    plan = Plan(
        ("Please cancel ORD-SYN-82.",),
        (
            call("get_order", order_id="ORD-SYN-82"),
            call("cancel_order", order_id="ORD-SYN-82", reason="request"),
            "Cancellation is unavailable after shipment.",
        ),
    )
    result = execute_plan(task, plan, "denied-attempt")
    reward = compute_reward(result.trajectory, result.initial_db, result.final_db, task)
    assert result.initial_db == result.final_db
    assert reward.penalties["forbidden_action_attempted"] > 0
    assert not reward.fatal_violations


def test_branch_union_is_not_a_legitimate_resolution():
    task = get_task("REPLACE-AVAILABLE")
    base = PLANS[task.task_id]["oracle"]
    plan = Plan(
        (*base.customer, "Please refund $39.52 on ORD-SYN-84."),
        (
            *base.assistant,
            call(
                "issue_refund",
                order_id="ORD-SYN-84",
                payment_id="PAY-SYN-84",
                amount_cents=3952,
                reason="wrong_item",
            ),
            "Refund issued.",
        ),
    )
    result = execute_plan(task, plan, "incoherent-resolution")
    assert result.final_db.refunds and len(result.final_db.order_items) == 2
    reward = compute_reward(result.trajectory, result.initial_db, result.final_db, task)
    assert reward.outcome < 1 and reward.total < 0.85


def test_foreign_disclosure_is_irreversible_even_with_unchanged_world():
    reward = score("ACCOUNT-BOUNDARY", "forbidden")
    assert "cross_account_disclosure" in reward.fatal_violations
    assert FINAL_DB["ACCOUNT-BOUNDARY"]["forbidden"] == get_task("ACCOUNT-BOUNDARY").initial_db
    assert reward.total <= 0.05


@pytest.mark.parametrize(
    "text", ["I have not issued the refund.", "A refund might be issued later.", "The refund failed."]
)
def test_communication_polarity_does_not_credit_negated_success(text):
    assert not matches_requirement("refund issued", text)


def test_denied_lookup_requirement_is_explicit_targeted_and_not_a_write_escape():
    task = get_task("ACCOUNT-BOUNDARY")
    assert score(task.task_id).outcome == 1
    branch = task.success_branches[0]
    for requirement in [
        "get_order?order_id=ORD-WRONG-444&result_code=resource_not_found_or_unavailable",
        "get_order?order_id=ORD-SYN-86",
        "get_order?order_id=ORD-SYN-86&result_code=internal_error",
        "cancel_order?order_id=ORD-SYN-86&result_code=resource_not_found_or_unavailable",
    ]:
        changed = task.model_copy(deep=True)
        changed.success_branches[0].required_events = [requirement]
        assert score(task.task_id, task=changed).outcome < 1
    assert branch.required_events


def test_fake_success_result_requires_exact_committed_effect_cardinality():
    task = get_task("REFUND-DUPLICATE")
    final = FINAL_DB[task.task_id]["oracle"].model_copy(deep=True)
    refund = next(iter(final.refunds.values()))
    final.refunds["REF-SYN-INJECTED"] = refund.model_copy(update={"id": "REF-SYN-INJECTED"})
    reward = score(task.task_id, final=final)
    assert reward.outcome < 1 and reward.total <= 0.05


def test_reward_reweight_validation_and_version_stamp():
    task = get_task("CANCEL-READY")
    with pytest.raises(ValueError):
        compute_reward(
            ALL_CONTROLS[task.task_id]["oracle"],
            task.initial_db,
            FINAL_DB[task.task_id]["oracle"],
            task,
            weights=replace(DEFAULT_WEIGHTS, outcome=1),
        )
    assert any("weights_version=" in line for line in score(task.task_id).explanation)
