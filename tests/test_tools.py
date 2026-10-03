from concurrent.futures import ThreadPoolExecutor

import pytest

from turncraft.db import CustomerServiceDB, check_referential_integrity, state_digest
from turncraft.models import EpisodeContext, OrderStatus, Severity, ToolCall, ToolErrorCode
from turncraft.task_registry import FINAL_DB, get_task
from turncraft.tools import build_default_registry


def context(task, episode="unit-tools"):
    return EpisodeContext(episode, task.task_id, task.authenticated_user_id, 8, 6)


def dispatch(task, state, name, *, registry=None, episode="unit-tools", **args):
    registry = registry or build_default_registry()
    return registry.execute(
        ToolCall(id="test-call", name=name, raw_arguments=args, source_span="test"),
        context(task, episode),
        state,
    )


def test_schema_identity_is_environment_bound():
    registry = build_default_registry()
    assert len(registry.names()) == 9
    for spec in registry.specs():
        schema = spec.args_model.model_json_schema()
        assert (
            not {"user_id", "authenticated_user_id", "episode_id", "idempotency_key"}
            & schema["properties"].keys()
        )
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_order", {}),
        ("get_order", {"order_id": "ORD-SYN-81", "user_id": "USR-VISITOR"}),
        (
            "issue_refund",
            {
                "order_id": "ORD-SYN-83",
                "payment_id": "PAY-SYN-83",
                "amount_cents": 0,
                "reason": "duplicate_charge",
            },
        ),
        (
            "issue_refund",
            {
                "order_id": "ORD-SYN-83",
                "payment_id": "PAY-SYN-83",
                "amount_cents": -1,
                "reason": "duplicate_charge",
            },
        ),
        ("create_return", {"order_id": "ORD-SYN-84", "item_id": "ITM-SYN-84", "reason": "invented"}),
    ],
)
def test_invalid_arguments_leave_world_unchanged(name, args):
    task = get_task("CANCEL-READY")
    before = state_digest(task.initial_db)
    result = dispatch(task, task.initial_db, name, **args)
    assert result.code == ToolErrorCode.INVALID_ARGUMENTS
    assert state_digest(task.initial_db) == before


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_order", {}),
        ("list_order_payments", {}),
        ("cancel_order", {"reason": "Synthetic refusal test"}),
        ("request_delivery_intercept", {"shipment_id": "SHP-SYN-86"}),
        ("create_return", {"item_id": "ITM-SYN-86", "reason": "wrong_item"}),
        ("issue_refund", {"payment_id": "PAY-SYN-86", "amount_cents": 10, "reason": "customer_request"}),
        (
            "create_replacement",
            {
                "original_item_id": "ITM-SYN-86",
                "replacement_sku": "SKU-ORGANIZER-MINT",
                "return_id": "RET-0001",
            },
        ),
        ("create_stock_notification", {"sku": "SKU-ORGANIZER-MINT"}),
    ],
)
def test_foreign_and_missing_are_publicly_indistinguishable(name, args):
    task = get_task("ACCOUNT-BOUNDARY")
    before = state_digest(task.initial_db)
    foreign = dispatch(task, task.initial_db, name, order_id="ORD-SYN-86", **args)
    missing = dispatch(task, task.initial_db, name, order_id="ORD-NONEXISTENT-999", **args)
    assert foreign.model_dump(exclude={"private_audit"}) == missing.model_dump(exclude={"private_audit"})
    assert foreign.code == ToolErrorCode.NOT_FOUND_OR_UNAVAILABLE
    assert state_digest(task.initial_db) == before


def test_cancel_idempotency_survives_registry_recreation():
    task = get_task("CANCEL-READY")
    args = dict(order_id="ORD-SYN-81", reason="Changed plans")
    assert dispatch(task, task.initial_db, "cancel_order", **args).ok
    before = state_digest(task.initial_db)
    result = dispatch(task, task.initial_db, "cancel_order", episode="another-episode", **args)
    assert result.code == ToolErrorCode.IDEMPOTENT_REPLAY
    assert state_digest(task.initial_db) == before


@pytest.mark.parametrize("task_id,number", [("DIVERT-PARCEL", 82), ("SPLIT-ARRIVAL", 87)])
def test_shipped_cancel_refused_with_dynamic_severity(task_id, number):
    task = get_task(task_id)
    call = ToolCall(
        id="cancel",
        name="cancel_order",
        raw_arguments={"order_id": f"ORD-SYN-{number}", "reason": "test"},
        source_span="test",
    )
    registry = build_default_registry()
    before = state_digest(task.initial_db)
    assert registry.severity_for(call, task.initial_db) == Severity.IRREVERSIBLE
    assert registry.execute(call, context(task), task.initial_db).code == ToolErrorCode.POLICY_DENIED
    assert state_digest(task.initial_db) == before


@pytest.mark.parametrize("refund_status", ["pending", "completed"])
def test_refund_budget_includes_committed_pending_money(refund_status):
    task = get_task("REFUND-DUPLICATE")
    args = dict(
        order_id="ORD-SYN-83", payment_id="PAY-SYN-83-EXTRA", amount_cents=2000, reason="duplicate_charge"
    )
    result = dispatch(task, task.initial_db, "issue_refund", **args)
    assert result.ok
    refund = next(iter(task.initial_db.refunds.values()))
    refund.status = type(refund.status)(refund_status)
    assert dispatch(task, task.initial_db, "issue_refund", **args).code == ToolErrorCode.IDEMPOTENT_REPLAY
    before = state_digest(task.initial_db)
    denied = dispatch(
        task, task.initial_db, "issue_refund", episode="fresh", **{**args, "amount_cents": 2001}
    )
    assert denied.code == ToolErrorCode.POLICY_DENIED
    assert state_digest(task.initial_db) == before
    assert len(task.initial_db.refunds) == 1


def test_replacement_has_one_persisted_effect_across_episodes():
    task = get_task("REPLACE-AVAILABLE")
    state = FINAL_DB[task.task_id]["oracle"].model_copy(deep=True)
    args = dict(
        order_id="ORD-SYN-84",
        original_item_id="ITM-SYN-84",
        replacement_sku="SKU-ORGANIZER-MINT",
        return_id="RET-0001",
    )
    before = state_digest(state)
    assert (
        dispatch(task, state, "create_replacement", episode="different", **args).code
        == ToolErrorCode.IDEMPOTENT_REPLAY
    )
    assert (
        dispatch(
            task,
            state,
            "create_replacement",
            episode="different",
            **{**args, "replacement_sku": "SKU-ORGANIZER-COBALT"},
        ).code
        == ToolErrorCode.POLICY_DENIED
    )
    assert state_digest(state) == before
    assert state.inventory["SKU-ORGANIZER-MINT"].available_quantity == 2


def test_multiwrite_exception_rolls_back_stock(monkeypatch):
    task = get_task("REPLACE-AVAILABLE")
    state = task.initial_db
    ret = dispatch(
        task, state, "create_return", order_id="ORD-SYN-84", item_id="ITM-SYN-84", reason="wrong_item"
    )
    before = state_digest(state)

    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic fault after inventory decrement")

    monkeypatch.setattr(CustomerServiceDB, "record_order_item", fail)
    result = dispatch(
        task,
        state,
        "create_replacement",
        order_id="ORD-SYN-84",
        original_item_id="ITM-SYN-84",
        replacement_sku="SKU-ORGANIZER-MINT",
        return_id=ret.data["return_id"],
    )
    assert result.code == ToolErrorCode.INTERNAL_ERROR
    assert state_digest(state) == before


def test_registry_is_stateless_across_independent_worlds():
    registry = build_default_registry()

    def run(index):
        task = get_task("CANCEL-READY")
        result = dispatch(
            task,
            task.initial_db,
            "cancel_order",
            registry=registry,
            episode=f"parallel-{index}",
            order_id="ORD-SYN-81",
            reason="test",
        )
        return result.ok, task.initial_db.orders["ORD-SYN-81"].status

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(ok and status == "cancelled" for ok, status in pool.map(run, range(4)))
    assert get_task("CANCEL-READY").initial_db.orders["ORD-SYN-81"].status == "pending"


def test_fixture_integrity_reset_and_diff():
    from turncraft.task_registry import TASKS

    for task in TASKS.values():
        check_referential_integrity(task.initial_db)
    task = get_task("CANCEL-READY")
    db = CustomerServiceDB(task.initial_db)
    initial = db.state.model_copy(deep=True)
    db.set_order_status("ORD-SYN-81", OrderStatus.CANCELLED)
    assert not db.diff(initial, db.state).is_empty
    db.reset(initial)
    assert db.state == initial
