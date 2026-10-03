import pytest

from turncraft.db import (
    CustomerServiceDB,
    ReferentialIntegrityError,
    check_referential_integrity,
    state_digest,
)
from turncraft.models import DatabaseState, EpisodeContext, ToolCall, ToolErrorCode
from turncraft.task_registry import get_task
from turncraft.tools import build_default_registry


@pytest.fixture
def two_accounts():
    task = get_task("REPLACE-AVAILABLE")
    foreign = get_task("ACCOUNT-BOUNDARY").initial_db
    state = task.initial_db.model_copy(deep=True)
    for field in DatabaseState.model_fields:
        for key, row in getattr(foreign, field).items():
            getattr(state, field).setdefault(key, row.model_copy(deep=True))
    check_referential_integrity(state)
    return state


@pytest.mark.parametrize(
    "field,child_id",
    [("item_ids", "ITM-SYN-86"), ("payment_ids", "PAY-SYN-86"), ("shipment_ids", "SHP-SYN-86")],
)
def test_constructor_rejects_child_listed_by_two_parents(two_accounts, field, child_id):
    state = two_accounts
    getattr(state.orders["ORD-SYN-84"], field).append(child_id)
    # Keep purchase reconciliation valid so a total mismatch cannot mask this regression.
    if field == "item_ids":
        state.order_items[child_id].unit_price_cents = 0
        state.orders["ORD-SYN-86"].total_cents = 0
    with pytest.raises(ReferentialIntegrityError, match=rf"{field}.*{child_id}.*ORD-SYN-86"):
        CustomerServiceDB(state)


@pytest.mark.parametrize("field", ["item_ids", "payment_ids", "shipment_ids"])
def test_constructor_rejects_child_missing_from_its_parent(two_accounts, field):
    getattr(two_accounts.orders["ORD-SYN-84"], field).clear()
    with pytest.raises(ReferentialIntegrityError, match="does not list this"):
        CustomerServiceDB(two_accounts)


@pytest.mark.parametrize("tool", ["get_order", "list_order_payments"])
@pytest.mark.parametrize(
    "field,child_id",
    [("item_ids", "ITM-SYN-86"), ("payment_ids", "PAY-SYN-86"), ("shipment_ids", "SHP-SYN-86")],
)
def test_read_fails_closed_when_world_is_corrupted_after_reset(two_accounts, tool, field, child_id):
    db = CustomerServiceDB(two_accounts)
    getattr(db.state.orders["ORD-SYN-84"], field).append(child_id)
    before = state_digest(db.state)
    result = build_default_registry().execute(
        ToolCall(id="read", name=tool, raw_arguments={"order_id": "ORD-SYN-84"}, source_span="test"),
        EpisodeContext("read-boundary", "synthetic", "USR-SANDBOX", 8, 6),
        db.state,
    )
    assert not result.ok
    assert result.code == ToolErrorCode.INTERNAL_ERROR
    assert result.data == {}
    assert "SYN-86" not in result.message
    assert state_digest(db.state) == before


@pytest.mark.parametrize("method", ["items_for_order", "payments_for_order", "shipments_for_order"])
@pytest.mark.parametrize("corruption", ["missing_child", "foreign_shipment_item", "foreign_payer"])
def test_read_adapter_reports_corrupted_relationships(two_accounts, method, corruption):
    db = CustomerServiceDB(two_accounts)
    if corruption == "missing_child":
        db.state.orders["ORD-SYN-84"].payment_ids.append("PAY-MISSING")
        expected = "unknown reference"
    elif corruption == "foreign_shipment_item":
        db.state.shipments["SHP-SYN-84"].item_ids.append("ITM-SYN-86")
        expected = "belongs to order"
    else:
        db.state.payments["PAY-SYN-84"].user_id = "USR-VISITOR"
        expected = "not order owner"
    with pytest.raises(ReferentialIntegrityError, match=expected):
        getattr(db, method)("ORD-SYN-84")


@pytest.mark.parametrize("method", ["items_for_order", "payments_for_order", "shipments_for_order"])
def test_valid_read_keeps_all_children_in_recorded_order(two_accounts, method):
    db = CustomerServiceDB(two_accounts)
    field = {
        "items_for_order": "item_ids",
        "payments_for_order": "payment_ids",
        "shipments_for_order": "shipment_ids",
    }[method]
    assert [row.id for row in getattr(db, method)("ORD-SYN-84")] == getattr(
        db.state.orders["ORD-SYN-84"], field
    )


def test_invalid_reset_retains_last_valid_world_and_snapshot(two_accounts):
    db = CustomerServiceDB(two_accounts)
    before = state_digest(db.state)
    snapshot = db.initial_snapshot
    malformed = two_accounts.model_copy(deep=True)
    malformed.orders["ORD-SYN-84"].payment_ids.append("PAY-SYN-86")
    with pytest.raises(ReferentialIntegrityError, match="payment_ids.*PAY-SYN-86.*ORD-SYN-86"):
        db.reset(malformed)
    assert state_digest(db.state) == before
    assert db.initial_snapshot == snapshot
