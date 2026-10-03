"""Validated in-memory commerce state, isolated snapshots and semantic diffs."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final, TypeVar, cast

from turncraft.models import (
    DatabaseState,
    DeliveryIntercept,
    InterceptStatus,
    InventoryRow,
    Notification,
    NotificationKind,
    NotificationStatus,
    Order,
    OrderItem,
    OrderStatus,
    Payment,
    PaymentStatus,
    Product,
    Refund,
    RefundReason,
    RefundStatus,
    Return,
    ReturnReason,
    ReturnStatus,
    Shipment,
    ShipmentStatus,
    User,
)

LOGGER = logging.getLogger(__name__)

# Row type of whichever collection a lookup targets, so ``_require`` returns a concrete model
# rather than Any and a mis-typed mutation is a type error instead of a runtime AttributeError.
RowT = TypeVar("RowT")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Generated ids read as RF-0001. Four digits sort lexicographically for any episode-scale
# volume, which keeps diffs and traces in creation order without a numeric sort.
ID_SEQUENCE_WIDTH: Final[int] = 4
ID_SEQUENCE_SEPARATOR: Final[str] = "-"

# 64 bits of SHA-256. This detects accidental state divergence between a live run and a
# replay; it is not a security control, so a short, readable prefix beats a full digest.
DIGEST_HEX_LENGTH: Final[int] = 16

CENTS_PER_UNIT: Final[int] = 100
MONEY_FIELD_SUFFIX: Final[str] = "_cents"
REFERENCE_FIELD_SUFFIX: Final[str] = "_id"

INITIAL_SNAPSHOT_LABEL: Final[str] = "initial"
FINAL_SNAPSHOT_LABEL: Final[str] = "final"

CHANGE_ADDED: Final[str] = "added"
CHANGE_REMOVED: Final[str] = "removed"
CHANGE_MODIFIED: Final[str] = "modified"

# Diff and validation walk collections in this order so output is stable across runs and
# machines. ``inventory`` is the one collection keyed by sku rather than by an ``id`` field.
COLLECTION_ORDER: Final[tuple[str, ...]] = (
    "users",
    "products",
    "inventory",
    "orders",
    "order_items",
    "payments",
    "shipments",
    "returns",
    "refunds",
    "delivery_intercepts",
    "notifications",
)
INVENTORY_COLLECTION: Final[str] = "inventory"

# Collections whose rows carry a derived idempotency key, scanned by ``find_by_idempotency_key``.
IDEMPOTENT_COLLECTIONS: Final[tuple[str, ...]] = (
    "returns",
    "refunds",
    "delivery_intercepts",
    "notifications",
)

# A payment in one of these states counts toward covering the order total. AUTHORIZED is
# included because the funds are committed even though they have not moved; policies.py keeps
# a deliberately narrower set for "money actually left the customer".
SETTLED_PAYMENT_STATUSES: Final[frozenset[PaymentStatus]] = frozenset(
    {
        PaymentStatus.AUTHORIZED,
        PaymentStatus.CAPTURED,
        PaymentStatus.PARTIALLY_REFUNDED,
        PaymentStatus.REFUNDED,
    }
)

# Id prefixes for rows this module creates. Fixtures use the same shapes, and the generator
# skips any candidate a fixture already claimed.
REFUND_ID_PREFIX: Final[str] = "RF"
RETURN_ID_PREFIX: Final[str] = "RET"
INTERCEPT_ID_PREFIX: Final[str] = "INT"
NOTIFICATION_ID_PREFIX: Final[str] = "NOT"
ORDER_ITEM_ID_PREFIX: Final[str] = "ITM"

# Fields rendered when a whole row is added or removed. Full rows are unreadable on a slide;
# these are the fields that tell the story of what the agent did.
_SUMMARY_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "users": ("name", "email"),
    "products": ("sku", "name", "color", "size"),
    "inventory": ("available_quantity",),
    "orders": ("status", "total_cents"),
    "order_items": ("order_id", "ordered_sku", "fulfilled_sku", "quantity"),
    "payments": ("order_id", "amount_cents", "status"),
    "shipments": ("order_id", "status", "carrier", "tracking_number"),
    "returns": ("order_id", "item_id", "reason", "status"),
    "refunds": ("payment_id", "amount_cents", "reason", "status"),
    "delivery_intercepts": ("order_id", "shipment_id", "status"),
    "notifications": ("user_id", "kind", "sku", "status"),
}


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ReferentialIntegrityError(ValueError):
    """A reference that does not resolve, or a total that does not reconcile.

    Raised by fixture validation and by mutations that would create a dangling row. The
    message names the exact field so a broken task fixture is a one-line fix, not a hunt.
    """

    def __init__(self, collection: str, entity_id: str, field: str, reason: str) -> None:
        self.collection = collection
        self.entity_id = entity_id
        self.field = field
        self.reason = reason
        super().__init__(f"{collection}[{entity_id}].{field}: {reason}")


class InventoryUnderflowError(RuntimeError):
    """A stock adjustment would drive availability below zero.

    Stock sufficiency is checked by ``policies.replacement_available`` before the tool calls
    ``adjust_inventory``. Reaching this exception means that guard was skipped; a negative
    stock level would silently corrupt every later availability check.
    """

    def __init__(self, sku: str, available: int, delta: int) -> None:
        self.sku = sku
        self.available = available
        self.delta = delta
        super().__init__(f"inventory[{sku}]: available={available} cannot absorb delta={delta}")


# --------------------------------------------------------------------------------------
# Money and value formatting
# --------------------------------------------------------------------------------------


def format_cents(cents: int) -> str:
    """Render integer cents as ``$74.00``. Arithmetic stays integral; only display divides."""
    sign = "-" if cents < 0 else ""
    whole, remainder = divmod(abs(cents), CENTS_PER_UNIT)
    return f"{sign}${whole}.{remainder:02d}"


def _display_name(field: str) -> str:
    """``payment_id`` -> ``payment``, ``amount_cents`` -> ``amount``. Diff lines, not data."""
    if field.endswith(MONEY_FIELD_SUFFIX):
        return field[: -len(MONEY_FIELD_SUFFIX)]
    if field.endswith(REFERENCE_FIELD_SUFFIX):
        return field[: -len(REFERENCE_FIELD_SUFFIX)]
    return field


def _display_value(field: str, value: Any) -> str:
    if field.endswith(MONEY_FIELD_SUFFIX) and isinstance(value, int) and not isinstance(value, bool):
        return format_cents(value)
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


# --------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------


def _canonical_json(state: DatabaseState) -> str:
    """Key-sorted JSON. Two structurally equal worlds must produce one identical string."""
    return json.dumps(state.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def state_digest(state: DatabaseState) -> str:
    """Short content hash of a world, used to prove a replay reconstructed the same state."""
    return hashlib.sha256(_canonical_json(state).encode("utf-8")).hexdigest()[:DIGEST_HEX_LENGTH]


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A point-in-time copy of the world. Construct via :meth:`capture`, never directly.

    The dataclass is frozen and ``state`` is a private deep copy, so a snapshot cannot be
    mutated by continued tool execution against the live database.
    """

    label: str
    state: DatabaseState
    digest: str

    @classmethod
    def capture(cls, label: str, state: DatabaseState) -> StateSnapshot:
        frozen = state.model_copy(deep=True)
        return cls(label=label, state=frozen, digest=state_digest(frozen))


# --------------------------------------------------------------------------------------
# Semantic diff
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One human-readable change. ``change`` is CHANGE_ADDED / REMOVED / MODIFIED."""

    collection: str
    entity_id: str
    change: str
    detail: str


@dataclass(frozen=True, slots=True)
class StateDiff:
    """Semantic difference between two worlds.

    ``print(diff)`` and ``str(diff)`` render the grouped human-readable form; ``entries``
    exposes the same information structurally for tests and for the trajectory record.
    """

    entries: tuple[DiffEntry, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    @property
    def lines(self) -> tuple[str, ...]:
        """Rendered lines including collection headers, in COLLECTION_ORDER."""
        rendered: list[str] = []
        for collection in COLLECTION_ORDER:
            grouped = [entry for entry in self.entries if entry.collection == collection]
            if not grouped:
                continue
            rendered.append(f"{collection}:")
            rendered.extend(f"  {entry.detail}" for entry in grouped)
        return tuple(rendered)

    def render(self) -> str:
        return "\n".join(self.lines)

    def __str__(self) -> str:
        return self.render()


# --------------------------------------------------------------------------------------
# Referential integrity
# --------------------------------------------------------------------------------------


def _row_key(collection: str, row: Any) -> str:
    key: str = row.sku if collection == INVENTORY_COLLECTION else row.id
    return key


def _known_skus(state: DatabaseState) -> set[str]:
    return {product.sku for product in state.products.values()}


def _check_keys_match_rows(state: DatabaseState) -> None:
    for collection in COLLECTION_ORDER:
        for key, row in getattr(state, collection).items():
            actual = _row_key(collection, row)
            if key != actual:
                raise ReferentialIntegrityError(
                    collection, key, "key", f"dictionary key does not match row identity {actual!r}"
                )


def _check_catalogue(state: DatabaseState, skus: set[str]) -> None:
    seen: dict[str, str] = {}
    for product in state.products.values():
        if not product.sku:
            raise ReferentialIntegrityError("products", product.id, "sku", "empty sku")
        if product.sku in seen:
            raise ReferentialIntegrityError(
                "products", product.id, "sku", f"sku already used by product {seen[product.sku]!r}"
            )
        seen[product.sku] = product.id
    for row in state.inventory.values():
        if row.sku not in skus:
            raise ReferentialIntegrityError("inventory", row.sku, "sku", "no product with this sku")
        if row.available_quantity < 0:
            raise ReferentialIntegrityError(
                "inventory", row.sku, "available_quantity", f"negative quantity {row.available_quantity}"
            )


def _check_order_children(state: DatabaseState, order: Order) -> None:
    for field, ids, target in (
        ("item_ids", order.item_ids, state.order_items),
        ("payment_ids", order.payment_ids, state.payments),
        ("shipment_ids", order.shipment_ids, state.shipments),
    ):
        if len(set(ids)) != len(ids):
            raise ReferentialIntegrityError("orders", order.id, field, "contains a duplicate id")
        for reference in ids:
            child = target.get(reference)
            if child is None:
                raise ReferentialIntegrityError("orders", order.id, field, f"unknown reference {reference!r}")
            if child.id != reference or child.order_id != order.id:
                raise ReferentialIntegrityError(
                    "orders",
                    order.id,
                    field,
                    f"reference {reference!r} identifies {child.id!r}, which belongs to order {child.order_id!r}, not {order.id!r}",
                )
            if field == "payment_ids" and child.user_id != order.user_id:
                raise ReferentialIntegrityError(
                    "payments",
                    child.id,
                    "user_id",
                    f"payer {child.user_id!r} is not order owner {order.user_id!r}",
                )
            if field == "shipment_ids":
                _check_shipment_items(state, child)


def _check_orders(state: DatabaseState) -> None:
    for order in state.orders.values():
        if order.user_id not in state.users:
            raise ReferentialIntegrityError("orders", order.id, "user_id", f"unknown user {order.user_id!r}")
        if order.total_cents < 0:
            raise ReferentialIntegrityError(
                "orders", order.id, "total_cents", f"negative total {order.total_cents}"
            )
        _check_order_children(state, order)


def _check_order_items(state: DatabaseState, skus: set[str]) -> None:
    for item in state.order_items.values():
        order = state.orders.get(item.order_id)
        if order is None:
            raise ReferentialIntegrityError(
                "order_items", item.id, "order_id", f"unknown order {item.order_id!r}"
            )
        if item.id not in order.item_ids:
            raise ReferentialIntegrityError(
                "order_items", item.id, "order_id", f"order {order.id!r} does not list this item"
            )
        for field, sku in (("ordered_sku", item.ordered_sku), ("fulfilled_sku", item.fulfilled_sku)):
            if sku not in skus:
                raise ReferentialIntegrityError("order_items", item.id, field, f"unknown sku {sku!r}")
        if item.quantity < 1:
            raise ReferentialIntegrityError(
                "order_items", item.id, "quantity", f"non-positive quantity {item.quantity}"
            )
        if item.unit_price_cents < 0:
            raise ReferentialIntegrityError(
                "order_items", item.id, "unit_price_cents", f"negative price {item.unit_price_cents}"
            )


def _check_payments(state: DatabaseState) -> None:
    for payment in state.payments.values():
        order = state.orders.get(payment.order_id)
        if order is None:
            raise ReferentialIntegrityError(
                "payments", payment.id, "order_id", f"unknown order {payment.order_id!r}"
            )
        if payment.id not in order.payment_ids:
            raise ReferentialIntegrityError(
                "payments", payment.id, "order_id", f"order {order.id!r} does not list this payment"
            )
        if payment.user_id not in state.users:
            raise ReferentialIntegrityError(
                "payments", payment.id, "user_id", f"unknown user {payment.user_id!r}"
            )
        if payment.user_id != order.user_id:
            raise ReferentialIntegrityError(
                "payments",
                payment.id,
                "user_id",
                f"payer {payment.user_id!r} is not order owner {order.user_id!r}",
            )
        if payment.status in SETTLED_PAYMENT_STATUSES and payment.amount_cents <= 0:
            raise ReferentialIntegrityError(
                "payments", payment.id, "amount_cents", f"settled payment with amount {payment.amount_cents}"
            )


def _check_shipment_items(state: DatabaseState, shipment: Shipment) -> None:
    for item_id in shipment.item_ids:
        item = state.order_items.get(item_id)
        if item is None:
            raise ReferentialIntegrityError("shipments", shipment.id, "item_ids", f"unknown item {item_id!r}")
        if item.order_id != shipment.order_id:
            raise ReferentialIntegrityError(
                "shipments",
                shipment.id,
                "item_ids",
                f"item {item_id!r} belongs to order {item.order_id!r}",
            )


def _check_shipments(state: DatabaseState) -> None:
    for shipment in state.shipments.values():
        order = state.orders.get(shipment.order_id)
        if order is None:
            raise ReferentialIntegrityError(
                "shipments", shipment.id, "order_id", f"unknown order {shipment.order_id!r}"
            )
        if shipment.id not in order.shipment_ids:
            raise ReferentialIntegrityError(
                "shipments", shipment.id, "order_id", f"order {order.id!r} does not list this shipment"
            )
        _check_shipment_items(state, shipment)


def _check_returns(state: DatabaseState) -> None:
    for record in state.returns.values():
        order = state.orders.get(record.order_id)
        if order is None:
            raise ReferentialIntegrityError(
                "returns", record.id, "order_id", f"unknown order {record.order_id!r}"
            )
        if record.user_id != order.user_id:
            raise ReferentialIntegrityError(
                "returns",
                record.id,
                "user_id",
                f"requester {record.user_id!r} is not order owner {order.user_id!r}",
            )
        item = state.order_items.get(record.item_id)
        if item is None:
            raise ReferentialIntegrityError(
                "returns", record.id, "item_id", f"unknown item {record.item_id!r}"
            )
        if item.order_id != record.order_id:
            raise ReferentialIntegrityError(
                "returns", record.id, "item_id", f"item {record.item_id!r} belongs to order {item.order_id!r}"
            )


def _check_refunds(state: DatabaseState) -> None:
    for refund in state.refunds.values():
        payment = state.payments.get(refund.payment_id)
        if payment is None:
            raise ReferentialIntegrityError(
                "refunds", refund.id, "payment_id", f"unknown payment {refund.payment_id!r}"
            )
        if refund.order_id not in state.orders:
            raise ReferentialIntegrityError(
                "refunds", refund.id, "order_id", f"unknown order {refund.order_id!r}"
            )
        if payment.order_id != refund.order_id:
            raise ReferentialIntegrityError(
                "refunds",
                refund.id,
                "order_id",
                f"payment {payment.id!r} belongs to order {payment.order_id!r}",
            )
        if refund.amount_cents <= 0:
            raise ReferentialIntegrityError(
                "refunds", refund.id, "amount_cents", f"non-positive amount {refund.amount_cents}"
            )
    for payment in state.payments.values():
        committed = state.refunded_cents_for_payment(payment.id)
        if committed > payment.amount_cents:
            raise ReferentialIntegrityError(
                "payments",
                payment.id,
                "amount_cents",
                f"refunds total {format_cents(committed)} exceed capture {format_cents(payment.amount_cents)}",
            )


def _check_intercepts(state: DatabaseState) -> None:
    for intercept in state.delivery_intercepts.values():
        order = state.orders.get(intercept.order_id)
        if order is None:
            raise ReferentialIntegrityError(
                "delivery_intercepts", intercept.id, "order_id", f"unknown order {intercept.order_id!r}"
            )
        shipment = state.shipments.get(intercept.shipment_id)
        if shipment is None:
            raise ReferentialIntegrityError(
                "delivery_intercepts",
                intercept.id,
                "shipment_id",
                f"unknown shipment {intercept.shipment_id!r}",
            )
        if shipment.order_id != intercept.order_id:
            raise ReferentialIntegrityError(
                "delivery_intercepts",
                intercept.id,
                "shipment_id",
                f"shipment {shipment.id!r} belongs to order {shipment.order_id!r}",
            )
        if intercept.user_id != order.user_id:
            raise ReferentialIntegrityError(
                "delivery_intercepts",
                intercept.id,
                "user_id",
                f"requester {intercept.user_id!r} is not order owner {order.user_id!r}",
            )


def _check_notifications(state: DatabaseState, skus: set[str]) -> None:
    for notification in state.notifications.values():
        if notification.user_id not in state.users:
            raise ReferentialIntegrityError(
                "notifications", notification.id, "user_id", f"unknown user {notification.user_id!r}"
            )
        if notification.order_id and notification.order_id not in state.orders:
            raise ReferentialIntegrityError(
                "notifications", notification.id, "order_id", f"unknown order {notification.order_id!r}"
            )
        if notification.sku and notification.sku not in skus:
            raise ReferentialIntegrityError(
                "notifications", notification.id, "sku", f"unknown sku {notification.sku!r}"
            )


def _check_order_totals(state: DatabaseState) -> None:
    """Line items are the source of truth for the total; settled payments must cover it.

    Two payment shapes reconcile, and the disjunction is deliberate:
      * one payment equal to the total  -- the normal single capture, and the shape a
        duplicate rides on (two captures each equal to the whole total);
      * payments summing to the total   -- a split capture.
    Anything else is an incoherent fixture, and an incoherent fixture makes the refund
    verifier untrustworthy.
    """
    for order in state.orders.values():
        if order.item_ids:
            line_total = sum(
                state.order_items[item_id].quantity * state.order_items[item_id].unit_price_cents
                for item_id in order.item_ids
            )
            if line_total != order.total_cents:
                raise ReferentialIntegrityError(
                    "orders",
                    order.id,
                    "total_cents",
                    f"line items total {format_cents(line_total)}, order says {format_cents(order.total_cents)}",
                )
        settled = [
            state.payments[payment_id]
            for payment_id in order.payment_ids
            if state.payments[payment_id].status in SETTLED_PAYMENT_STATUSES
        ]
        if not settled or order.total_cents == 0:
            continue
        amounts = [payment.amount_cents for payment in settled]
        if order.total_cents in amounts or sum(amounts) == order.total_cents:
            continue
        raise ReferentialIntegrityError(
            "orders",
            order.id,
            "payment_ids",
            f"settled payments {[format_cents(amount) for amount in amounts]} reconcile with neither a single "
            f"capture of {format_cents(order.total_cents)} nor a split summing to it",
        )


def check_referential_integrity(state: DatabaseState) -> None:
    """Validate a world in place. Raises :class:`ReferentialIntegrityError` on the first fault.

    Module-level so task fixtures can be validated at authoring time without constructing a
    database. ``CustomerServiceDB.validate_referential_integrity`` delegates here.
    """
    skus = _known_skus(state)
    _check_keys_match_rows(state)
    _check_catalogue(state, skus)
    _check_orders(state)
    _check_order_items(state, skus)
    _check_payments(state)
    _check_shipments(state)
    _check_returns(state)
    _check_refunds(state)
    _check_intercepts(state)
    _check_notifications(state, skus)
    _check_order_totals(state)


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------


class CustomerServiceDB:
    """One episode's mutable world, plus the snapshot/diff machinery around it."""

    def __init__(self, initial_state: DatabaseState | None = None) -> None:
        self._state = DatabaseState()
        self._initial_snapshot = StateSnapshot.capture(INITIAL_SNAPSHOT_LABEL, self._state)
        self._id_counters: dict[str, int] = {}
        self._idempotency_ledger: dict[str, str] = {}
        if initial_state is not None:
            self.reset(initial_state)

    # -- lifecycle ---------------------------------------------------------------------

    def reset(self, state: DatabaseState) -> StateSnapshot:
        """Load a fixture as a deep copy and return the initial snapshot.

        The deep copy is the whole point: the fixture object is shared across every episode
        of a task, so a shallow copy would let episode N's refund appear in episode N+1's
        starting world and silently invalidate the reward.
        """
        candidate = state.model_copy(deep=True)
        check_referential_integrity(candidate)
        self._state = candidate
        self._id_counters = {}
        self._idempotency_ledger = {}
        self._initial_snapshot = StateSnapshot.capture(INITIAL_SNAPSHOT_LABEL, self._state)
        return self._initial_snapshot

    @property
    def state(self) -> DatabaseState:
        """The live, mutable world. Tools mutate through the methods below, not through this."""
        return self._state

    @property
    def initial_snapshot(self) -> StateSnapshot:
        """Snapshot captured by the most recent :meth:`reset`."""
        return self._initial_snapshot

    def validate_referential_integrity(self, state: DatabaseState | None = None) -> None:
        """Validate the live world, or an explicit one. Raises :class:`ReferentialIntegrityError`."""
        check_referential_integrity(self._state if state is None else state)

    def snapshot(self, label: str = FINAL_SNAPSHOT_LABEL) -> StateSnapshot:
        """Deep-copy the live world into an immutable, digested snapshot."""
        return StateSnapshot.capture(label, self._state)

    @staticmethod
    def diff(before: StateSnapshot | DatabaseState, after: StateSnapshot | DatabaseState) -> StateDiff:
        """Semantic difference between two worlds, grouped by collection.

        Renders as, for example::

            refunds:
              + RF-0001: payment=PAY-DUP, amount=$74.00, reason=duplicate_charge
            payments:
              PAY-DUP.status: captured -> refunded

        A JSON dump would technically contain this, and nobody would read it during a demo.
        """
        start = before.state if isinstance(before, StateSnapshot) else before
        end = after.state if isinstance(after, StateSnapshot) else after
        entries: list[DiffEntry] = []
        for collection in COLLECTION_ORDER:
            entries.extend(_diff_collection(collection, getattr(start, collection), getattr(end, collection)))
        return StateDiff(entries=tuple(entries))

    # -- typed accessors ---------------------------------------------------------------
    # Every accessor returns None / [] for an unknown id rather than raising: "does not
    # exist" and "not yours" must reach the assistant as the same public error code, and a
    # caller that has to catch KeyError to build that result will eventually forget.

    def user(self, user_id: str) -> User | None:
        return self._state.users.get(user_id)

    def order(self, order_id: str) -> Order | None:
        return self._state.orders.get(order_id)

    def order_item(self, item_id: str) -> OrderItem | None:
        return self._state.order_items.get(item_id)

    def payment(self, payment_id: str) -> Payment | None:
        return self._state.payments.get(payment_id)

    def shipment(self, shipment_id: str) -> Shipment | None:
        return self._state.shipments.get(shipment_id)

    def refund(self, refund_id: str) -> Refund | None:
        return self._state.refunds.get(refund_id)

    def return_record(self, return_id: str) -> Return | None:
        return self._state.returns.get(return_id)

    def intercept(self, intercept_id: str) -> DeliveryIntercept | None:
        return self._state.delivery_intercepts.get(intercept_id)

    def inventory(self, sku: str) -> InventoryRow | None:
        return self._state.inventory.get(sku)

    def product_by_sku(self, sku: str) -> Product | None:
        for product in self._state.products.values():
            if product.sku == sku:
                return product
        return None

    def products_of_type(self, product_type: str) -> list[Product]:
        """Catalogue search seam for ``search_inventory``; ordered by sku for determinism."""
        matches = [
            product for product in self._state.products.values() if product.product_type == product_type
        ]
        return sorted(matches, key=lambda product: product.sku)

    def items_for_order(self, order_id: str) -> list[OrderItem]:
        return self._ordered_children(order_id, "item_ids", self._state.order_items)

    def payments_for_order(self, order_id: str) -> list[Payment]:
        """Payments in the order's recorded sequence. That order is the only chronology the
        fixture carries, and duplicate-capture detection depends on which capture came first."""
        return self._ordered_children(order_id, "payment_ids", self._state.payments)

    def shipments_for_order(self, order_id: str) -> list[Shipment]:
        return self._ordered_children(order_id, "shipment_ids", self._state.shipments)

    def refunds_for_payment(self, payment_id: str) -> list[Refund]:
        matches = [refund for refund in self._state.refunds.values() if refund.payment_id == payment_id]
        return sorted(matches, key=lambda refund: refund.id)

    def intercepts_for_shipment(self, shipment_id: str) -> list[DeliveryIntercept]:
        matches = [
            intercept
            for intercept in self._state.delivery_intercepts.values()
            if intercept.shipment_id == shipment_id
        ]
        return sorted(matches, key=lambda intercept: intercept.id)

    def notifications_for_user(self, user_id: str) -> list[Notification]:
        matches = [row for row in self._state.notifications.values() if row.user_id == user_id]
        return sorted(matches, key=lambda row: row.id)

    def refunded_cents_for_payment(self, payment_id: str) -> int:
        """Delegates to the contract so the tool guard and the reward verifier cannot diverge."""
        return self._state.refunded_cents_for_payment(payment_id)

    def _ordered_children(self, order_id: str, field: str, table: dict[str, Any]) -> list[Any]:
        order = self._state.orders.get(order_id)
        if order is None:
            return []
        # Worlds are mutable after reset; validate the whole parent before exposing any child.
        _check_order_children(self._state, order)
        return [table[child_id] for child_id in getattr(order, field)]

    # -- idempotency ledger ------------------------------------------------------------

    def find_by_idempotency_key(self, key: str) -> Refund | Return | DeliveryIntercept | Notification | None:
        """The row a previous call with this key created, if any.

        Keys are derived environment-side from (tool, args, episode), so a model that retries
        produces the same key and finds its own earlier row instead of a second side effect.
        """
        if not key:
            return None
        for collection in IDEMPOTENT_COLLECTIONS:
            for row in getattr(self._state, collection).values():
                if row.idempotency_key == key:
                    return cast("Refund | Return | DeliveryIntercept | Notification", row)
        return None

    def record_idempotency_key(self, key: str, effect: str) -> None:
        """Remember a keyed effect that creates no row of its own, such as an order cancellation."""
        if not key:
            LOGGER.warning("Refusing to record an empty idempotency key for effect %r.", effect)
            return
        self._idempotency_ledger[key] = effect

    def effect_for_idempotency_key(self, key: str) -> str | None:
        return self._idempotency_ledger.get(key) if key else None

    # -- mutations ---------------------------------------------------------------------
    # Policy lives in policies.py; these do what they are told. The only refusals here are
    # structural: an unresolvable reference, or stock going negative.

    def set_order_status(self, order_id: str, status: OrderStatus) -> Order:
        order = self._require(self._state.orders, order_id, "orders", "id")
        order.status = status
        return order

    def set_payment_status(self, payment_id: str, status: PaymentStatus) -> Payment:
        payment = self._require(self._state.payments, payment_id, "payments", "id")
        payment.status = status
        return payment

    def set_shipment_status(self, shipment_id: str, status: ShipmentStatus) -> Shipment:
        shipment = self._require(self._state.shipments, shipment_id, "shipments", "id")
        shipment.status = status
        return shipment

    def adjust_inventory(self, sku: str, delta: int) -> InventoryRow:
        """Move stock by ``delta``. Raises :class:`InventoryUnderflowError` below zero."""
        row = self._require(self._state.inventory, sku, "inventory", "sku")
        updated = row.available_quantity + delta
        if updated < 0:
            raise InventoryUnderflowError(sku=sku, available=row.available_quantity, delta=delta)
        row.available_quantity = updated
        return row

    def record_refund(
        self,
        *,
        payment_id: str,
        order_id: str,
        amount_cents: int,
        reason: RefundReason,
        idempotency_key: str,
        status: RefundStatus = RefundStatus.COMPLETED,
    ) -> Refund:
        """Insert a refund and re-derive the payment's status from committed refunds.

        The payment status is derived here, in one place, because a refund row and a stale
        ``captured`` payment status would let the over-refund verifier and the tool guard read
        the same world differently.
        """
        payment = self._require(self._state.payments, payment_id, "refunds", "payment_id")
        self._require(self._state.orders, order_id, "refunds", "order_id")
        refund = Refund(
            id=self._next_id(REFUND_ID_PREFIX, self._state.refunds),
            payment_id=payment_id,
            order_id=order_id,
            amount_cents=amount_cents,
            reason=reason,
            status=status,
            idempotency_key=idempotency_key,
        )
        self._state.refunds[refund.id] = refund
        committed = self._state.refunded_cents_for_payment(payment_id)
        if committed >= payment.amount_cents:
            payment.status = PaymentStatus.REFUNDED
        elif committed > 0:
            payment.status = PaymentStatus.PARTIALLY_REFUNDED
        return refund

    def record_return(
        self,
        *,
        order_id: str,
        user_id: str,
        item_id: str,
        reason: ReturnReason,
        idempotency_key: str,
        status: ReturnStatus = ReturnStatus.AUTHORIZED,
    ) -> Return:
        self._require(self._state.orders, order_id, "returns", "order_id")
        self._require(self._state.order_items, item_id, "returns", "item_id")
        record = Return(
            id=self._next_id(RETURN_ID_PREFIX, self._state.returns),
            order_id=order_id,
            user_id=user_id,
            item_id=item_id,
            reason=reason,
            status=status,
            idempotency_key=idempotency_key,
        )
        self._state.returns[record.id] = record
        return record

    def record_intercept(
        self,
        *,
        order_id: str,
        shipment_id: str,
        user_id: str,
        idempotency_key: str,
        status: InterceptStatus = InterceptStatus.REQUESTED,
    ) -> DeliveryIntercept:
        self._require(self._state.orders, order_id, "delivery_intercepts", "order_id")
        self._require(self._state.shipments, shipment_id, "delivery_intercepts", "shipment_id")
        intercept = DeliveryIntercept(
            id=self._next_id(INTERCEPT_ID_PREFIX, self._state.delivery_intercepts),
            order_id=order_id,
            shipment_id=shipment_id,
            user_id=user_id,
            status=status,
            idempotency_key=idempotency_key,
        )
        self._state.delivery_intercepts[intercept.id] = intercept
        return intercept

    def record_notification(
        self,
        *,
        user_id: str,
        kind: NotificationKind,
        idempotency_key: str,
        order_id: str = "",
        sku: str = "",
        status: NotificationStatus = NotificationStatus.REGISTERED,
    ) -> Notification:
        self._require(self._state.users, user_id, "notifications", "user_id")
        notification = Notification(
            id=self._next_id(NOTIFICATION_ID_PREFIX, self._state.notifications),
            user_id=user_id,
            kind=kind,
            status=status,
            order_id=order_id,
            sku=sku,
            idempotency_key=idempotency_key,
        )
        self._state.notifications[notification.id] = notification
        return notification

    def record_order_item(
        self,
        *,
        order_id: str,
        ordered_sku: str,
        fulfilled_sku: str,
        quantity: int = 1,
        unit_price_cents: int = 0,
        item_id: str | None = None,
    ) -> OrderItem:
        """Attach a line item to an order, e.g. the replacement unit ``create_replacement`` ships.

        This deliberately does not touch ``order.total_cents``: a like-for-like replacement is
        not a second sale, and rewriting the total would break the duplicate-capture reading of
        the payment history. Fixture reconciliation is validated at reset, not after mutations.
        """
        order = self._require(self._state.orders, order_id, "order_items", "order_id")
        resolved_item_id = item_id or self._next_id(ORDER_ITEM_ID_PREFIX, self._state.order_items)
        if resolved_item_id in self._state.order_items:
            raise ReferentialIntegrityError(
                "order_items", resolved_item_id, "id", "an order item with this id already exists"
            )
        item = OrderItem(
            id=resolved_item_id,
            order_id=order_id,
            ordered_sku=ordered_sku,
            fulfilled_sku=fulfilled_sku,
            quantity=quantity,
            unit_price_cents=unit_price_cents,
        )
        self._state.order_items[item.id] = item
        order.item_ids.append(item.id)
        return item

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _require(table: dict[str, RowT], key: str, collection: str, field: str) -> RowT:
        row = table.get(key)
        if row is None:
            raise ReferentialIntegrityError(collection, key, field, "no such row")
        return row

    def _next_id(self, prefix: str, table: dict[str, Any]) -> str:
        """Deterministic per-episode id. Skips anything a fixture already claimed."""
        counter = self._id_counters.get(prefix, 0)
        while True:
            counter += 1
            candidate = f"{prefix}{ID_SEQUENCE_SEPARATOR}{counter:0{ID_SEQUENCE_WIDTH}d}"
            if candidate not in table:
                self._id_counters[prefix] = counter
                return candidate


def _summarise_row(collection: str, payload: dict[str, Any]) -> str:
    fields = _SUMMARY_FIELDS.get(collection, ())
    parts = [
        f"{_display_name(field)}={_display_value(field, payload[field])}"
        for field in fields
        if field in payload
    ]
    return ", ".join(parts)


def _diff_collection(collection: str, before: dict[str, Any], after: dict[str, Any]) -> Iterable[DiffEntry]:
    entries: list[DiffEntry] = []
    for key in sorted(set(after) - set(before)):
        detail = _summarise_row(collection, after[key].model_dump(mode="json"))
        entries.append(DiffEntry(collection, key, CHANGE_ADDED, f"+ {key}: {detail}"))
    for key in sorted(set(before) - set(after)):
        detail = _summarise_row(collection, before[key].model_dump(mode="json"))
        entries.append(DiffEntry(collection, key, CHANGE_REMOVED, f"- {key}: {detail}"))
    for key in sorted(set(before) & set(after)):
        old = before[key].model_dump(mode="json")
        new = after[key].model_dump(mode="json")
        for field in sorted(old):
            if old[field] == new.get(field):
                continue
            detail = f"{key}.{field}: {_display_value(field, old[field])} -> {_display_value(field, new.get(field))}"
            entries.append(DiffEntry(collection, key, CHANGE_MODIFIED, detail))
    return entries


__all__ = [
    "CHANGE_ADDED",
    "CHANGE_MODIFIED",
    "CHANGE_REMOVED",
    "COLLECTION_ORDER",
    "CustomerServiceDB",
    "DIGEST_HEX_LENGTH",
    "DiffEntry",
    "FINAL_SNAPSHOT_LABEL",
    "INITIAL_SNAPSHOT_LABEL",
    "InventoryUnderflowError",
    "ReferentialIntegrityError",
    "SETTLED_PAYMENT_STATUSES",
    "StateDiff",
    "StateSnapshot",
    "check_referential_integrity",
    "format_cents",
    "state_digest",
]
