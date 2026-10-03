"""Composable synthetic world builders and deterministic filler. No external records are included."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Final

from turncraft.db import COLLECTION_ORDER, ReferentialIntegrityError, check_referential_integrity
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

PRODUCT_ID_PREFIX: Final[str] = "P-"

# --------------------------------------------------------------------------------------
# Row builders
# --------------------------------------------------------------------------------------


def make_user(user_id: str, name: str, *, email: str = "", shipping_address: str = "") -> DatabaseState:
    """One user row. ``email`` defaults to ``<user_id>@example.invalid`` (lowercased) when omitted."""
    resolved_email = email or f"{user_id.lower()}@example.invalid"
    user = User(id=user_id, name=name, email=resolved_email, shipping_address=shipping_address)
    return DatabaseState(users={user.id: user})


def make_product(
    sku: str,
    name: str,
    *,
    product_id: str = "",
    color: str = "",
    size: str = "",
    product_type: str = "",
    unit_price_cents: int = 0,
) -> DatabaseState:
    """One catalogue row. ``products`` is keyed by the product's own id, not its sku -- that is
    ``inventory``'s job. ``product_id`` defaults to ``P-<sku>``, unique whenever skus are."""
    resolved_id = product_id or f"{PRODUCT_ID_PREFIX}{sku}"
    product = Product(
        id=resolved_id,
        sku=sku,
        name=name,
        product_type=product_type,
        color=color,
        size=size,
        unit_price_cents=unit_price_cents,
    )
    return DatabaseState(products={product.id: product})


def make_inventory(sku: str, available_quantity: int) -> DatabaseState:
    """One stock row, keyed by sku."""
    row = InventoryRow(sku=sku, available_quantity=available_quantity)
    return DatabaseState(inventory={row.sku: row})


def make_order_item(
    item_id: str,
    order_id: str,
    ordered_sku: str,
    *,
    fulfilled_sku: str = "",
    quantity: int = 1,
    unit_price_cents: int = 0,
) -> DatabaseState:
    """One line item. ``fulfilled_sku`` defaults to ``ordered_sku``; pass it explicitly only for
    the wrong-item-shipped shape, where the two differ."""
    item = OrderItem(
        id=item_id,
        order_id=order_id,
        ordered_sku=ordered_sku,
        fulfilled_sku=fulfilled_sku or ordered_sku,
        quantity=quantity,
        unit_price_cents=unit_price_cents,
    )
    return DatabaseState(order_items={item.id: item})


def make_order(
    order_id: str,
    user_id: str,
    status: OrderStatus | str,
    *,
    item_ids: Sequence[str] = (),
    payment_ids: Sequence[str] = (),
    shipment_ids: Sequence[str] = (),
    total_cents: int = 0,
    placed_at: str = "",
) -> DatabaseState:
    """One order header. Once composed, ``total_cents`` must equal
    ``sum(item.quantity * item.unit_price_cents for item in item_ids)`` -- build the line items
    first and pass the total you actually built; ``compose()`` is what checks it, this is not."""
    order = Order(
        id=order_id,
        user_id=user_id,
        status=OrderStatus(status),
        item_ids=list(item_ids),
        payment_ids=list(payment_ids),
        shipment_ids=list(shipment_ids),
        total_cents=total_cents,
        placed_at=placed_at,
    )
    return DatabaseState(orders={order.id: order})


def make_payment(
    payment_id: str,
    order_id: str,
    user_id: str,
    amount_cents: int,
    status: PaymentStatus | str,
    *,
    processor_reference: str = "",
) -> DatabaseState:
    """One payment row. ``user_id`` must match the order's owner once composed."""
    payment = Payment(
        id=payment_id,
        order_id=order_id,
        user_id=user_id,
        amount_cents=amount_cents,
        status=PaymentStatus(status),
        processor_reference=processor_reference,
    )
    return DatabaseState(payments={payment.id: payment})


def make_shipment(
    shipment_id: str,
    order_id: str,
    status: ShipmentStatus | str,
    *,
    carrier: str = "",
    tracking_number: str = "",
    item_ids: Sequence[str] = (),
) -> DatabaseState:
    """One shipment row. Every id in ``item_ids`` must belong to ``order_id`` once composed."""
    shipment = Shipment(
        id=shipment_id,
        order_id=order_id,
        status=ShipmentStatus(status),
        carrier=carrier,
        tracking_number=tracking_number,
        item_ids=list(item_ids),
    )
    return DatabaseState(shipments={shipment.id: shipment})


def make_refund(
    refund_id: str,
    payment_id: str,
    order_id: str,
    amount_cents: int,
    reason: RefundReason | str,
    status: RefundStatus | str = RefundStatus.COMPLETED,
    *,
    idempotency_key: str = "",
) -> DatabaseState:
    """One refund row. Once composed, COMPLETED/PENDING refunds against ``payment_id`` must not
    sum past that payment's ``amount_cents`` -- see ``DatabaseState.refunded_cents_for_payment``."""
    refund = Refund(
        id=refund_id,
        payment_id=payment_id,
        order_id=order_id,
        amount_cents=amount_cents,
        reason=RefundReason(reason),
        status=RefundStatus(status),
        idempotency_key=idempotency_key,
    )
    return DatabaseState(refunds={refund.id: refund})


def make_return(
    return_id: str,
    order_id: str,
    user_id: str,
    item_id: str,
    reason: ReturnReason | str,
    status: ReturnStatus | str = ReturnStatus.REQUESTED,
    *,
    idempotency_key: str = "",
) -> DatabaseState:
    """One RMA row. ``user_id`` must match the order's owner and ``item_id`` must belong to
    ``order_id`` once composed."""
    record = Return(
        id=return_id,
        order_id=order_id,
        user_id=user_id,
        item_id=item_id,
        reason=ReturnReason(reason),
        status=ReturnStatus(status),
        idempotency_key=idempotency_key,
    )
    return DatabaseState(returns={record.id: record})


def make_delivery_intercept(
    intercept_id: str,
    order_id: str,
    shipment_id: str,
    user_id: str,
    status: InterceptStatus | str = InterceptStatus.REQUESTED,
    *,
    idempotency_key: str = "",
) -> DatabaseState:
    """One delivery-intercept row. ``shipment_id`` must belong to ``order_id`` and ``user_id``
    must match the order's owner once composed."""
    intercept = DeliveryIntercept(
        id=intercept_id,
        order_id=order_id,
        shipment_id=shipment_id,
        user_id=user_id,
        status=InterceptStatus(status),
        idempotency_key=idempotency_key,
    )
    return DatabaseState(delivery_intercepts={intercept.id: intercept})


def make_notification(
    notification_id: str,
    user_id: str,
    kind: NotificationKind | str,
    *,
    order_id: str = "",
    sku: str = "",
    status: NotificationStatus | str = NotificationStatus.REGISTERED,
    idempotency_key: str = "",
) -> DatabaseState:
    """One synthetic notification row."""
    notification = Notification(
        id=notification_id,
        user_id=user_id,
        kind=NotificationKind(kind),
        status=NotificationStatus(status),
        order_id=order_id,
        sku=sku,
        idempotency_key=idempotency_key,
    )
    return DatabaseState(notifications={notification.id: notification})


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def _merge_fragments(fragments: Sequence[DatabaseState]) -> dict[str, dict[str, Any]]:
    """Dict-union every collection across ``fragments``, in ``db.COLLECTION_ORDER``.

    A key defined by two fragments is almost always a copy-paste bug (the same order id built
    twice with different statuses) rather than an intentional overwrite, so it is a hard error
    here -- "last write wins" would silently discard one of the writes and nobody would notice
    until the reward disagreed with the fixture. Deliberately does not run the full
    referential-integrity sweep: :func:`realistic_filler` merges its own rows through this
    (they reference each other but not yet the caller's user row) before :func:`compose`
    validates the assembled whole.
    """
    merged: dict[str, dict[str, Any]] = {name: {} for name in COLLECTION_ORDER}
    for fragment in fragments:
        for name in COLLECTION_ORDER:
            for key, row in getattr(fragment, name).items():
                if key in merged[name]:
                    raise ReferentialIntegrityError(
                        name, key, "id", "defined by more than one fixture fragment"
                    )
                merged[name][key] = row
    return merged


def compose(*fragments: DatabaseState) -> DatabaseState:
    """Merge builder fragments into one validated world.

    Every collection is a plain dict union (see :func:`_merge_fragments`); a duplicate id
    across fragments is a hard error there. The full referential-integrity sweep
    (``db.check_referential_integrity``) runs last, so a dangling reference is reported with
    the exact collection/id/field/reason, the same vocabulary whether the break is a duplicate
    row or an order pointing at a payment that was never built.
    """
    state = DatabaseState(**_merge_fragments(fragments))
    check_referential_integrity(state)
    return state


# --------------------------------------------------------------------------------------
# Realism filler
# --------------------------------------------------------------------------------------

# Enough rows to prove the world is not just the two under test, cheap enough to read on stage.
DEFAULT_FILLER_ORDER_COUNT: Final[int] = 3
FILLER_ID_WIDTH: Final[int] = 4  # matches db.py's generated-id look: FILL0-ORD-0001
FILLER_MIN_PRICE_CENTS: Final[int] = 1500  # $15
FILLER_MAX_PRICE_CENTS: Final[int] = 24000  # $240, exclusive upper bound of the draw
FILLER_PRICE_STEP_CENTS: Final[int] = 100  # whole-dollar prices read as a real catalogue
FILLER_MAX_STOCK_UNITS: Final[int] = 12  # exclusive upper bound of the stock draw

# (order status, payment status, shipment status | None). Cycled by position, not drawn from
# the RNG, so the *shape* of the filler world never depends on the seed -- only cosmetics do.
_FillerProfile = tuple[OrderStatus, PaymentStatus, ShipmentStatus | None]
_FILLER_PROFILES: Final[tuple[_FillerProfile, ...]] = (
    (OrderStatus.DELIVERED, PaymentStatus.CAPTURED, ShipmentStatus.DELIVERED),
    (OrderStatus.PROCESSING, PaymentStatus.CAPTURED, None),
    (OrderStatus.SHIPPED, PaymentStatus.CAPTURED, ShipmentStatus.IN_TRANSIT),
    (OrderStatus.CANCELLED, PaymentStatus.VOIDED, None),
    (OrderStatus.CONFIRMED, PaymentStatus.AUTHORIZED, None),
)
_FILLER_PRODUCT_NOUNS: Final[tuple[str, ...]] = (
    "Wire Tray",
    "Card Index",
    "Fabric Pouch",
    "Binder Clips",
    "Modular Stand",
    "Cable Ties",
    "Desk Riser",
    "Zip Sleeve",
)
_FILLER_CARRIERS: Final[tuple[str, ...]] = ("SyntheticRoute-A", "SyntheticRoute-B", "SyntheticRoute-C")


def realistic_filler(user_id: str, seed: int, *, count: int = DEFAULT_FILLER_ORDER_COUNT) -> DatabaseState:
    """A handful of unrelated, self-contained orders for ``user_id``.

    Purely cosmetic ballast: a task world consisting solely of the one order under test is a
    tell -- "only one order exists, so it must be the one" is a shortcut a policy can learn
    that has nothing to do with the skill being tested. These rows carry their own catalogue,
    namespaced ``FILL<seed>-...`` so they can never collide with a task's own ids.

    Returns an unvalidated fragment, like every ``make_*`` builder: it does not create the
    user row (the task fixture already owns that identity) and its orders reference a user
    that does not exist yet from this function's point of view, so it cannot be integrity-
    checked in isolation. Compose it alongside the user that owns it:

        compose(make_user(user_id, "..."), realistic_filler(user_id, seed=7), ...)

    Deterministic in ``seed``: the *shape* of each order (its status/payment/shipment profile)
    cycles positionally through ``_FILLER_PROFILES``, so structure never depends on the RNG
    draw. Only cosmetic values -- price, product name, tracking number, carrier, stock level --
    come from ``random.Random(seed)``. Same seed, same digest, always: no wall-clock, no
    module-global ``random.random()``.
    """
    if count < 1:
        raise ValueError(f"realistic_filler count must be >= 1, got {count}")
    rng = random.Random(seed)
    namespace = f"FILL{seed}"
    fragments: list[DatabaseState] = []
    for position in range(count):
        ordinal = f"{position + 1:0{FILLER_ID_WIDTH}d}"
        status, payment_status, shipment_status = _FILLER_PROFILES[position % len(_FILLER_PROFILES)]
        sku = f"{namespace}-SKU-{ordinal}"
        order_id = f"{namespace}-ORD-{ordinal}"
        item_id = f"{namespace}-ITM-{ordinal}"
        payment_id = f"{namespace}-PAY-{ordinal}"
        price_cents = rng.randrange(FILLER_MIN_PRICE_CENTS, FILLER_MAX_PRICE_CENTS, FILLER_PRICE_STEP_CENTS)

        fragments.append(
            make_product(
                sku,
                rng.choice(_FILLER_PRODUCT_NOUNS),
                product_id=f"{namespace}-PROD-{ordinal}",
                unit_price_cents=price_cents,
            )
        )
        fragments.append(make_inventory(sku, rng.randrange(0, FILLER_MAX_STOCK_UNITS)))
        fragments.append(
            make_order_item(
                item_id, order_id, sku, fulfilled_sku=sku, quantity=1, unit_price_cents=price_cents
            )
        )
        fragments.append(
            make_payment(
                payment_id,
                order_id,
                user_id,
                price_cents,
                payment_status,
                processor_reference=f"{namespace}-PSP-{ordinal}",
            )
        )

        shipment_ids: list[str] = []
        if shipment_status is not None:
            shipment_id = f"{namespace}-SHP-{ordinal}"
            fragments.append(
                make_shipment(
                    shipment_id,
                    order_id,
                    shipment_status,
                    carrier=rng.choice(_FILLER_CARRIERS),
                    tracking_number=f"{namespace}{rng.randrange(10**8, 10**9)}",
                    item_ids=[item_id],
                )
            )
            shipment_ids = [shipment_id]

        fragments.append(
            make_order(
                order_id,
                user_id,
                status,
                item_ids=[item_id],
                payment_ids=[payment_id],
                shipment_ids=shipment_ids,
                total_cents=price_cents,
            )
        )
    return DatabaseState(**_merge_fragments(fragments))


__all__ = [
    "DEFAULT_FILLER_ORDER_COUNT",
    "FILLER_ID_WIDTH",
    "FILLER_MAX_PRICE_CENTS",
    "FILLER_MAX_STOCK_UNITS",
    "FILLER_MIN_PRICE_CENTS",
    "FILLER_PRICE_STEP_CENTS",
    "PRODUCT_ID_PREFIX",
    "ReferentialIntegrityError",
    "compose",
    "make_delivery_intercept",
    "make_inventory",
    "make_notification",
    "make_order",
    "make_order_item",
    "make_payment",
    "make_product",
    "make_refund",
    "make_return",
    "make_shipment",
    "make_user",
    "realistic_filler",
]
