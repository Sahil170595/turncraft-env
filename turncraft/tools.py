"""Nine validated, identity-bound commerce tools with atomic dispatch and persisted idempotency."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from pydantic import BaseModel, Field, ValidationError

from turncraft import policies
from turncraft.db import CustomerServiceDB, format_cents, state_digest
from turncraft.models import (
    ContractModel,
    DatabaseState,
    DeliveryIntercept,
    EpisodeContext,
    Notification,
    NotificationKind,
    NotificationStatus,
    Order,
    OrderStatus,
    Product,
    Refund,
    RefundReason,
    Return,
    ReturnReason,
    ReturnStatus,
    Severity,
    Shipment,
    ShipmentStatus,
    ToolCall,
    ToolErrorCode,
    ToolResult,
)
from turncraft.tool_protocol import ToolSpecLike, render_tool_catalogue

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Bounds a free-text argument (cancel_order's reason) so a runaway reply cannot bloat every
# persisted trajectory event; generous enough that no legitimate customer sentence hits it.
_MAX_REASON_TEXT_CHARS: Final[int] = 500

# Mirrors tool_protocol.MAX_ERROR_SPAN_CHARS: long enough to show a real validation failure
# in full, short enough that a pathological payload cannot bloat the trajectory.
_MAX_VALIDATION_MESSAGE_CHARS: Final[int] = 600

# Tool-owned success codes (ToolErrorCode covers only registry/policy failures). Exported so
# the reward layer can match on an exact string instead of guessing one.
CODE_ORDER_FOUND: Final[str] = "order_found"
CODE_PAYMENTS_LISTED: Final[str] = "payments_listed"
CODE_INVENTORY_SEARCHED: Final[str] = "inventory_searched"
CODE_ORDER_CANCELLED: Final[str] = "order_cancelled"
CODE_INTERCEPT_REQUESTED: Final[str] = "intercept_requested"
CODE_RETURN_CREATED: Final[str] = "return_created"
CODE_REFUND_ISSUED: Final[str] = "refund_issued"
CODE_REPLACEMENT_CREATED: Final[str] = "replacement_created"
CODE_NOTIFICATION_REGISTERED: Final[str] = "notification_registered"

# private_audit["reason"] vocabulary. Never rendered to the model;
# exported so the reward layer -- which must tell an ownership denial from a plain miss for
# authorization checks -- can match on an exact string instead of a free-text message.
AUDIT_REASON_ORDER_NOT_FOUND: Final[str] = "order_not_found"
AUDIT_REASON_OWNERSHIP_DENIED: Final[str] = "ownership_denied"
AUDIT_REASON_SHIPMENT_NOT_FOUND: Final[str] = "shipment_not_found_for_order"
AUDIT_REASON_ITEM_NOT_FOUND: Final[str] = "item_not_found_for_order"
AUDIT_REASON_PAYMENT_NOT_FOUND: Final[str] = "payment_not_found_for_order"
AUDIT_REASON_RETURN_NOT_FOUND: Final[str] = "return_not_found_for_item"
AUDIT_REASON_SKU_NOT_FOUND: Final[str] = "sku_not_found"
AUDIT_REASON_NOT_CANCELLABLE: Final[str] = "not_cancellable"
AUDIT_REASON_NOT_INTERCEPT_ELIGIBLE: Final[str] = "not_intercept_eligible"
AUDIT_REASON_RETURN_ALREADY_OPEN: Final[str] = "return_already_open"
AUDIT_REASON_EXCEEDS_REFUNDABLE_BALANCE: Final[str] = "exceeds_refundable_balance"
AUDIT_REASON_RETURN_NOT_REPLACEABLE: Final[str] = "return_not_replaceable"
AUDIT_REASON_REPLACEMENT_OUT_OF_STOCK: Final[str] = "replacement_out_of_stock"
AUDIT_REASON_REPLACEMENT_ALREADY_EXISTS: Final[str] = "replacement_already_exists"

_ORDER_NOT_FOUND_MESSAGE: Final[str] = "That order could not be found for this account."

# A return in one of these states is still open work; a second create_return for the same
# item would just fork the paperwork, not fix anything.
_ACTIVE_RETURN_STATUSES: Final[frozenset[ReturnStatus]] = frozenset(
    {ReturnStatus.REQUESTED, ReturnStatus.AUTHORIZED, ReturnStatus.IN_TRANSIT, ReturnStatus.RECEIVED}
)

# create_replacement accepts a return that has not yet been closed out from under the
# customer; RECEIVED/CLOSED/CANCELLED mean the RMA workflow has already moved past "ship a
# replacement".
_REPLACEABLE_RETURN_STATUSES: Final[frozenset[ReturnStatus]] = frozenset(
    {ReturnStatus.REQUESTED, ReturnStatus.AUTHORIZED}
)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

# args: the validated Args model (or None for a tool with no schema); ctx: trusted identity
# and budgets; the CustomerServiceDB view; the environment-derived idempotency key.
ToolHandler = Callable[[Any, EpisodeContext, CustomerServiceDB, str], ToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One registered tool. Structurally satisfies ``tool_protocol.ToolSpecLike``."""

    name: str
    description: str
    args_model: type[BaseModel] | None
    handler: ToolHandler
    reversibility: policies.SeverityRule


class ToolRegistry:
    """Name lookup, argument validation, idempotency-key derivation, and dispatch.

    Holds no episode data. A registry may be shared by arbitrarily many concurrent episodes.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel] | None,
        fn: ToolHandler,
        reversibility: policies.SeverityRule,
    ) -> None:
        if name in self._specs:
            LOGGER.warning("Re-registering tool %r; the previous handler is discarded.", name)
        self._specs[name] = ToolSpec(
            name=name, description=description, args_model=args_model, handler=fn, reversibility=reversibility
        )

    def names(self) -> list[str]:
        """Sorted for a deterministic, reproducible tool catalogue and parser known-set."""
        return sorted(self._specs)

    def specs(self) -> Sequence[ToolSpecLike]:
        # ToolSpec structurally satisfies ToolSpecLike (name/description/args_model); mypy
        # does not infer that through a covariant container in this position, hence the cast.
        return cast("Sequence[ToolSpecLike]", list(self._specs.values()))

    def schemas(self) -> str:
        """The rendered tool catalogue, as ``agents.SupportAssistant`` puts in its system prompt."""
        return render_tool_catalogue(self)

    def severity_for(self, call: ToolCall, db: DatabaseState) -> Severity | None:
        """Reversibility of ``call`` against ``db`` as it stands right now, pre-dispatch.

        Called by the runner before ``execute``, so it works off ``call.raw_arguments`` --
        unvalidated -- rather than a validated model. An unregistered name yields ``None``
        (unstamped event) rather than a guess; ``execute`` will refuse it as ``UNKNOWN_TOOL``.
        """
        spec = self._specs.get(call.name)
        if spec is None:
            return None
        return spec.reversibility(self._view(db), call.raw_arguments)

    def execute(self, call: ToolCall, context: EpisodeContext, db: DatabaseState) -> ToolResult:
        """Validate, derive the idempotency key, dispatch. Never raises."""
        spec = self._specs.get(call.name)
        if spec is None:
            LOGGER.warning("execute() called for unregistered tool %r.", call.name)
            return ToolResult(
                ok=False, code=ToolErrorCode.UNKNOWN_TOOL, message=f"No such tool: {call.name}."
            )

        args: BaseModel | None = None
        if spec.args_model is not None:
            try:
                args = spec.args_model.model_validate(call.raw_arguments)
            except ValidationError as exc:
                LOGGER.info("Argument validation failed for %s: %s", call.name, exc)
                return ToolResult(ok=False, code=ToolErrorCode.INVALID_ARGUMENTS, message=_clip(str(exc)))

        # Handlers run on an isolated working world. This is the transaction boundary:
        # any exception or unsuccessful result leaves the caller's state byte-for-byte intact.
        working = db.model_copy(deep=True)
        view = self._view(working)
        idempotency_key = policies.derive_idempotency_key(call.name, args, context.episode_id)
        try:
            result = spec.handler(args, context, view, idempotency_key)
        except Exception as exc:  # a handler bug is an observation, never a crashed episode
            LOGGER.exception("Tool %r raised %s during dispatch.", call.name, type(exc).__name__)
            return ToolResult(
                ok=False,
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"The {call.name} tool failed unexpectedly.",
                private_audit={"exception": type(exc).__name__},
            )
        if result.ok and state_digest(working) != state_digest(db):
            self._commit_state(db, working)
        return result

    def _view(self, state: DatabaseState) -> CustomerServiceDB:
        """Adapt one working ``DatabaseState`` into a fresh ``CustomerServiceDB`` wrapper.

        The public constructor and ``reset()`` deep-copy; this adapter must mutate the
        transaction's working world. A fresh wrapper is essential: storing it on the registry
        can cross-wire databases when parallel episodes share that registry.
        """
        wrapper = CustomerServiceDB()
        wrapper._state = state  # Intentional adapter for this transaction's working world.
        return wrapper

    @staticmethod
    def _commit_state(target: DatabaseState, source: DatabaseState) -> None:
        """Atomically publish a successful working state through the caller-owned object.

        The runner retains the ``DatabaseState`` object identity, so replace its model fields
        rather than rebinding it. ``source`` is already a deep episode-local copy.
        """

        for field_name in DatabaseState.model_fields:
            setattr(target, field_name, getattr(source, field_name))


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _clip(text: str) -> str:
    if len(text) <= _MAX_VALIDATION_MESSAGE_CHARS:
        return text
    return text[:_MAX_VALIDATION_MESSAGE_CHARS] + "... [truncated]"


def _not_found(message: str, reason: str) -> ToolResult:
    return ToolResult(
        ok=False,
        code=ToolErrorCode.NOT_FOUND_OR_UNAVAILABLE,
        message=message,
        private_audit={"reason": reason},
    )


def _owned_order_or_denial(
    view: CustomerServiceDB, ctx: EpisodeContext, order_id: str
) -> tuple[Order | None, ToolResult | None]:
    """Resolve ``order_id`` iff it belongs to the authenticated caller.

    Nonexistent and not-owned collapse onto the identical public result: only
    ``private_audit``, never rendered to the model, distinguishes the two for the grader.
    """
    if policies.owns_order(view, ctx.authenticated_user_id, order_id):
        return view.order(order_id), None
    reason = AUDIT_REASON_ORDER_NOT_FOUND if view.order(order_id) is None else AUDIT_REASON_OWNERSHIP_DENIED
    return None, _not_found(_ORDER_NOT_FOUND_MESSAGE, reason)


def _policy_reversibility(tool_name: str) -> policies.SeverityRule:
    """Bind ``policies.severity_for`` to one tool name, for storage on its ``ToolSpec``.

    Severity determination stays entirely inside ``policies.py``; the registry only routes.
    """

    def _rule(db: CustomerServiceDB, args: Mapping[str, Any]) -> Severity:
        return policies.severity_for(tool_name, db, args)

    return _rule


def _cancel_order_alternatives(view: CustomerServiceDB, order: Order) -> list[str]:
    if order.status == OrderStatus.DELIVERED:
        return [policies.TOOL_CREATE_RETURN]
    if order.status in policies.PARCEL_RELEASED_ORDER_STATUSES:
        shipments = view.shipments_for_order(order.id)
        if any(policies.intercept_eligible(view, shipment.id) for shipment in shipments):
            return [policies.TOOL_REQUEST_DELIVERY_INTERCEPT]
    return []


def _intercept_alternatives(shipment: Shipment) -> list[str]:
    if shipment.status == ShipmentStatus.DELIVERED:
        return [policies.TOOL_CREATE_RETURN]
    return []


def _active_return_for_item(view: CustomerServiceDB, item_id: str) -> Return | None:
    for record in view.state.returns.values():
        if record.item_id == item_id and record.status in _ACTIVE_RETURN_STATUSES:
            return record
    return None


def _active_notification_for_sku(view: CustomerServiceDB, user_id: str, sku: str) -> Notification | None:
    for notification in view.notifications_for_user(user_id):
        if (
            notification.sku == sku
            and notification.kind == NotificationKind.BACK_IN_STOCK
            and notification.status == NotificationStatus.REGISTERED
        ):
            return notification
    return None


def _in_stock(view: CustomerServiceDB, sku: str) -> bool:
    row = view.inventory(sku)
    return row is not None and row.available_quantity > 0


def _stock_entry(view: CustomerServiceDB, product: Product) -> dict[str, Any]:
    row = view.inventory(product.sku)
    payload = product.model_dump(mode="json")
    payload["available_quantity"] = row.available_quantity if row is not None else 0
    return payload


# --------------------------------------------------------------------------------------
# get_order (read)
# --------------------------------------------------------------------------------------


class GetOrderArgs(ContractModel):
    order_id: str = Field(..., min_length=1, description="Order identifier, e.g. 'ORD-DEMO-81'.")


def _handle_get_order(
    args: GetOrderArgs, _ctx: EpisodeContext, view: CustomerServiceDB, _idem: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, _ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None  # owns_order guarantees this once denial is None
    items = view.items_for_order(order.id)
    shipments = view.shipments_for_order(order.id)
    return ToolResult(
        ok=True,
        code=CODE_ORDER_FOUND,
        message=f"Found order {order.id}.",
        data={
            "order": order.model_dump(mode="json"),
            "items": [item.model_dump(mode="json") for item in items],
            "shipments": [shipment.model_dump(mode="json") for shipment in shipments],
        },
    )


# --------------------------------------------------------------------------------------
# list_order_payments (read)
# --------------------------------------------------------------------------------------


class ListOrderPaymentsArgs(ContractModel):
    order_id: str = Field(..., min_length=1, description="Order identifier, e.g. 'ORD-DEMO-81'.")


def _handle_list_order_payments(
    args: ListOrderPaymentsArgs, ctx: EpisodeContext, view: CustomerServiceDB, _idem: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    payments = view.payments_for_order(order.id)
    return ToolResult(
        ok=True,
        code=CODE_PAYMENTS_LISTED,
        message=f"{len(payments)} payment(s) on order {order.id}.",
        data={"payments": [payment.model_dump(mode="json") for payment in payments]},
    )


# --------------------------------------------------------------------------------------
# search_inventory (read; no ownership check -- the catalogue is not order-scoped)
# --------------------------------------------------------------------------------------


class SearchInventoryArgs(ContractModel):
    product_type: str = Field(..., min_length=1, description="Coarse category, e.g. 'organizer'.")
    size: str = Field(..., min_length=1, description="Size token, e.g. 'ONE'.")
    preferred_color: str | None = Field(
        default=None, description="Customer's preferred color. Omit or send null for any color."
    )


def _handle_search_inventory(
    args: SearchInventoryArgs, _ctx: EpisodeContext, view: CustomerServiceDB, _idem: str
) -> ToolResult:
    in_stock = [p for p in view.products_of_type(args.product_type) if _in_stock(view, p.sku)]
    same_size = [p for p in in_stock if p.size == args.size]
    other_size = [p for p in in_stock if p.size != args.size]
    if args.preferred_color:
        exact = [p for p in same_size if p.color == args.preferred_color]
        similar = [p for p in same_size if p.color != args.preferred_color] + other_size
    else:
        exact = same_size
        similar = other_size
    return ToolResult(
        ok=True,
        code=CODE_INVENTORY_SEARCHED,
        message=f"{len(exact)} exact match(es), {len(similar)} similar match(es).",
        data={
            "exact_matches": [_stock_entry(view, p) for p in exact],
            "similar_matches": [_stock_entry(view, p) for p in similar],
        },
    )


# --------------------------------------------------------------------------------------
# cancel_order (write; keyless idempotency -- no row of its own to replay-detect from)
# --------------------------------------------------------------------------------------


class CancelOrderArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    reason: str = Field(
        ..., min_length=1, max_length=_MAX_REASON_TEXT_CHARS, description="Why the customer wants to cancel."
    )


def _handle_cancel_order(
    args: CancelOrderArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    # CANCELLED is itself the persisted effect. Do not depend on a registry-local ledger:
    # retries after a worker restart must remain idempotent.
    if order.status == OrderStatus.CANCELLED:
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message=f"Order {order.id} was already cancelled.",
            data={"order_id": order.id, "status": order.status.value},
        )
    if not policies.is_cancellable(order):
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=f"Order {order.id} has already left the warehouse and cannot be cancelled directly.",
            data={
                "allowed_next_actions": _cancel_order_alternatives(view, order),
                "order_status": order.status.value,
            },
            private_audit={"reason": AUDIT_REASON_NOT_CANCELLABLE, "order_status": order.status.value},
        )
    view.set_order_status(order.id, OrderStatus.CANCELLED)
    return ToolResult(
        ok=True,
        code=CODE_ORDER_CANCELLED,
        message=f"Order {order.id} has been cancelled.",
        data={"order_id": order.id},
    )


# --------------------------------------------------------------------------------------
# request_delivery_intercept (write; row-based idempotency via DeliveryIntercept)
# --------------------------------------------------------------------------------------


class RequestDeliveryInterceptArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    shipment_id: str = Field(..., min_length=1)


def _handle_request_delivery_intercept(
    args: RequestDeliveryInterceptArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    shipment = view.shipment(args.shipment_id)
    if shipment is None or shipment.order_id != order.id:
        return _not_found("That shipment could not be found on this order.", AUDIT_REASON_SHIPMENT_NOT_FOUND)
    prior = view.find_by_idempotency_key(idem_key)
    if isinstance(prior, DeliveryIntercept):
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message="An intercept for this shipment is already on file.",
            data={"intercept_id": prior.id, "status": prior.status.value},
        )
    if not policies.intercept_eligible(view, shipment.id):
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=f"Shipment {shipment.id} can no longer be intercepted.",
            data={
                "allowed_next_actions": _intercept_alternatives(shipment),
                "shipment_status": shipment.status.value,
            },
            private_audit={
                "reason": AUDIT_REASON_NOT_INTERCEPT_ELIGIBLE,
                "shipment_status": shipment.status.value,
            },
        )
    intercept = view.record_intercept(
        order_id=order.id,
        shipment_id=shipment.id,
        user_id=ctx.authenticated_user_id,
        idempotency_key=idem_key,
    )
    return ToolResult(
        ok=True,
        code=CODE_INTERCEPT_REQUESTED,
        message=f"Requested an intercept for shipment {shipment.id}.",
        data={"intercept_id": intercept.id},
    )


# --------------------------------------------------------------------------------------
# create_return (write; row-based idempotency via Return)
# --------------------------------------------------------------------------------------


class CreateReturnArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    reason: ReturnReason


def _handle_create_return(
    args: CreateReturnArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    item = view.order_item(args.item_id)
    if item is None or item.order_id != order.id:
        return _not_found("That item could not be found on this order.", AUDIT_REASON_ITEM_NOT_FOUND)
    prior = view.find_by_idempotency_key(idem_key)
    if isinstance(prior, Return):
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message="A return is already on file for this item.",
            data={"return_id": prior.id, "status": prior.status.value},
        )
    existing = _active_return_for_item(view, item.id)
    if existing is not None:
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=f"Item {item.id} already has an open return ({existing.id}).",
            data={"return_id": existing.id, "status": existing.status.value},
            private_audit={"reason": AUDIT_REASON_RETURN_ALREADY_OPEN},
        )
    record = view.record_return(
        order_id=order.id,
        user_id=ctx.authenticated_user_id,
        item_id=item.id,
        reason=args.reason,
        idempotency_key=idem_key,
    )
    return ToolResult(
        ok=True,
        code=CODE_RETURN_CREATED,
        message=f"Return {record.id} authorized.",
        data={"return_id": record.id, "status": record.status.value},
    )


# --------------------------------------------------------------------------------------
# issue_refund (write; row-based idempotency via Refund)
# --------------------------------------------------------------------------------------


class IssueRefundArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    payment_id: str = Field(..., min_length=1)
    amount_cents: int = Field(..., gt=0, description="Refund amount in integer cents.")
    reason: RefundReason


def _handle_issue_refund(
    args: IssueRefundArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    payment = view.payment(args.payment_id)
    if payment is None or payment.order_id != order.id:
        return _not_found("That payment could not be found on this order.", AUDIT_REASON_PAYMENT_NOT_FOUND)
    prior = view.find_by_idempotency_key(idem_key)
    if isinstance(prior, Refund):
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message=f"Refund {prior.id} was already issued.",
            data={"refund_id": prior.id, "amount_cents": prior.amount_cents, "status": prior.status.value},
        )
    balance = policies.refundable_balance_cents(view, payment.id)
    if args.amount_cents > balance:
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=(
                f"Payment {payment.id} has only {format_cents(balance)} refundable; "
                f"requested {format_cents(args.amount_cents)}."
            ),
            data={
                "refundable_balance_cents": balance,
                "already_refunded_cents": view.refunded_cents_for_payment(payment.id),
            },
            private_audit={"reason": AUDIT_REASON_EXCEEDS_REFUNDABLE_BALANCE},
        )
    refund = view.record_refund(
        payment_id=payment.id,
        order_id=order.id,
        amount_cents=args.amount_cents,
        reason=args.reason,
        idempotency_key=idem_key,
    )
    return ToolResult(
        ok=True,
        code=CODE_REFUND_ISSUED,
        message=f"Refund {refund.id} for {format_cents(refund.amount_cents)} issued.",
        data={"refund_id": refund.id, "payment_id": payment.id, "amount_cents": refund.amount_cents},
    )


# --------------------------------------------------------------------------------------
# create_replacement (write; keyless idempotency -- OrderItem carries no idempotency_key)
# --------------------------------------------------------------------------------------


class CreateReplacementArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    original_item_id: str = Field(..., min_length=1)
    replacement_sku: str = Field(..., min_length=1)
    return_id: str = Field(..., min_length=1)


def _handle_create_replacement(
    args: CreateReplacementArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    original_item = view.order_item(args.original_item_id)
    if original_item is None or original_item.order_id != order.id:
        return _not_found("That item could not be found on this order.", AUDIT_REASON_ITEM_NOT_FOUND)
    return_record = view.return_record(args.return_id)
    if (
        return_record is None
        or return_record.order_id != order.id
        or return_record.item_id != original_item.id
    ):
        return _not_found("That return could not be found for this item.", AUDIT_REASON_RETURN_NOT_FOUND)
    replacement_item_id = policies.replacement_effect_id(original_item.id)
    prior_replacement = view.order_item(replacement_item_id)
    if prior_replacement is not None:
        if prior_replacement.fulfilled_sku == args.replacement_sku:
            return ToolResult(
                ok=True,
                code=ToolErrorCode.IDEMPOTENT_REPLAY,
                message="This replacement was already processed.",
                data={
                    "return_id": return_record.id,
                    "item_id": prior_replacement.id,
                    "sku": prior_replacement.fulfilled_sku,
                },
            )
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message="A replacement has already been created for this returned item.",
            data={
                "return_id": return_record.id,
                "item_id": prior_replacement.id,
                "sku": prior_replacement.fulfilled_sku,
            },
            private_audit={"reason": AUDIT_REASON_REPLACEMENT_ALREADY_EXISTS},
        )
    if return_record.status not in _REPLACEABLE_RETURN_STATUSES:
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=f"Return {return_record.id} is not in a state that allows a replacement.",
            data={"return_status": return_record.status.value},
            private_audit={"reason": AUDIT_REASON_RETURN_NOT_REPLACEABLE},
        )
    if not policies.replacement_available(view, args.replacement_sku):
        return ToolResult(
            ok=False,
            code=ToolErrorCode.POLICY_DENIED,
            message=f"{args.replacement_sku} is not available to ship as a replacement.",
            data={"allowed_next_actions": [policies.TOOL_CREATE_STOCK_NOTIFICATION]},
            private_audit={"reason": AUDIT_REASON_REPLACEMENT_OUT_OF_STOCK},
        )
    # MIN_REPLACEMENT_QUANTITY doubles as the ship quantity: replacement_available already
    # proved at least this many units are on hand, and this tool only ever ships one unit.
    view.adjust_inventory(args.replacement_sku, -policies.MIN_REPLACEMENT_QUANTITY)
    replacement_product = view.product_by_sku(args.replacement_sku)
    unit_price = replacement_product.unit_price_cents if replacement_product is not None else 0
    item = view.record_order_item(
        order_id=order.id,
        ordered_sku=original_item.ordered_sku,
        fulfilled_sku=args.replacement_sku,
        quantity=policies.MIN_REPLACEMENT_QUANTITY,
        unit_price_cents=unit_price,
        item_id=replacement_item_id,
    )
    return ToolResult(
        ok=True,
        code=CODE_REPLACEMENT_CREATED,
        message=f"Replacement {args.replacement_sku} created as item {item.id}.",
        data={"item_id": item.id, "sku": args.replacement_sku},
    )


# --------------------------------------------------------------------------------------
# create_stock_notification (write; row-based idempotency via Notification, plus a
# per-(user, sku) guard since a fresh order_id would otherwise derive a fresh key)
# --------------------------------------------------------------------------------------


class CreateStockNotificationArgs(ContractModel):
    order_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)


def _handle_create_stock_notification(
    args: CreateStockNotificationArgs, ctx: EpisodeContext, view: CustomerServiceDB, idem_key: str
) -> ToolResult:
    order, denial = _owned_order_or_denial(view, ctx, args.order_id)
    if denial is not None:
        return denial
    assert order is not None
    if view.product_by_sku(args.sku) is None:
        return _not_found("That item could not be found in the catalog.", AUDIT_REASON_SKU_NOT_FOUND)
    prior = view.find_by_idempotency_key(idem_key)
    if isinstance(prior, Notification):
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message=f"You're already registered for {args.sku}.",
            data={"notification_id": prior.id},
        )
    existing = _active_notification_for_sku(view, ctx.authenticated_user_id, args.sku)
    if existing is not None:
        return ToolResult(
            ok=True,
            code=ToolErrorCode.IDEMPOTENT_REPLAY,
            message=f"You're already registered for {args.sku}.",
            data={"notification_id": existing.id},
        )
    notification = view.record_notification(
        user_id=ctx.authenticated_user_id,
        kind=NotificationKind.BACK_IN_STOCK,
        idempotency_key=idem_key,
        order_id=order.id,
        sku=args.sku,
    )
    return ToolResult(
        ok=True,
        code=CODE_NOTIFICATION_REGISTERED,
        message=f"You'll be notified when {args.sku} is back in stock.",
        data={"notification_id": notification.id},
    )


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_default_registry() -> ToolRegistry:
    """Construct the nine-tool registry used by the live demo and the test suite."""
    registry = ToolRegistry()
    registry.register(
        policies.TOOL_GET_ORDER,
        "Look up an order, its line items, and its shipments. Read-only.",
        GetOrderArgs,
        _handle_get_order,
        _policy_reversibility(policies.TOOL_GET_ORDER),
    )
    registry.register(
        policies.TOOL_LIST_ORDER_PAYMENTS,
        "List the payment records on an order, in the order they were charged. Read-only.",
        ListOrderPaymentsArgs,
        _handle_list_order_payments,
        _policy_reversibility(policies.TOOL_LIST_ORDER_PAYMENTS),
    )
    registry.register(
        policies.TOOL_SEARCH_INVENTORY,
        "Search the catalog for in-stock items by type and size, with an optional preferred color. Read-only.",
        SearchInventoryArgs,
        _handle_search_inventory,
        _policy_reversibility(policies.TOOL_SEARCH_INVENTORY),
    )
    registry.register(
        policies.TOOL_CANCEL_ORDER,
        "Cancel an order that has not yet shipped. Refuses once the parcel has left the warehouse.",
        CancelOrderArgs,
        _handle_cancel_order,
        _policy_reversibility(policies.TOOL_CANCEL_ORDER),
    )
    registry.register(
        policies.TOOL_REQUEST_DELIVERY_INTERCEPT,
        "Ask the carrier to divert a shipment before delivery. At most one active intercept per shipment.",
        RequestDeliveryInterceptArgs,
        _handle_request_delivery_intercept,
        _policy_reversibility(policies.TOOL_REQUEST_DELIVERY_INTERCEPT),
    )
    registry.register(
        policies.TOOL_CREATE_RETURN,
        "Open a return authorization for one order item. At most one open return per item.",
        CreateReturnArgs,
        _handle_create_return,
        _policy_reversibility(policies.TOOL_CREATE_RETURN),
    )
    registry.register(
        policies.TOOL_ISSUE_REFUND,
        "Refund a captured payment, up to the amount not already refunded.",
        IssueRefundArgs,
        _handle_issue_refund,
        _policy_reversibility(policies.TOOL_ISSUE_REFUND),
    )
    registry.register(
        policies.TOOL_CREATE_REPLACEMENT,
        "Ship a replacement unit against an authorized return. Requires available stock.",
        CreateReplacementArgs,
        _handle_create_replacement,
        _policy_reversibility(policies.TOOL_CREATE_REPLACEMENT),
    )
    registry.register(
        policies.TOOL_CREATE_STOCK_NOTIFICATION,
        "Register the customer for a back-in-stock alert on one SKU.",
        CreateStockNotificationArgs,
        _handle_create_stock_notification,
        _policy_reversibility(policies.TOOL_CREATE_STOCK_NOTIFICATION),
    )
    return registry


__all__ = [
    "AUDIT_REASON_EXCEEDS_REFUNDABLE_BALANCE",
    "AUDIT_REASON_ITEM_NOT_FOUND",
    "AUDIT_REASON_NOT_CANCELLABLE",
    "AUDIT_REASON_NOT_INTERCEPT_ELIGIBLE",
    "AUDIT_REASON_ORDER_NOT_FOUND",
    "AUDIT_REASON_OWNERSHIP_DENIED",
    "AUDIT_REASON_PAYMENT_NOT_FOUND",
    "AUDIT_REASON_REPLACEMENT_ALREADY_EXISTS",
    "AUDIT_REASON_REPLACEMENT_OUT_OF_STOCK",
    "AUDIT_REASON_RETURN_ALREADY_OPEN",
    "AUDIT_REASON_RETURN_NOT_FOUND",
    "AUDIT_REASON_RETURN_NOT_REPLACEABLE",
    "AUDIT_REASON_SHIPMENT_NOT_FOUND",
    "AUDIT_REASON_SKU_NOT_FOUND",
    "CODE_INTERCEPT_REQUESTED",
    "CODE_INVENTORY_SEARCHED",
    "CODE_NOTIFICATION_REGISTERED",
    "CODE_ORDER_CANCELLED",
    "CODE_ORDER_FOUND",
    "CODE_PAYMENTS_LISTED",
    "CODE_REFUND_ISSUED",
    "CODE_REPLACEMENT_CREATED",
    "CODE_RETURN_CREATED",
    "CancelOrderArgs",
    "CreateReplacementArgs",
    "CreateReturnArgs",
    "CreateStockNotificationArgs",
    "GetOrderArgs",
    "IssueRefundArgs",
    "ListOrderPaymentsArgs",
    "RequestDeliveryInterceptArgs",
    "SearchInventoryArgs",
    "ToolHandler",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
]
