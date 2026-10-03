"""Typed contracts for identity, world state, trajectories, tasks and reward vectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------


class ContractModel(BaseModel):
    """Base for every model in this file: unknown fields are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Episode identity and budgets
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """Trusted per-episode runtime state, supplied by the orchestrator.

    The assistant model never produces any of these values. ``ToolRegistry.execute``
    receives this object alongside the model-supplied arguments; ownership checks read
    ``authenticated_user_id`` from here and nowhere else.
    """

    episode_id: str
    task_id: str
    authenticated_user_id: str
    max_dialogue_turns: int
    max_tool_rounds_per_turn: int


# --------------------------------------------------------------------------------------
# Action severity (reversibility taxonomy)
# --------------------------------------------------------------------------------------


class Severity(StrEnum):
    """Cost of the inverse of an action, computed at call time from world state.

    Never a static per-tool constant: ``cancel_order`` on a ``pending`` order is
    REVERSIBLE and on a ``shipped`` order is IRREVERSIBLE.
    """

    FREE = "free"  # pure read, nothing emitted
    REVERSIBLE = "reversible"  # inverse exists, we absorb it, nothing escaped
    COSTLY = "costly"  # inverse exists but an external party absorbs it
    IRREVERSIBLE = "irreversible"  # no inverse, or money/parcel/information escaped


# Ordered low -> high. Confirmation downgrades one step, concealment upgrades one step;
# both operations index this ladder so the two callers cannot disagree on the ordering.
SEVERITY_LADDER: tuple[Severity, ...] = (
    Severity.FREE,
    Severity.REVERSIBLE,
    Severity.COSTLY,
    Severity.IRREVERSIBLE,
)


# --------------------------------------------------------------------------------------
# Tool call / result envelope
# --------------------------------------------------------------------------------------


class ToolCall(ContractModel):
    """One normalized tool call recovered from raw assistant text by the parser."""

    id: str
    name: str
    raw_arguments: dict[str, Any]
    source_span: str  # verbatim text the call was parsed from; kept for trace forensics


class ToolErrorCode(StrEnum):
    """Registry- and policy-level failure codes. Domain success codes live in the tools module."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_UNAVAILABLE = "tool_unavailable"
    POLICY_DENIED = "policy_denied"
    # Deliberately conflates "does not exist" with "not yours" so the public result
    # cannot be used to probe another customer's record set.
    NOT_FOUND_OR_UNAVAILABLE = "resource_not_found_or_unavailable"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    INTERNAL_ERROR = "internal_error"


class ToolResult(ContractModel):
    """Structured envelope returned by every tool. Handlers return this; they never raise.

    A denied operation returns ``ok=False`` and MUST leave world state byte-identical.
    """

    ok: bool
    code: str  # stable machine-readable; ToolErrorCode for failures, tool-owned on success
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    # Observer/reward-only detail. Never rendered into the assistant's tool observation:
    # The grader needs to distinguish ownership denial while the assistant
    # only learns "not found or unavailable".
    private_audit: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Text tool-call parser
# --------------------------------------------------------------------------------------


class ParseErrorCode(StrEnum):
    """Canonical vocabulary for parser failures returned to the model as observations."""

    MALFORMED_JSON = "malformed_json"
    UNCLOSED_TAG = "unclosed_tag"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    MISSING_TOOL_NAME = "missing_tool_name"
    ARGUMENTS_NOT_OBJECT = "arguments_not_object"
    EMPTY_TOOL_CALL = "empty_tool_call"


class ToolParseError(ContractModel):
    """A recoverable parse failure. The parser reports these; it never raises."""

    code: str  # use ParseErrorCode
    message: str
    raw_span: str


class ParseOutcome(ContractModel):
    """Result of splitting one raw assistant message into user-visible prose and calls.

    ``visible_text`` is the only field that may ever reach the user simulator's history.
    """

    visible_text: str
    calls: list[ToolCall] = Field(default_factory=list)
    errors: list[ToolParseError] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------

Actor = Literal["user", "assistant", "tool", "environment"]

EventKind = Literal[
    "message",
    "tool_call",
    "tool_result",
    "invalid_tool_call",
    "policy_denial",
    "termination",
]


class TerminationReason(StrEnum):
    """Closed set of episode endings. Whether an ending was *justified* is the grader's call."""

    USER_DONE = "user_done"
    MAX_DIALOGUE_TURNS = "max_dialogue_turns"
    MAX_TOOL_ROUNDS = "max_tool_rounds"
    REPEATED_NO_PROGRESS = "repeated_no_progress"
    INVALID_USER_OUTPUT = "invalid_user_output"
    PROVIDER_ERROR = "provider_error"


class TrajectoryEvent(ContractModel):
    """One ordered, typed record of what happened. The grader reads these, not raw API messages."""

    sequence: int
    actor: Actor
    kind: EventKind
    visible_to_user: bool
    content: dict[str, Any] = Field(default_factory=dict)
    # Stamped by the orchestrator from world state at the moment the call was taken,
    # so the reward engine reads severity classes without importing a tool.
    severity: Severity | None = None


class Trajectory(ContractModel):
    """The complete ordered record of an episode.

    This object is the sole interface between the simulation and the reward engine;
    nothing in the reward package imports a tool, a policy, or the registry.
    """

    episode_id: str
    task_id: str
    events: list[TrajectoryEvent] = Field(default_factory=list)
    termination_reason: TerminationReason | None = None


# --------------------------------------------------------------------------------------
# World: status vocabularies
# --------------------------------------------------------------------------------------


class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    PARTIALLY_SHIPPED = "partially_shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class PaymentStatus(StrEnum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    PARTIALLY_REFUNDED = "partially_refunded"
    REFUNDED = "refunded"
    FAILED = "failed"
    VOIDED = "voided"


class ShipmentStatus(StrEnum):
    LABEL_CREATED = "label_created"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    INTERCEPTED = "intercepted"
    RETURNED_TO_SENDER = "returned_to_sender"
    LOST = "lost"


class RefundStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class RefundReason(StrEnum):
    DUPLICATE_CHARGE = "duplicate_charge"
    WRONG_ITEM = "wrong_item"
    ITEM_NOT_RECEIVED = "item_not_received"
    DAMAGED = "damaged"
    ORDER_CANCELLED = "order_cancelled"
    CUSTOMER_REQUEST = "customer_request"
    PRICE_ADJUSTMENT = "price_adjustment"
    OTHER = "other"


class ReturnStatus(StrEnum):
    REQUESTED = "requested"
    AUTHORIZED = "authorized"
    IN_TRANSIT = "in_transit"
    RECEIVED = "received"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ReturnReason(StrEnum):
    WRONG_ITEM = "wrong_item"
    DAMAGED = "damaged"
    NOT_AS_DESCRIBED = "not_as_described"
    NO_LONGER_NEEDED = "no_longer_needed"
    ARRIVED_LATE = "arrived_late"
    OTHER = "other"


class InterceptStatus(StrEnum):
    REQUESTED = "requested"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class NotificationKind(StrEnum):
    BACK_IN_STOCK = "back_in_stock"
    ORDER_UPDATE = "order_update"
    REFUND_CONFIRMATION = "refund_confirmation"


class NotificationStatus(StrEnum):
    REGISTERED = "registered"
    SENT = "sent"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------------------
# World: row models
# --------------------------------------------------------------------------------------


class User(ContractModel):
    id: str
    name: str
    email: str
    shipping_address: str = ""


class Product(ContractModel):
    id: str
    sku: str
    name: str
    product_type: str = ""  # coarse category, e.g. "jacket"; drives substitute search
    color: str = ""
    size: str = ""
    unit_price_cents: int = 0


class InventoryRow(ContractModel):
    sku: str
    available_quantity: int = 0


class OrderItem(ContractModel):
    id: str
    order_id: str
    ordered_sku: str
    # Differs from ordered_sku exactly when the warehouse shipped the wrong variant.
    fulfilled_sku: str
    quantity: int = 1
    unit_price_cents: int = 0


class Order(ContractModel):
    id: str
    user_id: str
    status: OrderStatus
    item_ids: list[str] = Field(default_factory=list)
    payment_ids: list[str] = Field(default_factory=list)
    shipment_ids: list[str] = Field(default_factory=list)
    total_cents: int = 0
    placed_at: str = ""  # frozen ISO-8601 string; no wall-clock is read inside an episode


class Payment(ContractModel):
    id: str
    order_id: str
    user_id: str
    amount_cents: int
    status: PaymentStatus
    processor_reference: str = ""  # distinguishes a genuine duplicate capture from a retry


class Shipment(ContractModel):
    id: str
    order_id: str
    status: ShipmentStatus
    carrier: str = ""
    tracking_number: str = ""
    item_ids: list[str] = Field(default_factory=list)


class Return(ContractModel):
    id: str
    order_id: str
    user_id: str
    item_id: str
    reason: ReturnReason
    status: ReturnStatus
    idempotency_key: str = ""


class Refund(ContractModel):
    id: str
    payment_id: str
    order_id: str
    amount_cents: int
    reason: RefundReason
    status: RefundStatus
    idempotency_key: str = ""


class DeliveryIntercept(ContractModel):
    id: str
    order_id: str
    shipment_id: str
    user_id: str
    status: InterceptStatus
    idempotency_key: str = ""


class Notification(ContractModel):
    id: str
    user_id: str
    kind: NotificationKind
    status: NotificationStatus
    order_id: str = ""
    sku: str = ""
    idempotency_key: str = ""


# --------------------------------------------------------------------------------------
# World: aggregate state
# --------------------------------------------------------------------------------------


class DatabaseState(ContractModel):
    """The whole world, as dictionaries keyed by stable id (``inventory`` is keyed by sku).

    Episode lifecycle is deep-copy in, snapshot, mutate, snapshot, discard. Use
    ``model_copy(deep=True)`` for the copies; the tool layer mutates rows in place.
    """

    users: dict[str, User] = Field(default_factory=dict)
    orders: dict[str, Order] = Field(default_factory=dict)
    order_items: dict[str, OrderItem] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict)
    inventory: dict[str, InventoryRow] = Field(default_factory=dict)
    payments: dict[str, Payment] = Field(default_factory=dict)
    shipments: dict[str, Shipment] = Field(default_factory=dict)
    returns: dict[str, Return] = Field(default_factory=dict)
    refunds: dict[str, Refund] = Field(default_factory=dict)
    delivery_intercepts: dict[str, DeliveryIntercept] = Field(default_factory=dict)
    notifications: dict[str, Notification] = Field(default_factory=dict)

    def refunded_cents_for_payment(self, payment_id: str) -> int:
        """Cents already committed against a payment.

        Lives on the contract because the eligibility check (tools) and the over-refund
        verifier (reward) must never disagree about this number. PENDING counts as
        committed: the processor reference is already issued, so the money has left.
        """
        return sum(
            refund.amount_cents
            for refund in self.refunds.values()
            if refund.payment_id == payment_id
            and refund.status in (RefundStatus.COMPLETED, RefundStatus.PENDING)
        )


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------

# Predicates are evaluated from a named registry. Task data never carries Python.
PredicateParam = str | int | bool | None | list[str]


class PredicateSpec(ContractModel):
    """A named, registry-resolved assertion over world state. No eval, ever."""

    name: str
    params: dict[str, PredicateParam] = Field(default_factory=dict)


class OutcomeBranch(ContractModel):
    """One legitimate ending. Branches are plural because customer service is disjunctive.

    Scoring takes the best *fully consistent* branch, never a union of partial credit
    drawn from mutually incompatible branches.
    """

    id: str
    description: str
    credit: Literal["full", "partial", "escalation"] = "full"
    required_state: list[PredicateSpec] = Field(default_factory=list)
    preserved_state: list[PredicateSpec] = Field(default_factory=list)
    required_events: list[str] = Field(default_factory=list)  # ordered event/tool names
    required_communication_facts: list[str] = Field(default_factory=list)


# Budget defaults: eight dialogue turns is roughly twice the oracle length for the longest
# canonical task, and six tool rounds covers read-read-write plus two recovery attempts.
DEFAULT_MAX_DIALOGUE_TURNS = 8
DEFAULT_MAX_TOOL_ROUNDS_PER_TURN = 6


class TaskSpec(ContractModel):
    """A task, its world, and its verifier as one object.

    ``hidden_user_goal`` is rendered into the user simulator's system prompt only. It is
    never shown to the support assistant, and nothing the task expects the assistant to
    discover may appear in ``assistant_policy``.
    """

    task_id: str
    title: str
    authenticated_user_id: str
    hidden_user_goal: str
    assistant_policy: str
    initial_db: DatabaseState
    success_branches: list[OutcomeBranch] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    forbidden_events: list[str] = Field(default_factory=list)
    communication_requirements: list[str] = Field(default_factory=list)
    max_dialogue_turns: int = DEFAULT_MAX_DIALOGUE_TURNS
    max_tool_rounds_per_turn: int = DEFAULT_MAX_TOOL_ROUNDS_PER_TURN
    tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# User simulator output
# --------------------------------------------------------------------------------------


class UserTurn(ContractModel):
    """Structured user-simulator response. Only ``message`` crosses to the assistant."""

    message: str
    done: bool = False
    reason: str = ""  # harness metadata; never rendered into any model prompt


# --------------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------------


class RewardBreakdown(ContractModel):
    """Scalar for ranking, vector for debugging.

    ``total`` is authoritative and already includes penalties and fatal gating; callers
    must not recompute it from the components.
    """

    outcome: float = 0.0
    process: float = 0.0
    communication: float = 0.0
    efficiency: float = 0.0
    penalties: dict[str, float] = Field(default_factory=dict)
    fatal_violations: list[str] = Field(default_factory=list)
    total: float = 0.0
    explanation: list[str] = Field(default_factory=list)


__all__ = [
    "Actor",
    "ContractModel",
    "DEFAULT_MAX_DIALOGUE_TURNS",
    "DEFAULT_MAX_TOOL_ROUNDS_PER_TURN",
    "DatabaseState",
    "DeliveryIntercept",
    "EpisodeContext",
    "EventKind",
    "InterceptStatus",
    "InventoryRow",
    "Notification",
    "NotificationKind",
    "NotificationStatus",
    "Order",
    "OrderItem",
    "OrderStatus",
    "OutcomeBranch",
    "ParseErrorCode",
    "ParseOutcome",
    "Payment",
    "PaymentStatus",
    "PredicateParam",
    "PredicateSpec",
    "Product",
    "Refund",
    "RefundReason",
    "RefundStatus",
    "Return",
    "ReturnReason",
    "ReturnStatus",
    "RewardBreakdown",
    "SEVERITY_LADDER",
    "Severity",
    "Shipment",
    "ShipmentStatus",
    "TaskSpec",
    "TerminationReason",
    "ToolCall",
    "ToolErrorCode",
    "ToolParseError",
    "ToolResult",
    "Trajectory",
    "TrajectoryEvent",
    "User",
    "UserTurn",
]
