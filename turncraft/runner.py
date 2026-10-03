"""Bounded episode orchestration, text-tool dispatch, visibility separation and retry ownership."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, TypeVar

from turncraft.config import Settings, get_settings
from turncraft.db import state_digest
from turncraft.models import (
    Actor,
    DatabaseState,
    EpisodeContext,
    EventKind,
    ParseOutcome,
    Severity,
    TaskSpec,
    TerminationReason,
    ToolCall,
    ToolErrorCode,
    ToolParseError,
    ToolResult,
    Trajectory,
    TrajectoryEvent,
    UserTurn,
)

LOGGER = logging.getLogger(__name__)

# --- Wire format for observations fed back to the assistant ----------------------------
# The parser owns assistant -> environment (`<tool_call>`); the runner owns the return leg.
# Distinct tag names so a result can never be mistaken for a call by any parser.

TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"
TOOL_ERROR_OPEN = "<tool_error>"
TOOL_ERROR_CLOSE = "</tool_error>"

# Blocks within one observation message, and merged same-role history entries.
OBSERVATION_SEPARATOR = "\n"
HISTORY_JOIN_SEPARATOR = "\n\n"

# --- Message roles ---------------------------------------------------------------------
# assistant_history is an Anthropic Messages payload: user/assistant only, first entry
# must be `user`. The outer loop always appends the customer message first, so it is.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# user_history is an audit artefact, not an API payload. Neutral speaker labels make the
# context-isolation assertion unambiguous and make the list un-sendable by accident.
SPEAKER_CUSTOMER = "customer"
SPEAKER_SUPPORT = "support"

# --- Retry ------------------------------------------------------------------------------
# The runner is the single retry owner for logical assistant/user calls. Provider clients
# must make one transport attempt; otherwise their retry budget multiplies with this one.
# Linear backoff keeps the worst-case episode delay predictable.
RETRY_BACKOFF_SECONDS = 0.5

# SimulatedUser uses this reason after exhausting content-repair attempts. It remains a
# UserTurn so the public simulator protocol does not change, but it is not a real customer
# completion and therefore must never be classified as USER_DONE.
INVALID_USER_OUTPUT_REASON = "user_sim_invalid_output"

_T = TypeVar("_T")


# --------------------------------------------------------------------------------------
# Ports — the seams the orchestrator is written against
# --------------------------------------------------------------------------------------


class ToolParserProtocol(Protocol):
    """Splits one raw assistant message into visible prose and normalized calls.

    Satisfied by a module (``tool_protocol``) as well as by an object. Never raises:
    malformed input comes back as :class:`ToolParseError` entries.
    """

    def parse(self, text: str, known_tool_names: Sequence[str]) -> ParseOutcome: ...


class ToolRegistryProtocol(Protocol):
    """Name lookup, argument validation, policy enforcement, and dispatch.

    ``execute`` is the validation boundary: the runner does not know the argument models,
    so a schema failure is a ``ToolResult(ok=False, code=INVALID_ARGUMENTS)``, not an
    exception. Implementations may also expose
    ``severity_for(call, db) -> Severity`` to stamp reversibility from pre-call state;
    see :func:`_registry_severity`.
    """

    def names(self) -> list[str]: ...

    def execute(self, call: ToolCall, context: EpisodeContext, db: DatabaseState) -> ToolResult: ...


class AssistantProtocol(Protocol):
    """One support-assistant completion. Returns raw text; the runner does the parsing."""

    def one_turn(self, history: list[dict[str, str]]) -> str: ...


class UserSimulatorProtocol(Protocol):
    """The customer. Sees visible assistant text only; keeps its own history."""

    def start(self) -> UserTurn: ...

    def respond(self, assistant_text: str) -> UserTurn: ...


# Severity is domain knowledge the runner does not own; it asks and records the answer.
SeverityResolver = Callable[[ToolCall, DatabaseState], Severity | None]


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Everything the grader and the demo need, and nothing the agent produced raw.

    ``initial_db`` and ``final_db`` are independent deep copies; ``task.initial_db`` is
    never mutated, so episodes cannot contaminate one another.
    """

    trajectory: Trajectory
    initial_db: DatabaseState
    final_db: DatabaseState
    context: EpisodeContext
    user_history: list[dict[str, str]] = field(default_factory=list)
    assistant_history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _AssistantTurn:
    """Outcome of one inner tool loop."""

    visible_text: str
    termination: TerminationReason | None = None
    detail: str = ""


# --------------------------------------------------------------------------------------
# Observation encoding (runner-owned wire format)
# --------------------------------------------------------------------------------------


def encode_tool_observation(call: ToolCall, result: ToolResult) -> str:
    """Render one tool result as the assistant's observation.

    The payload is an explicit key whitelist, so ``ToolResult.private_audit`` — which the
    grader reads to distinguish an ownership denial from a genuine miss — cannot reach the
    model by anyone forgetting to strip it.
    """
    payload = {
        "id": call.id,
        "name": call.name,
        "ok": result.ok,
        "code": result.code,
        "message": result.message,
        "data": result.data,
    }
    return f"{TOOL_RESULT_OPEN}{json.dumps(payload, sort_keys=True, default=str)}{TOOL_RESULT_CLOSE}"


def encode_parse_error(error: ToolParseError) -> str:
    """Render one parse failure as an observation the model can correct from.

    ``raw_span`` is deliberately omitted: echoing a malformed ``<tool_call>`` fragment back
    into the history re-introduces the exact text a parser must not re-read.
    """
    payload = {"code": error.code, "message": error.message}
    return f"{TOOL_ERROR_OPEN}{json.dumps(payload, sort_keys=True)}{TOOL_ERROR_CLOSE}"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _call_fingerprint(call: ToolCall, result: ToolResult) -> str:
    """Identity of a (call, outcome) pair: same action, same answer."""
    payload = json.dumps(
        {"name": call.name, "arguments": call.raw_arguments, "ok": result.ok, "code": result.code},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _registry_severity(registry: ToolRegistryProtocol, call: ToolCall, db: DatabaseState) -> Severity | None:
    """Ask the registry for reversibility at call time.

    The runner has no way to compute severity itself, so a registry that declines yields an
    unstamped event rather than a fabricated one. Logged once per failure, never guessed.
    """
    resolver = getattr(registry, "severity_for", None)
    if resolver is None:
        LOGGER.debug("Registry exposes no severity_for(); tool events will carry no severity.")
        return None
    try:
        severity = resolver(call, db)
    except Exception as exc:  # a broken severity hook must not abort a valid tool call
        LOGGER.warning(
            "severity_for(%s) raised %s: %s; recording no severity.", call.name, type(exc).__name__, exc
        )
        return None
    if severity is None:
        return None
    if not isinstance(severity, Severity):
        LOGGER.warning("severity_for(%s) returned %r, not a Severity; recording none.", call.name, severity)
        return None
    return severity


def _append_message(history: list[dict[str, str]], role: str, content: str) -> None:
    """Append to a history, skipping empties and merging consecutive same-role entries.

    Empty content blocks are rejected by the Messages API, and merging keeps the payload
    strictly alternating regardless of how many observation blocks a round produced.
    """
    if not content.strip():
        return
    if history and history[-1]["role"] == role:
        history[-1]["content"] = f"{history[-1]['content']}{HISTORY_JOIN_SEPARATOR}{content}"
        return
    history.append({"role": role, "content": content})


class _TrajectoryWriter:
    """Owns the sequence counter so no caller can emit an out-of-order event."""

    def __init__(self, episode_id: str, task_id: str) -> None:
        self.trajectory = Trajectory(episode_id=episode_id, task_id=task_id)
        self._sequence = 0

    def emit(
        self,
        *,
        actor: Actor,
        kind: EventKind,
        visible_to_user: bool,
        content: dict[str, Any],
        severity: Severity | None = None,
    ) -> TrajectoryEvent:
        event = TrajectoryEvent(
            sequence=self._sequence,
            actor=actor,
            kind=kind,
            visible_to_user=visible_to_user,
            content=content,
            severity=severity,
        )
        self._sequence += 1
        self.trajectory.events.append(event)
        return event


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


class EpisodeRunner:
    """Drives one episode. Construct, call :meth:`run` once, read the result."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        assistant: AssistantProtocol,
        user: UserSimulatorProtocol,
        registry: ToolRegistryProtocol,
        parser: ToolParserProtocol,
        episode_id: str | None = None,
        settings: Settings | None = None,
        severity_resolver: SeverityResolver | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._task = task
        self._assistant = assistant
        self._user = user
        self._registry = registry
        self._parser = parser
        self._settings = settings if settings is not None else get_settings()
        self._severity_resolver = severity_resolver
        self._sleep = sleep

        # Deterministic default: no wall-clock inside an episode, and a rerun of the same
        # task produces the same id. Callers wanting uniqueness pass their own.
        self._context = EpisodeContext(
            episode_id=episode_id if episode_id else f"ep-{task.task_id}",
            task_id=task.task_id,
            authenticated_user_id=task.authenticated_user_id,
            max_dialogue_turns=task.max_dialogue_turns,
            max_tool_rounds_per_turn=task.max_tool_rounds_per_turn,
        )

        self._db = task.initial_db.model_copy(deep=True)
        self._initial_db = task.initial_db.model_copy(deep=True)

        self._writer = _TrajectoryWriter(self._context.episode_id, task.task_id)
        self._user_history: list[dict[str, str]] = []
        self._assistant_history: list[dict[str, str]] = []

        # No-progress bookkeeping. Cleared whenever new evidence arrives — a new customer
        # message, or a world-state change.
        self._repeat_counts: dict[str, int] = {}
        self._last_visible_text: str | None = None
        self._digest_at_last_visible_text: str | None = None
        self._tool_names = list(registry.names())

    # -- public ------------------------------------------------------------------------

    def run(self) -> EpisodeResult:
        termination, detail = self._run_conversation()
        self._writer.trajectory.termination_reason = termination
        self._writer.emit(
            actor="environment",
            kind="termination",
            visible_to_user=False,
            content={"reason": termination.value, "detail": detail},
        )
        return EpisodeResult(
            trajectory=self._writer.trajectory,
            initial_db=self._initial_db,
            final_db=self._db.model_copy(deep=True),
            context=self._context,
            user_history=self._user_history,
            assistant_history=self._assistant_history,
        )

    # -- outer loop --------------------------------------------------------------------

    def _run_conversation(self) -> tuple[TerminationReason, str]:
        opening, failure = self._call_with_retry("SimulatedUser.start", self._user.start)
        if opening is None:
            return TerminationReason.PROVIDER_ERROR, failure
        if not isinstance(opening, UserTurn):
            LOGGER.error("SimulatedUser.start returned %r, not a UserTurn.", type(opening).__name__)
            return TerminationReason.INVALID_USER_OUTPUT, f"start returned {type(opening).__name__}"

        user_turn = opening
        if user_turn.done and user_turn.reason == INVALID_USER_OUTPUT_REASON:
            self._record_customer_message(user_turn.message)
            return TerminationReason.INVALID_USER_OUTPUT, user_turn.reason
        if user_turn.done:
            # Degenerate but legal: the customer opens and closes in one breath.
            self._record_customer_message(user_turn.message)
            return TerminationReason.USER_DONE, user_turn.reason

        for _ in range(self._context.max_dialogue_turns):
            self._record_customer_message(user_turn.message)

            turn = self._run_tool_loop()

            if turn.visible_text.strip():
                self._writer.emit(
                    actor="assistant",
                    kind="message",
                    visible_to_user=True,
                    content={"text": turn.visible_text},
                )
                self._user_history.append({"role": SPEAKER_SUPPORT, "content": turn.visible_text})

            if turn.termination is not None:
                return turn.termination, turn.detail

            if self._is_repeated_visible_text(turn.visible_text):
                return (
                    TerminationReason.REPEATED_NO_PROGRESS,
                    "assistant repeated a reply with no state change",
                )

            nxt, failure = self._call_with_retry(
                "SimulatedUser.respond", partial(self._user.respond, turn.visible_text)
            )
            if nxt is None:
                return TerminationReason.PROVIDER_ERROR, failure
            if not isinstance(nxt, UserTurn):
                LOGGER.error("SimulatedUser.respond returned %r, not a UserTurn.", type(nxt).__name__)
                return TerminationReason.INVALID_USER_OUTPUT, f"respond returned {type(nxt).__name__}"

            user_turn = nxt
            if user_turn.done and user_turn.reason == INVALID_USER_OUTPUT_REASON:
                self._record_customer_message(user_turn.message)
                return TerminationReason.INVALID_USER_OUTPUT, user_turn.reason
            if user_turn.done:
                # Record the closing line; it is real dialogue, it just gets no reply.
                self._record_customer_message(user_turn.message)
                return TerminationReason.USER_DONE, user_turn.reason

        return (
            TerminationReason.MAX_DIALOGUE_TURNS,
            f"budget of {self._context.max_dialogue_turns} turns exhausted",
        )

    # -- inner loop --------------------------------------------------------------------

    def _run_tool_loop(self) -> _AssistantTurn:
        for _ in range(self._context.max_tool_rounds_per_turn):
            raw, failure = self._call_with_retry(
                "SupportAssistant.one_turn", lambda: self._assistant.one_turn(self._assistant_history)
            )
            if raw is None:
                return _AssistantTurn("", TerminationReason.PROVIDER_ERROR, failure)

            text = raw if isinstance(raw, str) else str(raw)
            _append_message(self._assistant_history, ROLE_ASSISTANT, text)

            outcome = self._parser.parse(text, self._tool_names)

            observations: list[str] = []
            for error in outcome.errors:
                self._writer.emit(
                    actor="assistant",
                    kind="invalid_tool_call",
                    visible_to_user=False,
                    content={"code": error.code, "message": error.message, "raw_span": error.raw_span},
                )
                observations.append(encode_parse_error(error))

            if not outcome.calls:
                if observations:
                    # Nothing executable this round: hand the errors back and let the model
                    # correct itself within the same tool budget.
                    _append_message(
                        self._assistant_history, ROLE_USER, OBSERVATION_SEPARATOR.join(observations)
                    )
                    continue
                return _AssistantTurn(outcome.visible_text)

            # The turn acted, so its prose is an explanation of privileged work: recorded,
            # never forwarded to the customer.
            if outcome.visible_text.strip():
                self._writer.emit(
                    actor="assistant",
                    kind="message",
                    visible_to_user=False,
                    content={"text": outcome.visible_text},
                )

            # Evaluate no-progress over the complete tool round. A duplicate call may sit
            # before or after a new observation or a valid write in the same assistant
            # message; progress anywhere in the round clears the provisional stall.
            digest_before_round = state_digest(self._db)
            saw_stall = False
            saw_new_evidence = False
            round_fingerprints: set[str] = set()
            for call in outcome.calls:
                observation, stalled, new_evidence, fingerprint = self._execute_call(call)
                observations.append(observation)
                saw_stall = saw_stall or stalled
                saw_new_evidence = saw_new_evidence or new_evidence
                round_fingerprints.add(fingerprint)

            world_changed = state_digest(self._db) != digest_before_round
            # Preserve the original all-duplicate guard: A,A in an otherwise empty round
            # is a stall even though the first A established the baseline. A genuinely
            # different new observation (A,B or B,A) is progress for the whole round.
            only_one_observation = len(round_fingerprints) == 1
            no_progress = (
                saw_stall and not world_changed and not (saw_new_evidence and not only_one_observation)
            )

            _append_message(self._assistant_history, ROLE_USER, OBSERVATION_SEPARATOR.join(observations))

            if no_progress:
                return _AssistantTurn(
                    "",
                    TerminationReason.REPEATED_NO_PROGRESS,
                    "identical tool call produced an identical result with no state change",
                )

        # Budget gone mid-tool-loop: everything in this turn was privileged, so nothing is
        # customer-visible and the episode ends here rather than replying with silence.
        return _AssistantTurn(
            "",
            TerminationReason.MAX_TOOL_ROUNDS,
            f"budget of {self._context.max_tool_rounds_per_turn} tool rounds exhausted",
        )

    # -- dispatch ----------------------------------------------------------------------

    def _execute_call(self, call: ToolCall) -> tuple[str, bool, bool, str]:
        """Execute one call and return its observation plus progress metadata."""
        digest_before = state_digest(self._db)
        severity = (
            self._severity_resolver(call, self._db)
            if self._severity_resolver is not None
            else _registry_severity(self._registry, call, self._db)
        )

        self._writer.emit(
            actor="assistant",
            kind="tool_call",
            visible_to_user=False,
            content={"call_id": call.id, "name": call.name, "arguments": call.raw_arguments},
            severity=severity,
        )

        try:
            result = self._registry.execute(call, self._context, self._db)
        except Exception as exc:  # a handler bug is an observation, never a crashed episode
            LOGGER.exception(
                "Tool %r raised %s during execute; recording internal_error.", call.name, type(exc).__name__
            )
            result = ToolResult(
                ok=False,
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"The {call.name} tool failed unexpectedly.",
                private_audit={"exception": type(exc).__name__},
            )

        self._writer.emit(
            actor="tool",
            kind="tool_result",
            visible_to_user=False,
            content={
                "call_id": call.id,
                "name": call.name,
                "ok": result.ok,
                "code": result.code,
                "message": result.message,
                "data": result.data,
                "private_audit": result.private_audit,
            },
            severity=severity,
        )

        # An attempted-but-refused action is evidence the grader needs on its own terms:
        # the tool-layer denial prevented the harm, the attempt still happened.
        if not result.ok and result.code == ToolErrorCode.POLICY_DENIED:
            self._writer.emit(
                actor="environment",
                kind="policy_denial",
                visible_to_user=False,
                content={
                    "call_id": call.id,
                    "name": call.name,
                    "code": result.code,
                    "private_audit": result.private_audit,
                },
                severity=severity,
            )

        digest_after = state_digest(self._db)
        stalled, new_evidence, fingerprint = self._note_progress(call, result, digest_before, digest_after)
        return encode_tool_observation(call, result), stalled, new_evidence, fingerprint

    # -- no-progress detection ---------------------------------------------------------

    def _note_progress(
        self, call: ToolCall, result: ToolResult, digest_before: str, digest_after: str
    ) -> tuple[bool, bool, str]:
        """Return ``(stalled, new_evidence, fingerprint)`` for one call.

        Deliberately wider than "the same error twice": a successful read repeated with an
        identical result is also no new evidence, and an unbounded read loop is the failure
        mode most likely to eat a live demo.
        """
        fingerprint = _call_fingerprint(call, result)
        new_evidence = fingerprint not in self._repeat_counts
        if digest_after != digest_before:
            self._repeat_counts.clear()
            return False, True, fingerprint
        count = self._repeat_counts.get(fingerprint, 0) + 1
        self._repeat_counts[fingerprint] = count
        if count >= self._settings.no_progress_repeat_limit:
            LOGGER.info(
                "No progress: %s repeated %d times with code %r and no state change.",
                call.name,
                count,
                result.code,
            )
            return True, new_evidence, fingerprint
        return False, new_evidence, fingerprint

    def _is_repeated_visible_text(self, text: str) -> bool:
        """True when two consecutive replies are identical and the world did not move."""
        digest = state_digest(self._db)
        stalled = (
            self._last_visible_text is not None
            and text.strip() == self._last_visible_text
            and digest == self._digest_at_last_visible_text
        )
        self._last_visible_text = text.strip()
        self._digest_at_last_visible_text = digest
        if stalled:
            LOGGER.info("No progress: assistant repeated an identical reply with no state change.")
        return stalled

    # -- plumbing ----------------------------------------------------------------------

    def _record_customer_message(self, message: str) -> None:
        self._writer.emit(actor="user", kind="message", visible_to_user=True, content={"text": message})
        self._user_history.append({"role": SPEAKER_CUSTOMER, "content": message})
        _append_message(self._assistant_history, ROLE_USER, message)
        # A new customer message is new evidence; stale repeat counters must not carry over.
        self._repeat_counts.clear()

    def _call_with_retry(self, label: str, action: Callable[[], _T]) -> tuple[_T | None, str]:
        """Run an agent call, retrying a bounded number of times. Never propagates."""
        attempts = max(1, self._settings.max_retries)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                return action(), ""
            except Exception as exc:
                # The provider exception hierarchy is not importable here by design, and a
                # transport failure must become a recorded termination, not a traceback.
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("%s failed on attempt %d/%d: %s", label, attempt, attempts, last_error)
                if attempt < attempts:
                    self._sleep(RETRY_BACKOFF_SECONDS * attempt)
        LOGGER.error("%s exhausted %d attempt(s); terminating with provider_error.", label, attempts)
        return None, last_error


# --------------------------------------------------------------------------------------
# Façade
# --------------------------------------------------------------------------------------


def run_episode(
    task: TaskSpec,
    *,
    assistant: AssistantProtocol,
    user: UserSimulatorProtocol,
    registry: ToolRegistryProtocol,
    parser: ToolParserProtocol,
    episode_id: str | None = None,
    settings: Settings | None = None,
    severity_resolver: SeverityResolver | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> EpisodeResult:
    """Run one episode end to end.

    Budgets come from ``task``; ``settings`` supplies only the no-progress limit and the
    retry count. ``sleep`` is injected so offline tests exercise the retry path instantly.
    """
    return EpisodeRunner(
        task,
        assistant=assistant,
        user=user,
        registry=registry,
        parser=parser,
        episode_id=episode_id,
        settings=settings,
        severity_resolver=severity_resolver,
        sleep=sleep,
    ).run()


__all__ = [
    "AssistantProtocol",
    "EpisodeResult",
    "EpisodeRunner",
    "HISTORY_JOIN_SEPARATOR",
    "INVALID_USER_OUTPUT_REASON",
    "OBSERVATION_SEPARATOR",
    "RETRY_BACKOFF_SECONDS",
    "ROLE_ASSISTANT",
    "ROLE_USER",
    "SPEAKER_CUSTOMER",
    "SPEAKER_SUPPORT",
    "SeverityResolver",
    "TOOL_ERROR_CLOSE",
    "TOOL_ERROR_OPEN",
    "TOOL_RESULT_CLOSE",
    "TOOL_RESULT_OPEN",
    "ToolParserProtocol",
    "ToolRegistryProtocol",
    "UserSimulatorProtocol",
    "encode_parse_error",
    "encode_tool_observation",
    "run_episode",
]
