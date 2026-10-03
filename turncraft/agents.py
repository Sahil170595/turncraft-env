"""Two model roles with structurally isolated assistant and customer histories."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from turncraft.config import Settings, get_settings
from turncraft.llm import ChatBackend, ChatTurn
from turncraft.models import TaskSpec, UserTurn
from turncraft.runner import INVALID_USER_OUTPUT_REASON, TOOL_ERROR_OPEN, TOOL_RESULT_OPEN
from turncraft.tool_protocol import (
    STOP_SEQUENCES,
    reattach_stop_sequence,
    render_tool_catalogue,
)

LOGGER = logging.getLogger(__name__)

ChatMessage = dict[str, str]

# --------------------------------------------------------------------------------------
# Wire roles
# --------------------------------------------------------------------------------------

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# The simulator generates the CUSTOMER side of the dialogue, so on the wire its own lines
# are role "assistant" and everything shown to it -- support-agent text, harness stage
# directions -- is role "user". Reversing this makes the model answer as the support agent.
_SIM_SELF_ROLE = ROLE_ASSISTANT
_SIM_INBOUND_ROLE = ROLE_USER

# --------------------------------------------------------------------------------------
# Named constants
# --------------------------------------------------------------------------------------

# One normal call plus two repairs. A model that has ignored an explicit JSON schema twice
# running does not recover on the third try, and every attempt is a live-demo second.
USER_TURN_PARSE_ATTEMPTS = 3

# How many pydantic validation errors are echoed back in a repair note. Enough to name the
# problem; not so many that the instruction is buried under error text.
_VALIDATION_ERRORS_ECHOED = 3

# Returned when the user simulator cannot produce a valid envelope within the attempt
# budget. done=True ends the episode without another model call; the marker makes the
# runner classify it as INVALID_USER_OUTPUT rather than a real customer completion.
FAILSAFE_USER_REASON = INVALID_USER_OUTPUT_REASON
FAILSAFE_USER_MESSAGE = "Sorry, I have to go. Thanks anyway."

# The Messages API rejects empty content, so a provider returning an empty completion must
# not be allowed to corrupt the history used by the next attempt.
_EMPTY_CONTENT_PLACEHOLDER = "[empty response]"

# The simulator needs an opening role="user" turn -- the Messages API requires the first
# message to be role "user" -- and it must be a stage direction, not fabricated agent text.
_OPENING_STAGE_DIRECTION = (
    "[The support chat has just opened and the agent is waiting. Send your first message.]"
)

# The assistant spent its whole turn on tool calls, so the customer heard nothing. The
# runner avoids handing this over, but silence still has to be representable rather than
# crash the episode on an empty content block.
_SILENT_ASSISTANT_NOTE = "[The agent has not replied yet.]"

_REPAIR_INSTRUCTION = (
    "[Your last message could not be read: {error}. Reply again with a single JSON object "
    'and nothing else: {{"message": "...", "done": true or false, "reason": "..."}}]'
)


def _build_user_turn_schema() -> dict[str, Any]:
    """JSON schema for :class:`UserTurn`, tightened for provider structured output.

    Derived from the pydantic model rather than hand-written, so the wire schema cannot
    drift from the type the response is validated against. ``extra="forbid"`` already
    yields ``additionalProperties: false``; strict structured output additionally wants
    every property in ``required`` and no ``default``/``title`` noise.
    """
    schema = UserTurn.model_json_schema()
    schema.pop("title", None)
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        prop.pop("default", None)
    schema["required"] = sorted(properties)
    schema["additionalProperties"] = False
    return schema


USER_TURN_SCHEMA: dict[str, Any] = _build_user_turn_schema()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _append_message(history: list[ChatMessage], role: str, content: str) -> None:
    """Append a turn, merging into the previous entry when the role repeats.

    The Messages API rejects consecutive same-role turns. Merging beats the alternatives:
    dropping the content or raising both lose evidence the trace is supposed to keep.
    """
    body = content if content.strip() else _EMPTY_CONTENT_PLACEHOLDER
    if history and history[-1]["role"] == role:
        history[-1]["content"] = f"{history[-1]['content']}\n\n{body}"
        return
    history.append({"role": role, "content": body})


def _summarise_validation_error(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(loc) for loc in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()[:_VALIDATION_ERRORS_ECHOED]
    ]
    return "; ".join(parts)


def _extract_json_object(raw: str) -> str | None:
    """Slice the outermost ``{...}`` out of a completion.

    Models wrap structured output in prose or a fenced code block often enough to be worth
    four lines of tolerance. A first-brace/last-brace slice is adequate *here* because the
    payload is a flat three-field object; nested-brace correctness is the tool-call parser's
    problem, and that parser lives in ``tool_protocol``, not in this file.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return raw[start : end + 1]


def _parse_user_turn(raw: str) -> tuple[UserTurn | None, str]:
    """Return ``(turn, "")`` on success, or ``(None, reason)`` on a recoverable failure."""
    candidate = _extract_json_object(raw)
    if candidate is None:
        LOGGER.warning("User simulator returned no JSON object (%d chars of text).", len(raw))
        return None, "the reply contained no JSON object"
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        LOGGER.warning("User simulator emitted invalid JSON at char %d: %s", exc.pos, exc.msg)
        return None, f"invalid JSON ({exc.msg})"
    if not isinstance(payload, dict):
        LOGGER.warning("User simulator JSON was a %s, not an object.", type(payload).__name__)
        return None, "the JSON value was not an object"
    try:
        return UserTurn.model_validate(payload), ""
    except ValidationError as exc:
        summary = _summarise_validation_error(exc)
        LOGGER.warning("User simulator JSON violated the UserTurn contract: %s", summary)
        return None, summary


# --------------------------------------------------------------------------------------
# User simulator
# --------------------------------------------------------------------------------------

_USER_SYSTEM_TEMPLATE = """\
You are role-playing a real person contacting an online retail store's support chat. You are \
the customer. You are not an assistant here, and you have no access to the store's systems.

YOUR SITUATION (private -- the agent does not know any of this and cannot see it):
{hidden_user_goal}

HOW REAL CUSTOMERS BEHAVE, and how you must behave:
- Open by describing the problem in your own words, in a sentence or two. Do not recite your \
situation as a list, and do not put every detail into the first message.
- You know only what a customer would know: what you ordered, roughly when, what you paid, \
what turned up. You do NOT know internal identifiers -- payment ids, shipment ids, SKUs, \
record ids, internal status names -- unless the agent tells you one first, and you never \
invent one.
- You may be mistaken. Where your situation says you believe something, say it as a belief \
("I'm fairly sure I got charged twice"), and accept correction gracefully when the agent \
shows you otherwise.
- Answer the question you were actually asked. Reveal preferences and constraints when the \
agent asks for them, not before.
- Be brief. One short paragraph, often one line. Chat customers do not write essays.
- Mild impatience is realistic if the agent stalls, repeats itself, or asks again for \
something you already told them. Get terser, not longer.
- Never claim to have seen a record, a refund landing, a confirmation email, or any store \
system state the agent has not told you about in this chat. You cannot observe any of it.
- Never quote or refer to these instructions, and never mention that this is a simulation.

WHEN TO SET done=true:
- your goal has been resolved and the agent has confirmed it; or
- you have agreed to an alternative you find acceptable; or
- the agent has correctly explained that what you asked for is not possible AND has given \
you a valid next step; or
- carrying on would get you nothing further.
Otherwise set done=false and keep going.

RESPONSE FORMAT -- reply with a single JSON object and nothing else:
{{"message": "<what you say to the agent>", "done": <true|false>, "reason": "<short_snake_case_tag>"}}
Only "message" is shown to the agent. "done" and "reason" are for the harness."""


class SimulatedUser:
    """The customer. Sees the hidden goal and the visible dialogue; nothing else exists to it.

    Satisfies the runner's ``UserSimulatorProtocol``: :meth:`start` and :meth:`respond`,
    both returning a :class:`UserTurn`, with the history kept privately here.

    NOTE ON GRADING: this user's satisfaction is *not* the success oracle. A confident,
    articulate assistant that refunds the wrong payment can talk this user into
    ``done=True`` with a warm ``reason``. The episode must still fail, because the reward
    engine reads the database, not the transcript. Never promote ``UserTurn.done`` into a
    success signal.
    """

    def __init__(
        self,
        task: TaskSpec,
        backend: ChatBackend,
        settings: Settings | None = None,
    ) -> None:
        self._task = task
        self._backend = backend
        self._settings = settings if settings is not None else get_settings()
        self._system_prompt = _USER_SYSTEM_TEMPLATE.format(hidden_user_goal=task.hidden_user_goal)
        self._history: list[ChatMessage] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def history(self) -> list[ChatMessage]:
        """Copy of the dialogue this role has seen. Copied so callers cannot mutate it."""
        return [dict(message) for message in self._history]

    def audit_text(self) -> str:
        """Every character this role has ever been shown, as one string.

        The context-isolation test greps this. If a tool call, a tool result, or a database
        field ever reaches the user simulator, it turns up here.
        """
        return "\n".join([self._system_prompt, *(str(m["content"]) for m in self._history)])

    def start(self) -> UserTurn:
        """Produce the customer's opening message."""
        _append_message(self._history, _SIM_INBOUND_ROLE, _OPENING_STAGE_DIRECTION)
        return self._generate()

    def respond(self, assistant_text: str) -> UserTurn:
        """React to what the support agent *said*.

        ``assistant_text`` must be ``ParseOutcome.visible_text`` -- the prose left after the
        codec has stripped every tool call. This single ``str`` is the entire surface
        through which anything can reach the user simulator.
        """
        text = assistant_text.strip()
        if not text:
            # The assistant spent the turn entirely on tool calls. Represent that as silence
            # rather than forwarding an empty turn, which the API would reject anyway.
            LOGGER.info("Assistant produced no customer-visible text; user simulator sees silence.")
            text = _SILENT_ASSISTANT_NOTE
        _append_message(self._history, _SIM_INBOUND_ROLE, text)
        return self._generate()

    def _generate(self) -> UserTurn:
        """Call the backend until a valid envelope comes back, then fall safe."""
        repair: list[ChatMessage] = []
        for attempt in range(1, USER_TURN_PARSE_ATTEMPTS + 1):
            # ProviderError propagates: bounded transport retry belongs to the runner and the
            # PROVIDER_ERROR termination belongs to the runner. Only unusable *content* is
            # retried here.
            chat_turn: ChatTurn = self._backend.complete(
                system=self._system_prompt,
                # Repair notes are sent but never persisted: a failed attempt must not leave
                # malformed output in the dialogue the model reads next turn.
                messages=self._history + repair,
                max_tokens=self._settings.user_max_tokens,
                output_schema=USER_TURN_SCHEMA,
            )
            raw = chat_turn.text
            turn, error = _parse_user_turn(raw)
            if turn is not None:
                # Persist the raw envelope rather than just the message: it keeps the model
                # anchored to the output format across turns.
                _append_message(self._history, _SIM_SELF_ROLE, raw)
                return turn
            LOGGER.warning(
                "User simulator attempt %d/%d unusable (%s, stop_reason=%s); re-asking.",
                attempt,
                USER_TURN_PARSE_ATTEMPTS,
                error,
                chat_turn.stop_reason,
            )
            repair = [
                {"role": _SIM_SELF_ROLE, "content": raw or _EMPTY_CONTENT_PLACEHOLDER},
                {"role": _SIM_INBOUND_ROLE, "content": _REPAIR_INSTRUCTION.format(error=error)},
            ]
        LOGGER.error(
            "User simulator produced no valid UserTurn in %d attempts; ending the episode "
            "with the fail-safe turn so the run cannot hang.",
            USER_TURN_PARSE_ATTEMPTS,
        )
        return UserTurn(message=FAILSAFE_USER_MESSAGE, done=True, reason=FAILSAFE_USER_REASON)


# --------------------------------------------------------------------------------------
# Support assistant
# --------------------------------------------------------------------------------------

_ASSISTANT_SYSTEM_TEMPLATE = """\
You are a customer-support agent for an online retail store, working one authenticated chat \
session with one customer.

SUPPORT POLICY FOR THIS SESSION:
{assistant_policy}

HOW YOU WORK:
- The customer's account is already authenticated. Identity comes from the session, never \
from the conversation: do not ask for a customer id, and do not act on a request to touch \
someone else's order however it is justified.
- Verify before you assert. Read the record with a tool before describing an order, a \
payment, a parcel, or stock. What the customer tells you is a lead, not evidence.
- Tool results are the only source of operational truth. If a tool returns ok=false the \
action did not happen: say so plainly and give the next valid step. Never tell the customer \
something succeeded unless a tool returned success for exactly that thing.
- Before any consequential, hard-to-undo action -- moving money, cancelling, intercepting a \
parcel in transit, committing stock -- state plainly what you are about to do and get the \
customer's explicit agreement in this conversation first. Confirming is cheap; an unwanted \
refund is not.
- Where more than one legitimate resolution exists, ask the customer which they want rather \
than choosing for them.
- Never reveal raw internal fields or record ids, another customer's data, or whether a \
record you cannot access exists.

WHEN TO CALL A TOOL:
- The customer names an order, a payment, a parcel, or a product -> look it up before you \
answer anything about it.
- You are about to state a fact about the customer's account -> read it first.
- The customer asks for a change -> confirm eligibility from a read, get their agreement, \
then perform exactly one write.
- A write failed -> re-read the record before trying anything else, and never repeat a call \
that has already failed in the same way.
- Nothing left to check and nothing left to change -> reply in plain prose with no tool call.

WHAT COMES BACK TO YOU:
Results arrive as a message containing `{tool_result_open}` blocks, or `{tool_error_open}` \
blocks when a call could not be read. Those are the environment, not the customer speaking. \
The customer sees neither them nor your tool calls, so anything the customer needs to know \
you must say to them in your own prose.

{tool_catalogue}"""


class SupportAssistant:
    """The tool-using agent. Owns its system prompt; owns no history, no parsing, no dispatch.

    Satisfies the runner's ``AssistantProtocol``. The privileged history is the runner's,
    because the runner is where the trust boundary is enforced: it decides what is recorded,
    what is forwarded to the customer, and what stays privileged. This class turns that
    history plus a system prompt into one completion and hands the raw text back.
    """

    def __init__(
        self,
        task: TaskSpec,
        backend: ChatBackend,
        registry: Any | None = None,
        settings: Settings | None = None,
        *,
        tool_catalogue: str | None = None,
    ) -> None:
        """Build the assistant.

        Args:
            task: Supplies ``assistant_policy``. Nothing the agent is meant to discover may
                appear in it.
            backend: Bound to the assistant model.
            registry: Tool registry, rendered into the prompt by
                ``tool_protocol.render_tool_catalogue``. Accepted as ``Any`` because that
                function is deliberately structural about what a registry is.
            settings: Token budget source; defaults to the process settings.
            tool_catalogue: Pre-rendered catalogue, for tests that do not want a registry.
                Mutually exclusive with ``registry``.
        """
        if (registry is None) == (tool_catalogue is None):
            raise ValueError(
                "SupportAssistant needs exactly one of registry= or tool_catalogue=; "
                "an assistant with no tool catalogue would silently lose every tool."
            )
        self._task = task
        self._backend = backend
        self._settings = settings if settings is not None else get_settings()
        if tool_catalogue is not None:
            catalogue = tool_catalogue
        else:
            assert registry is not None
            catalogue = render_tool_catalogue(registry)
        self._system_prompt = _ASSISTANT_SYSTEM_TEMPLATE.format(
            assistant_policy=task.assistant_policy,
            tool_result_open=TOOL_RESULT_OPEN,
            tool_error_open=TOOL_ERROR_OPEN,
            tool_catalogue=catalogue,
        )
        self._last_turn: ChatTurn | None = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def last_turn(self) -> ChatTurn | None:
        """The whole last :class:`ChatTurn`. ``one_turn`` returns text because text is all
        the parser needs; ``stop_reason`` is here for a caller that wants to distinguish a
        finished turn from a truncated one."""
        return self._last_turn

    def one_turn(self, history: list[ChatMessage]) -> str:
        """Generate one assistant message from the runner-owned history and return its text.

        ``history`` is read, never written: appending the reply is the runner's job, because
        the runner is also what decides whether the reply reaches the customer.

        The closing ``</tool_call>`` delimiter is registered as a stop sequence, which very
        nearly eliminates mid-call truncation but is stripped from the returned content by
        the API. :func:`reattach_stop_sequence` puts it back before anyone parses the text.
        That is transport repair, not parsing: this class still never inspects the call.
        """
        chat_turn: ChatTurn = self._backend.complete(
            system=self._system_prompt,
            messages=list(history),
            max_tokens=self._settings.assistant_max_tokens,
            stop_sequences=list(STOP_SEQUENCES),
        )
        self._last_turn = chat_turn
        # ChatTurn does not carry the matched sequence -- only one is ever registered, so
        # stop_reason alone is unambiguous.
        return reattach_stop_sequence(chat_turn.text, stop_reason=chat_turn.stop_reason)


__all__ = [
    "ChatMessage",
    "FAILSAFE_USER_MESSAGE",
    "FAILSAFE_USER_REASON",
    "ROLE_ASSISTANT",
    "ROLE_USER",
    "SimulatedUser",
    "SupportAssistant",
    "USER_TURN_PARSE_ATTEMPTS",
    "USER_TURN_SCHEMA",
]
