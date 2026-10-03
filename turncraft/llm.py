"""Provider adapter and provider-neutral chat protocol. No SDK agent loop is used."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import anthropic

from turncraft.config import Settings, get_settings, require_api_key
from turncraft.models import ContractModel

LOGGER = logging.getLogger(__name__)

# --- Transport constants --------------------------------------------------------------

# The runner owns retries for each logical assistant/user call. Both the SDK and this
# adapter therefore make exactly one transport attempt; otherwise configured limits
# multiply across layers.
SDK_INTERNAL_RETRIES = 0

# First backoff. Small enough that a single blip is invisible on stage.
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_BACKOFF_FACTOR = 2.0
# Ceiling per sleep. With the default three retries the worst case is ~3.5s of waiting,
# which is a pause; anything longer reads as a hang.
RETRY_MAX_DELAY_SECONDS = 8.0
# No jitter: one process, one request in flight, and a predictable worst-case latency is
# worth more here than thundering-herd protection we cannot trigger.

# 4xx means the request is wrong and will be wrong again; only 5xx is worth repeating.
SERVER_ERROR_STATUS_FLOOR = 500
RETRY_AFTER_HEADER = "retry-after"
# A server may ask us to wait minutes. Bounded so a rate limit fails the episode with a
# recorded reason instead of freezing the demo.
MAX_RETRY_AFTER_SECONDS = 30.0

# Response shape. Only text blocks are user/parser visible; thinking blocks are dropped.
TEXT_BLOCK_TYPE = "text"
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class ProviderError(RuntimeError):
    """A provider-side failure, normalised so callers never import ``anthropic``.

    Raised for non-retryable API errors, for exhausted retries, and for locally
    detected requests the Messages API would reject. The originating SDK exception is
    always preserved as ``__cause__``. Callers map this onto
    ``TerminationReason.PROVIDER_ERROR``.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.attempts = attempts


class ChatTurn(ContractModel):
    """One provider-neutral model turn.

    ``text`` is the concatenation of the response's text blocks — the only thing the
    tool-call parser and the user simulator ever see. ``stop_reason`` is passed through
    verbatim (``end_turn`` / ``max_tokens`` / ``stop_sequence`` / ``refusal`` / ...);
    the harness needs it to tell a finished turn from a truncated one, which is how a
    half-written ``<tool_call>`` envelope is diagnosed rather than guessed at. The
    matched stop sequence itself is not carried: the caller supplied the list and
    ``stop_reason == "stop_sequence"`` is enough to know one fired.
    """

    text: str
    stop_reason: str
    usage: dict[str, int] = {}
    raw_model: str = ""


@runtime_checkable
class ChatBackend(Protocol):
    """What the harness is allowed to ask of a language model.

    Two implementations exist: :class:`AnthropicBackend` (live) and
    ``tests.fakes.ScriptedBackend`` (offline). The model id is bound to the backend
    instance, not passed per call, so the assistant and the user simulator are separate
    objects with separate budgets and cannot accidentally share a model.
    """

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stop_sequences: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> ChatTurn:
        """Return one completion.

        Args:
            system: System prompt. Sent as the top-level ``system`` parameter; omitted
                when empty.
            messages: Alternating ``{"role": "user"|"assistant", "content": ...}`` dicts.
                Must be non-empty and must start with a user turn.
            max_tokens: Hard output cap. Required by the Messages API.
            stop_sequences: Optional strings that end generation early.
            output_schema: Optional JSON Schema. When given, the response is constrained
                to JSON matching it (the user simulator's ``{message, done, reason}``).

        Raises:
            ProviderError: any provider-side failure, including exhausted retries.
        """
        ...


def _extract_text(content: Iterable[Any]) -> str:
    """Join the response's text blocks. Thinking and any future block type are ignored."""
    return "".join(
        getattr(block, "text", "") for block in content if getattr(block, "type", None) == TEXT_BLOCK_TYPE
    )


def _extract_usage(usage: Any) -> dict[str, int]:
    """Copy the integer token counters into a plain JSON-safe dict."""
    if usage is None:
        return {}
    collected: dict[str, int] = {}
    for name in USAGE_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            collected[name] = value
    return collected


def _retry_after_seconds(exc: anthropic.APIStatusError) -> float | None:
    """Honour a ``retry-after`` header when the server sends a usable one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get(RETRY_AFTER_HEADER)
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        # Header may be an HTTP-date rather than a delta-seconds count. Not worth
        # parsing; fall back to our own backoff rather than guessing.
        LOGGER.warning("Ignoring unparseable %s header %r; using local backoff.", RETRY_AFTER_HEADER, raw)
        return None
    if seconds <= 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


class AnthropicBackend:
    """A thin, hand-rolled Messages API client. Implements :class:`ChatBackend`."""

    def __init__(
        self,
        *,
        model: str,
        settings: Settings | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Build a live backend.

        Raises:
            MissingApiKeyError: if ``ANTHROPIC_API_KEY`` is unset. Constructing this
                class is the point at which credentials become mandatory; importing
                this module never requires them.
        """
        resolved = settings if settings is not None else get_settings()
        api_key = require_api_key(resolved)
        if not model.strip():
            raise ValueError("Configure ASSISTANT_MODEL and USER_SIM_MODEL before live execution.")
        self._model = model
        # Kept as an accepted keyword for public API compatibility. Retry ownership lives
        # in EpisodeRunner, so this adapter always performs one attempt.
        if max_retries not in (None, 0):
            LOGGER.debug(
                "Ignoring AnthropicBackend(max_retries=%d); EpisodeRunner is the retry owner.",
                max_retries,
            )
        self._max_retries = 0
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=resolved.request_timeout_seconds,
            max_retries=SDK_INTERNAL_RETRIES,
        )

    @classmethod
    def for_assistant(cls, settings: Settings | None = None) -> AnthropicBackend:
        """Backend for the tool-using assistant, on the configured assistant model."""
        resolved = settings if settings is not None else get_settings()
        return cls(model=resolved.assistant_model, settings=resolved)

    @classmethod
    def for_user_sim(cls, settings: Settings | None = None) -> AnthropicBackend:
        """Backend for the user simulator, on the configured (cheaper) user model."""
        resolved = settings if settings is not None else get_settings()
        return cls(model=resolved.user_model, settings=resolved)

    @property
    def model(self) -> str:
        """The model id every call from this backend is sent to."""
        return self._model

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stop_sequences: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> ChatTurn:
        """See :meth:`ChatBackend.complete`."""
        if not messages:
            raise ProviderError(
                "complete() needs at least one message; the Messages API rejects an empty list."
            )
        if max_tokens <= 0:
            raise ProviderError(f"max_tokens must be positive, got {max_tokens}.")

        request = self._build_request(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            output_schema=output_schema,
        )
        total_attempts = self._max_retries + 1  # max_retries counts retries, not attempts

        for attempt in range(1, total_attempts + 1):
            try:
                message = self._client.messages.create(**request)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                status_code = getattr(exc, "status_code", None)
                request_id = getattr(exc, "request_id", None)
                retryable, delay = self._retry_plan(exc, attempt)
                if not retryable:
                    LOGGER.error(
                        "Anthropic call failed, not retryable (model=%s status=%s request_id=%s): %s",
                        self._model,
                        status_code,
                        request_id,
                        exc,
                    )
                    raise ProviderError(
                        str(exc), status_code=status_code, request_id=request_id, attempts=attempt
                    ) from exc
                if attempt >= total_attempts:
                    LOGGER.error(
                        "Anthropic call failed after %d attempt(s) (model=%s status=%s request_id=%s): %s",
                        attempt,
                        self._model,
                        status_code,
                        request_id,
                        exc,
                    )
                    raise ProviderError(
                        f"giving up after {attempt} attempt(s): {exc}",
                        status_code=status_code,
                        request_id=request_id,
                        attempts=attempt,
                    ) from exc
                LOGGER.warning(
                    "Retrying Anthropic call in %.1fs (attempt %d/%d, model=%s status=%s request_id=%s): %s",
                    delay,
                    attempt,
                    total_attempts,
                    self._model,
                    status_code,
                    request_id,
                    exc,
                )
                time.sleep(delay)
            except anthropic.AnthropicError as exc:
                # Anything else the SDK can raise (webhook/response-validation errors).
                # Wrapped, not swallowed, so no SDK type escapes this module.
                LOGGER.error("Unexpected Anthropic SDK error (model=%s): %s", self._model, exc)
                raise ProviderError(str(exc), attempts=attempt) from exc
            else:
                return self._to_chat_turn(message)

        # Unreachable: every path above returns or raises. Guards against a future edit
        # dropping the terminal raise and turning a failure into a silent None.
        raise ProviderError("retry loop ended without a response", attempts=total_attempts)

    def _build_request(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stop_sequences: list[str] | None,
        output_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the Messages payload. Optional keys are omitted, never sent as None."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            request["system"] = system
        if stop_sequences:
            request["stop_sequences"] = stop_sequences
        if output_schema is not None:
            request["output_config"] = {"format": {"type": "json_schema", "schema": output_schema}}
        return request

    def _retry_plan(self, exc: Exception, attempt: int) -> tuple[bool, float]:
        """Decide whether to retry ``exc`` and how long to wait first."""
        backoff = min(
            RETRY_BASE_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** (attempt - 1)),
            RETRY_MAX_DELAY_SECONDS,
        )
        if isinstance(exc, anthropic.RateLimitError):
            # Subclass of APIStatusError, so this branch must come first.
            return True, _retry_after_seconds(exc) or backoff
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code >= SERVER_ERROR_STATUS_FLOOR, backoff
        # APIConnectionError / APITimeoutError: the socket died before a status existed.
        # Same transient class as a 5xx, so treated the same way.
        return True, backoff

    def _to_chat_turn(self, message: Any) -> ChatTurn:
        """Normalise an SDK ``Message`` into a :class:`ChatTurn`."""
        request_id = getattr(message, "_request_id", None)
        text = _extract_text(message.content or [])
        stop_reason = message.stop_reason or ""
        if not text:
            # HTTP 200 with no text block: truncation mid-thinking, or a refusal.
            # Reported rather than silently handed on as an empty assistant turn.
            LOGGER.warning(
                "Anthropic response carried no text block (model=%s stop_reason=%s request_id=%s).",
                message.model,
                stop_reason,
                request_id,
            )
        return ChatTurn(
            text=text,
            stop_reason=stop_reason,
            usage=_extract_usage(message.usage),
            raw_model=message.model or self._model,
        )


__all__ = [
    "AnthropicBackend",
    "ChatBackend",
    "ChatTurn",
    "MAX_RETRY_AFTER_SECONDS",
    "ProviderError",
    "RETRY_BACKOFF_FACTOR",
    "RETRY_BASE_DELAY_SECONDS",
    "RETRY_MAX_DELAY_SECONDS",
    "SDK_INTERNAL_RETRIES",
    "SERVER_ERROR_STATUS_FLOOR",
]
