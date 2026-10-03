"""Text tool-call catalogue and parser. Customer-visible text is separated from privileged calls."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from turncraft.models import (
    ParseErrorCode,
    ParseOutcome,
    ToolCall,
    ToolParseError,
)

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Wire format constants
# --------------------------------------------------------------------------------------

TOOL_CALL_OPEN_TAG = "<tool_call>"
TOOL_CALL_CLOSE_TAG = "</tool_call>"

# Registering the closing tag as an API stop sequence is a real tradeoff, taken here
# deliberately rather than by default:
#   + it very nearly eliminates mid-call truncation, because generation cannot run past the
#     end of a call and burn max_tokens on prose;
#   - it serialises the turn to at most one tool call, since everything the model would have
#     written afterwards is discarded (acceptable: tool execution is serialised anyway);
#   - the matched stop sequence is NOT included in the returned content, so the text comes
#     back with an unbalanced opening tag and would parse as UNCLOSED_TAG.
# The third point is the trap. Callers must run the response through
# reattach_stop_sequence() before parse(). The parser stays correct either way, so a caller
# that does not register the stop sequence needs no special handling.
STOP_SEQUENCE = TOOL_CALL_CLOSE_TAG
STOP_SEQUENCES: tuple[str, ...] = (STOP_SEQUENCE,)

# Messages API stop_reason values this module reasons about.
STOP_REASON_STOP_SEQUENCE = "stop_sequence"
STOP_REASON_MAX_TOKENS = "max_tokens"

# Envelope keys. "input" is accepted as an alias because that is what Anthropic's native
# tool_use block calls the field, and a model that has seen a lot of tool use emits it by
# reflex; accepting it costs one branch and saves a whole round trip in a live episode.
ENVELOPE_NAME_KEY = "name"
ENVELOPE_ARGUMENTS_KEY = "arguments"
ENVELOPE_ARGUMENTS_ALIAS = "input"

# Verbatim span echoed back to the model in a ToolParseError, truncated. 600 characters is
# longer than any well-formed call against our schemas, so a real mistake is shown in full,
# while a runaway generation cannot bloat every event in the persisted trajectory.
MAX_ERROR_SPAN_CHARS = 600

# What an illustrative (fenced) call collapses to in visible text.
ILLUSTRATIVE_PLACEHOLDER = "[tool-call example omitted]"
# What the last-resort scrub leaves behind. Distinct from the above so a trace reader can
# tell "the model was explaining the format" from "the scanner did not recognise this".
REDACTED_MARKUP_PLACEHOLDER = "[tool-call markup removed]"

# blake2s prefix length for call ids. 4 bytes / 8 hex chars is collision-free at the scale
# of a bounded episode (single-digit calls per turn) and keeps traces readable.
_CALL_ID_DIGEST_BYTES = 4

# Rendered for a tool that declares no pydantic argument model.
_NO_ARGUMENT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

# --------------------------------------------------------------------------------------
# Pre-compiled patterns (module level: parse() runs inside the episode loop)
# --------------------------------------------------------------------------------------

# Tolerate "<tool_call >" but nothing more exotic; anything else falls to the residual scrub.
_OPEN_TAG_RE = re.compile(r"<tool_call\s*>")
_CLOSE_TAG_RE = re.compile(r"</tool_call\s*>")

# CommonMark-ish fenced blocks: up to three leading spaces, three or more backticks or
# tildes, optional info string, body, and a closing fence of the same character (a longer
# closing run is absorbed by the trailing [^\n]*). An unclosed fence runs to end of text.
_FENCE_RE = re.compile(
    r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n?.*?(?:^[ \t]{0,3}(?P=fence)[^\n]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Last-resort guard on the trust boundary: any tag-shaped token the span scanner did not
# claim, including attribute-bearing and truncated variants.
_RESIDUAL_TAG_RE = re.compile(r"<\s*/?\s*tool_call\b[^>]*>?", re.IGNORECASE)

_TRAILING_SPACE_RE = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Internal marker standing where an executed call was removed, so the horizontal whitespace
# either side of it collapses to one space instead of leaving "A  B". NUL cannot appear in
# model output; if it somehow did it would collapse to a space, which is harmless.
_SEAM = "\x00"
_SEAM_RE = re.compile(r"[ \t]*\x00[ \t]*")

# Consumes exactly one JSON value from a position in a larger string. This is the "proper
# scan" that makes braces inside strings and nested objects safe.
_DECODER = json.JSONDecoder()


# --------------------------------------------------------------------------------------
# Tool catalogue rendering
# --------------------------------------------------------------------------------------


class ToolSpecLike(Protocol):
    """Structural view of a registry entry.

    Structural rather than a concrete import so that this module and the tool registry can
    be developed independently and cannot form an import cycle (the registry imports
    :func:`parse` and :data:`STOP_SEQUENCE` from here).
    """

    name: str
    description: str
    args_model: type[BaseModel] | None


class ToolRegistryLike(Protocol):
    """Anything that can hand over its tool specs."""

    def specs(self) -> Sequence[ToolSpecLike]: ...


# The two worked examples, assembled separately so the rendered prompt keeps each call on
# one line (the model must see the one-line shape) without a 166-column source line.
_CATALOGUE_EXAMPLE_SIMPLE = '<tool_call>{"name": "lookup_widget", "arguments": {"widget_id": "W-1234", "include_history": true}}</tool_call>'
_CATALOGUE_EXAMPLE_NESTED = (
    '<tool_call>{"name": "lookup_widget", "arguments": '
    '{"note": "customer wrote \\"send the {blue} one\\"", '
    '"filters": {"color": "blue", "sizes": ["s", "m"]}}}</tool_call>'
)

_CATALOGUE_PROTOCOL = (
    """\
# Tool protocol

You have no native tool API. To call a tool, emit the call inline as XML-delimited JSON:

<tool_call>{"name": "<tool name>", "arguments": {<json object>}}</tool_call>

Rules

1. The payload between the tags is one JSON object with two keys: "name" and "arguments".
2. "arguments" is a JSON object; send {} when the tool takes no arguments, and send only
   the keys listed in that tool's schema below.
3. Strict JSON only: double quotes, no single quotes, no trailing commas, no comments.
4. Emit at most one <tool_call> per message. Generation stops at the closing tag, so
   anything you write after it is discarded.
5. Text outside the tags is customer-visible. Never narrate a tool call to the customer and
   never paste raw tool output into a customer sentence.
6. Never send a user id. The session is already authenticated; the environment resolves
   ownership itself and will deny anything outside this customer's records.
7. A <tool_call> written inside a fenced code block is treated as documentation and is not
   executed. That is the only safe way to show this syntax to anyone.
8. A failed call comes back as a structured error. Fix the call or explain the limitation
   to the customer. Never repeat an identical failing call, and never tell the customer an
   action succeeded unless a tool returned success.

Worked examples

The tool `lookup_widget` used here is fictional and shows only the shape. The tools you
actually have are listed under "Available tools".

A call with two simple arguments:

"""
    + _CATALOGUE_EXAMPLE_SIMPLE
    + """

A call whose string argument contains braces and quotes, alongside a nested object. Escape
the inner quotes; do not reword the argument to avoid braces:

"""
    + _CATALOGUE_EXAMPLE_NESTED
    + """

Rejected: 'single quoted' payloads, {"name": "x", "arguments": {},} with its trailing comma,
and any call whose closing tag is missing.

Available tools
"""
)


def _iter_specs(
    registry: ToolRegistryLike | Mapping[str, ToolSpecLike] | Iterable[ToolSpecLike],
) -> list[ToolSpecLike]:
    """Accept a registry object, a name->spec mapping, or a plain iterable of specs."""
    specs_accessor = getattr(registry, "specs", None)
    if callable(specs_accessor):
        return list(specs_accessor())
    if isinstance(registry, Mapping):
        return list(registry.values())
    try:
        return list(registry)  # type: ignore[arg-type]
    except TypeError as exc:
        LOGGER.error(
            "Tool registry of type %s exposes neither .specs(), a mapping, nor iteration (%s).",
            type(registry).__name__,
            exc,
        )
        raise TypeError(
            f"render_tool_catalogue needs a registry with .specs(), a mapping, or an iterable of "
            f"tool specs; got {type(registry).__name__}."
        ) from exc


def _args_schema(spec: ToolSpecLike, tool_name: str) -> dict[str, Any]:
    """JSON Schema for one tool's arguments. Pydantic generates it; we only render it."""
    args_model = getattr(spec, "args_model", None)
    if args_model is None:
        return dict(_NO_ARGUMENT_SCHEMA)
    try:
        schema: dict[str, Any] = dict(args_model.model_json_schema())
    except Exception as exc:  # pydantic raises several unrelated types for unbuildable models
        LOGGER.error(
            "Could not build a JSON schema for tool %s from %r (%s); rendering a no-argument schema.",
            tool_name,
            args_model,
            exc,
        )
        return dict(_NO_ARGUMENT_SCHEMA)
    # Pydantic titles the schema with the class name ("GetOrderArgs"); the heading above it
    # already carries the tool name, and the internal class name is noise to the model.
    schema.pop("title", None)
    return schema


def render_tool_catalogue(
    registry: ToolRegistryLike | Mapping[str, ToolSpecLike] | Iterable[ToolSpecLike],
) -> str:
    """Render the tool catalogue as system-prompt text.

    Order follows the registry, which is insertion-ordered, so the rendered prompt is stable
    across runs -- prompt text is part of what makes an episode reproducible.
    """
    specs = _iter_specs(registry)
    if not specs:
        LOGGER.warning("Rendering an empty tool catalogue; the assistant will have no tools available.")

    parts: list[str] = [_CATALOGUE_PROTOCOL]
    for spec in specs:
        tool_name = getattr(spec, "name", "")
        if not tool_name:
            LOGGER.error("Skipping a tool spec with no name while rendering the catalogue: %r", spec)
            continue
        description = (getattr(spec, "description", "") or "").strip()
        if not description:
            LOGGER.warning("Tool %s has no description; the model receives its schema only.", tool_name)
        parts.append(f"\n### {tool_name}\n")
        if description:
            parts.append(f"\n{description}\n")
        parts.append("\nArguments (JSON Schema):\n\n```json\n")
        parts.append(json.dumps(_args_schema(spec, tool_name), indent=2))
        parts.append("\n```\n")
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Stop-sequence handling
# --------------------------------------------------------------------------------------


def reattach_stop_sequence(
    text: str,
    *,
    stop_reason: str | None = None,
    stop_sequence: str | None = None,
) -> str:
    """Put back the closing tag the Messages API stripped when it matched a stop sequence.

    The API omits the matched stop sequence from the returned content, so a response that
    stopped on ``</tool_call>`` arrives with an unbalanced opening tag and would otherwise
    parse as :attr:`ParseErrorCode.UNCLOSED_TAG`. Pass the response's ``stop_reason`` and
    ``stop_sequence`` fields rather than sniffing the text: only the response knows which
    sequence matched.
    """
    if stop_reason != STOP_REASON_STOP_SEQUENCE:
        return text
    if stop_sequence is None:
        # Only one sequence is ever registered, so this is unambiguous; still worth a trace
        # line, because a caller registering more sequences later would need to pass it.
        LOGGER.debug(
            "stop_reason=%s with no stop_sequence reported; assuming %r.", stop_reason, STOP_SEQUENCE
        )
    elif stop_sequence != STOP_SEQUENCE:
        LOGGER.debug(
            "Response stopped on %r, not the tool-call delimiter; leaving text unchanged.", stop_sequence
        )
        return text
    if text.endswith(STOP_SEQUENCE):
        return text
    return text + STOP_SEQUENCE


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawSpan:
    """One lexically-located ``<tool_call>`` region, before any interpretation."""

    start: int
    end: int
    payload: str  # text between the tags
    text: str  # the whole span, tags included
    closed: bool


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _json_value_end(text: str, payload_start: int) -> int | None:
    """Index just past one complete JSON value starting at ``payload_start``, or None.

    This is what makes ``{"note": "a } b"}`` and nested objects safe. Naive brace matching
    ends the call at the first ``}`` inside the string and produces a plausible-looking
    wrong call, which is worse than a parse error.
    """
    start = _skip_whitespace(text, payload_start)
    if start >= len(text):
        return None
    try:
        _, end = _DECODER.raw_decode(text, start)
    except json.JSONDecodeError as exc:
        # Expected control flow, not a failure: the payload is malformed or truncated, and
        # the caller falls back to a literal close-tag search so the error is bounded.
        LOGGER.debug(
            "Structural JSON scan failed at offset %d (%s); falling back to close-tag search.", start, exc
        )
        return None
    return end


def _scan_spans(text: str) -> list[_RawSpan]:
    """Locate every ``<tool_call>`` region lexically. No interpretation, no errors."""
    spans: list[_RawSpan] = []
    position = 0
    while True:
        opening = _OPEN_TAG_RE.search(text, position)
        if opening is None:
            return spans
        payload_start = opening.end()

        close_start = -1
        close_end = -1
        value_end = _json_value_end(text, payload_start)
        if value_end is not None:
            adjacent = _CLOSE_TAG_RE.match(text, _skip_whitespace(text, value_end))
            if adjacent is not None:
                close_start, close_end = adjacent.start(), adjacent.end()
        if close_start < 0:
            # Search from the end of the decoded value when we have one, so a close tag that
            # appears *inside* a string literal cannot truncate an otherwise valid payload.
            search_from = value_end if value_end is not None else payload_start
            fallback = _CLOSE_TAG_RE.search(text, search_from)
            if fallback is None:
                spans.append(
                    _RawSpan(
                        start=opening.start(),
                        end=len(text),
                        payload=text[payload_start:],
                        text=text[opening.start() :],
                        closed=False,
                    )
                )
                return spans
            close_start, close_end = fallback.start(), fallback.end()

        spans.append(
            _RawSpan(
                start=opening.start(),
                end=close_end,
                payload=text[payload_start:close_start],
                text=text[opening.start() : close_end],
                closed=True,
            )
        )
        position = close_end


def _fenced_ranges(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _FENCE_RE.finditer(text))


def _within(index: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _clip(span_text: str) -> str:
    if len(span_text) <= MAX_ERROR_SPAN_CHARS:
        return span_text
    return span_text[:MAX_ERROR_SPAN_CHARS] + "... [truncated]"


def _error(code: ParseErrorCode, message: str, span_text: str) -> ToolParseError:
    LOGGER.warning("Tool-call parse error [%s]: %s", code.value, message)
    return ToolParseError(code=code.value, message=message, raw_span=_clip(span_text))


def _call_id(span_text: str, ordinal: int) -> str:
    """Deterministic id: same text always yields the same id.

    No uuid4 and no clock, because an episode must replay identically. A repeated identical
    call therefore repeats its id, which the no-progress detector can use for free.
    """
    digest = hashlib.blake2s(span_text.encode("utf-8"), digest_size=_CALL_ID_DIGEST_BYTES).hexdigest()
    return f"call-{ordinal}-{digest}"


def _interpret(
    span: _RawSpan,
    ordinal: int,
    known_tools: Collection[str],
) -> tuple[ToolCall | None, ToolParseError | None]:
    """Turn one located span into a call or a structured error. Never raises."""
    if not span.closed:
        return None, _error(
            ParseErrorCode.UNCLOSED_TAG,
            f"A {TOOL_CALL_OPEN_TAG} was opened but never closed. End every call with "
            f"{TOOL_CALL_CLOSE_TAG} and keep the whole call in one message.",
            span.text,
        )

    payload = span.payload.strip()
    if not payload:
        return None, _error(
            ParseErrorCode.EMPTY_TOOL_CALL,
            'Empty tool call. The tags must contain a JSON object such as {"name": "...", "arguments": {}}.',
            span.text,
        )

    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, _error(
            ParseErrorCode.MALFORMED_JSON,
            f"The tool-call payload is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}). "
            f"Use double quotes, no trailing commas, and put nothing but the JSON object between the tags.",
            span.text,
        )

    if not isinstance(envelope, dict):
        return None, _error(
            ParseErrorCode.MALFORMED_JSON,
            f"The tool-call payload must be a JSON object, not a {type(envelope).__name__}. "
            f'Expected {{"name": "...", "arguments": {{...}}}}.',
            span.text,
        )
    if not envelope:
        return None, _error(
            ParseErrorCode.EMPTY_TOOL_CALL,
            'The tool-call payload is an empty object. It needs at least a "name".',
            span.text,
        )

    raw_name = envelope.get(ENVELOPE_NAME_KEY)
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, _error(
            ParseErrorCode.MISSING_TOOL_NAME,
            f'The tool call has no usable "{ENVELOPE_NAME_KEY}". Supply the tool name as a non-empty string.',
            span.text,
        )
    name = raw_name.strip()

    if name not in known_tools:
        valid = ", ".join(sorted(known_tools)) if known_tools else "(no tools are registered)"
        message = f"Unknown tool {name!r}. Valid tool names are: {valid}."
        close = difflib.get_close_matches(name, sorted(known_tools), n=1)
        if close:
            message += f" Did you mean {close[0]!r}?"
        return None, _error(ParseErrorCode.UNKNOWN_TOOL, message, span.text)

    if ENVELOPE_ARGUMENTS_KEY in envelope:
        raw_arguments = envelope[ENVELOPE_ARGUMENTS_KEY]
    elif ENVELOPE_ARGUMENTS_ALIAS in envelope:
        raw_arguments = envelope[ENVELOPE_ARGUMENTS_ALIAS]
    else:
        raw_arguments = {}
    if raw_arguments is None:  # "arguments": null is a plausible spelling of "no arguments"
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        return None, _error(
            ParseErrorCode.ARGUMENTS_NOT_OBJECT,
            f'"{ENVELOPE_ARGUMENTS_KEY}" must be a JSON object mapping argument names to values, '
            f"not a {type(raw_arguments).__name__}.",
            span.text,
        )

    # Envelope keys beyond name/arguments are ignored rather than rejected: a stray
    # "thinking" or "id" key is harmless, and burning a round trip to teach the model our
    # exact envelope shape buys nothing the schema does not already say.
    return (
        ToolCall(
            id=_call_id(span.text, ordinal),
            name=name,
            raw_arguments=raw_arguments,
            source_span=span.text,
        ),
        None,
    )


def _scrub_residual_markup(match: re.Match[str]) -> str:
    LOGGER.warning(
        "Neutralised unrecognised tool-call markup %r before it could reach the user simulator.",
        _clip(match.group(0)),
    )
    return REDACTED_MARKUP_PLACEHOLDER


def _finalise_visible(text: str) -> str:
    """Scrub residual markup and tidy the whitespace left by removing spans."""
    cleaned = _SEAM_RE.sub(" ", text)
    cleaned = _RESIDUAL_TAG_RE.sub(_scrub_residual_markup, cleaned)
    cleaned = _TRAILING_SPACE_RE.sub("", cleaned)
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def parse(text: str, known_tools: Collection[str]) -> ParseOutcome:
    """Split one raw assistant message into user-visible prose, tool calls, and parse errors.

    ``known_tools`` is the set of registered tool names; membership is all this module needs
    (argument validation is the registry's job). Errors are returned, never raised: each one
    is an observation the model can act on.
    """
    if not text:
        return ParseOutcome(visible_text="")

    spans = _scan_spans(text)
    if not spans:
        return ParseOutcome(visible_text=_finalise_visible(text))

    fenced = _fenced_ranges(text)
    calls: list[ToolCall] = []
    errors: list[ToolParseError] = []
    segments: list[str] = []
    cursor = 0
    ordinal = 0

    for span in spans:
        segments.append(text[cursor : span.start])
        cursor = span.end
        if _within(span.start, fenced):
            # Documented decision: inside a fence the model is explaining the format, not
            # invoking it. Not executed, not an error -- but still collapsed, because "not
            # executed" must not degrade into "syntax leaked to the customer".
            LOGGER.debug(
                "Ignoring illustrative tool call inside a fenced code block at offset %d.", span.start
            )
            segments.append(ILLUSTRATIVE_PLACEHOLDER)
            continue
        ordinal += 1
        segments.append(_SEAM)
        call, error = _interpret(span, ordinal, known_tools)
        if call is not None:
            calls.append(call)
        if error is not None:
            errors.append(error)
    segments.append(text[cursor:])

    return ParseOutcome(visible_text=_finalise_visible("".join(segments)), calls=calls, errors=errors)


__all__ = [
    "ENVELOPE_ARGUMENTS_ALIAS",
    "ENVELOPE_ARGUMENTS_KEY",
    "ENVELOPE_NAME_KEY",
    "ILLUSTRATIVE_PLACEHOLDER",
    "MAX_ERROR_SPAN_CHARS",
    "REDACTED_MARKUP_PLACEHOLDER",
    "STOP_REASON_MAX_TOKENS",
    "STOP_REASON_STOP_SEQUENCE",
    "STOP_SEQUENCE",
    "STOP_SEQUENCES",
    "TOOL_CALL_CLOSE_TAG",
    "TOOL_CALL_OPEN_TAG",
    "ToolRegistryLike",
    "ToolSpecLike",
    "parse",
    "reattach_stop_sequence",
    "render_tool_catalogue",
]
