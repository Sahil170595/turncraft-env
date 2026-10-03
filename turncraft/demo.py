"""Turncraft CLI: scripted controls execute real tools; optional live mode uses two model roles."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turncraft import db as db_module
from turncraft import tool_protocol
from turncraft.agents import SimulatedUser, SupportAssistant
from turncraft.config import MissingApiKeyError, get_settings
from turncraft.db import CustomerServiceDB, format_cents
from turncraft.llm import AnthropicBackend, ProviderError
from turncraft.models import (
    DatabaseState,
    EpisodeContext,
    TaskSpec,
    TerminationReason,
    ToolCall,
    ToolErrorCode,
    Trajectory,
    TrajectoryEvent,
)
from turncraft.reward_weights import DEFAULT_WEIGHTS
from turncraft.rewards import compute_reward
from turncraft.runner import run_episode
from turncraft.task_registry import ALL_CONTROLS, get_task, list_tasks
from turncraft.tools import build_default_registry

# --------------------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------------------

RULE = "-" * 78
INDENT = "    "
# The single most valuable thing on screen: everything the customer could never see is
# indented and tagged, so the trust boundary is legible at a glance during the demo.
HIDDEN_TAG = "[HIDDEN FROM USER]"
_SPEAKER_LABEL = {"user": "Customer", "assistant": "Agent"}

MODE_SCRIPTED_GOOD = "scripted-good"
MODE_SCRIPTED_BAD = "scripted-bad"
MODE_LIVE = "live"
MODE_REPLAY = "replay"
MODES = (MODE_SCRIPTED_GOOD, MODE_SCRIPTED_BAD, MODE_LIVE, MODE_REPLAY)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# --------------------------------------------------------------------------------------
# Episode artifacts: the common currency every mode produces
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EpisodeArtifacts:
    episode_id: str
    task: TaskSpec
    trajectory: Trajectory
    initial_db: DatabaseState
    final_db: DatabaseState


# --------------------------------------------------------------------------------------
# scripted-good / scripted-bad: replay a control trajectory through the real registry
# --------------------------------------------------------------------------------------


def _scripted_control_key(mode: str, controls: dict[str, Trajectory]) -> str:
    if mode == MODE_SCRIPTED_GOOD:
        return "oracle"
    # scripted-bad: forbidden if the pack shipped one, else the weaker near_miss control.
    return "forbidden" if "forbidden" in controls else "near_miss"


def _execute_script(task: TaskSpec, script: Trajectory, episode_id: str) -> tuple[Trajectory, DatabaseState]:
    """Replay ``script`` against a fresh world: dialogue text trusted, tool effects recomputed.

    Every ``message`` / ``invalid_tool_call`` event is carried through verbatim -- that is
    the authored plot. Every ``tool_call`` is re-executed against ``build_default_registry()``
    so its result, severity, and any policy-denial event are the REAL outcome of running that
    call against this world, never the control author's guess at one.
    """
    registry = build_default_registry()
    view = CustomerServiceDB(task.initial_db)
    context = EpisodeContext(
        episode_id=episode_id,
        task_id=task.task_id,
        authenticated_user_id=task.authenticated_user_id,
        max_dialogue_turns=task.max_dialogue_turns,
        max_tool_rounds_per_turn=task.max_tool_rounds_per_turn,
    )

    events: list[TrajectoryEvent] = []
    sequence = 0
    termination_detail = ""

    for event in sorted(script.events, key=lambda e: e.sequence):
        if event.kind in ("tool_result", "policy_denial"):
            continue  # recomputed alongside its tool_call, below
        if event.kind == "termination":
            termination_detail = str(event.content.get("detail", ""))
            continue  # synthesised once, at the end, from script.termination_reason

        if event.kind != "tool_call":
            # message / invalid_tool_call: verbatim content, resequenced to stay contiguous.
            events.append(event.model_copy(update={"sequence": sequence}))
            sequence += 1
            continue

        call = ToolCall(
            id=str(event.content.get("call_id") or f"call-{sequence}"),
            name=str(event.content.get("name", "")),
            raw_arguments=dict(event.content.get("arguments") or {}),
            source_span="scripted",
        )
        severity = registry.severity_for(call, view.state)
        events.append(
            TrajectoryEvent(
                sequence=sequence,
                actor="assistant",
                kind="tool_call",
                visible_to_user=False,
                content={"call_id": call.id, "name": call.name, "arguments": call.raw_arguments},
                severity=severity,
            )
        )
        sequence += 1

        result = registry.execute(call, context, view.state)
        events.append(
            TrajectoryEvent(
                sequence=sequence,
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
        )
        sequence += 1

        if not result.ok and result.code == ToolErrorCode.POLICY_DENIED:
            events.append(
                TrajectoryEvent(
                    sequence=sequence,
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
            )
            sequence += 1

    termination_reason = script.termination_reason or TerminationReason.USER_DONE
    events.append(
        TrajectoryEvent(
            sequence=sequence,
            actor="environment",
            kind="termination",
            visible_to_user=False,
            content={"reason": termination_reason.value, "detail": termination_detail},
        )
    )

    trajectory = Trajectory(
        episode_id=episode_id, task_id=task.task_id, events=events, termination_reason=termination_reason
    )
    return trajectory, view.state.model_copy(deep=True)


def _run_scripted(task: TaskSpec, mode: str, control: str | None = None) -> _EpisodeArtifacts:
    controls = ALL_CONTROLS.get(task.task_id)
    if not controls:
        raise SystemExit(f"Task {task.task_id!r} has no CONTROLS entry; cannot run --mode {mode}.")
    key = control or _scripted_control_key(mode, controls)
    script = controls.get(key)
    if script is None:
        raise SystemExit(
            f"Task {task.task_id!r} has no {key!r} control trajectory; cannot run --mode {mode}."
        )

    episode_id = f"{mode}-{task.task_id}" + (f"-{control}" if control else "")
    trajectory, final_db = _execute_script(task, script, episode_id)
    return _EpisodeArtifacts(
        episode_id=episode_id,
        task=task,
        trajectory=trajectory,
        initial_db=task.initial_db.model_copy(deep=True),
        final_db=final_db,
    )


# --------------------------------------------------------------------------------------
# live: drive one real episode through the frozen EpisodeRunner
# --------------------------------------------------------------------------------------


def _run_live(task: TaskSpec) -> _EpisodeArtifacts:
    settings = get_settings()
    registry = build_default_registry()
    assistant_backend = AnthropicBackend.for_assistant(settings)
    user_backend = AnthropicBackend.for_user_sim(settings)
    assistant = SupportAssistant(task, assistant_backend, registry=registry, settings=settings)
    user = SimulatedUser(task, user_backend, settings=settings)

    episode_id = f"live-{task.task_id}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    result = run_episode(
        task,
        assistant=assistant,
        user=user,
        registry=registry,
        parser=tool_protocol,
        episode_id=episode_id,
        settings=settings,
    )
    artifacts = _EpisodeArtifacts(
        episode_id=episode_id,
        task=task,
        trajectory=result.trajectory,
        initial_db=result.initial_db,
        final_db=result.final_db,
    )
    saved_to = _persist_run(artifacts)
    print(f"[saved run -> {saved_to}]")
    return artifacts


def _persist_run(artifacts: _EpisodeArtifacts) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{artifacts.episode_id}.json"
    payload = {
        "schema_version": "turncraft.episode.v1",
        "mode": "live" if artifacts.episode_id.startswith("live-") else "scripted-offline",
        "episode_id": artifacts.episode_id,
        "task_id": artifacts.task.task_id,
        "trajectory": artifacts.trajectory.model_dump(mode="json"),
        "initial_db": artifacts.initial_db.model_dump(mode="json"),
        "final_db": artifacts.final_db.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# replay: load a persisted run back, no model call
# --------------------------------------------------------------------------------------


def _resolve_run_path(run_arg: str) -> Path:
    direct = Path(run_arg)
    if direct.is_file():
        return direct
    fallback = RUNS_DIR / run_arg
    if fallback.is_file():
        return fallback
    raise SystemExit(f"Run file not found: {run_arg!r} (also tried {fallback})")


def _load_replay(run_arg: str, expected_task_id: str) -> _EpisodeArtifacts:
    path = _resolve_run_path(run_arg)
    payload = json.loads(path.read_text(encoding="utf-8"))

    recorded_task_id = payload.get("task_id", "")
    if recorded_task_id and recorded_task_id != expected_task_id:
        raise SystemExit(
            f"--task {expected_task_id!r} does not match the recorded run's task "
            f"{recorded_task_id!r} ({path}). Pass --task {recorded_task_id!r} to replay it."
        )
    task = get_task(expected_task_id)

    trajectory = Trajectory.model_validate(payload["trajectory"])
    initial_db = DatabaseState.model_validate(payload["initial_db"])
    final_db = DatabaseState.model_validate(payload["final_db"])
    episode_id = payload.get("episode_id", path.stem)
    return _EpisodeArtifacts(
        episode_id=episode_id, task=task, trajectory=trajectory, initial_db=initial_db, final_db=final_db
    )


# --------------------------------------------------------------------------------------
# Rendering: the one place anything is printed
# --------------------------------------------------------------------------------------


def _render_episode(artifacts: _EpisodeArtifacts) -> None:
    print(RULE)
    # 1. Task title only -- never the reward predicates. Nothing about how this task is
    #    graded is shown before the run plays out.
    print(f"TASK {artifacts.task.task_id}: {artifacts.task.title}")
    print(f"episode_id={artifacts.episode_id}")
    print(RULE)

    # 2. Compact initial database state.
    _render_state(artifacts.initial_db, heading="Initial database state")
    print(RULE)

    # 3. The conversation, streamed in order. Tool calls/results indented and marked
    #    HIDDEN FROM USER -- the trust boundary made legible.
    print("Conversation:")
    for event in artifacts.trajectory.events:
        if event.kind == "termination":
            continue
        for line in _render_event(event):
            print(line)
    print(RULE)

    # 4. Termination reason.
    _render_termination(artifacts.trajectory)
    print(RULE)

    # 5. Semantic database diff, before -> after.
    print("Database diff (before -> after):")
    diff = CustomerServiceDB.diff(artifacts.initial_db, artifacts.final_db)
    if diff.is_empty:
        print(f"{INDENT}(no changes)")
    else:
        for line in diff.lines:
            print(f"{INDENT}{line}")
    print(RULE)

    # 6. Reward breakdown by component, not just the total.
    _render_reward(artifacts)
    print(RULE)


def _render_state(state: DatabaseState, *, heading: str) -> None:
    print(f"{heading}:")
    any_rows = False
    for collection in db_module.COLLECTION_ORDER:
        rows: dict[str, Any] = getattr(state, collection)
        if not rows:
            continue
        any_rows = True
        print(f"{INDENT}{collection} ({len(rows)}):")
        for key in sorted(rows):
            print(f"{INDENT}{INDENT}{_summarise_row(collection, key, rows[key])}")
    if not any_rows:
        print(f"{INDENT}(empty)")


def _summarise_row(collection: str, key: str, row: Any) -> str:
    if collection == "users":
        return f"{key}: {row.name} <{row.email}>"
    if collection == "products":
        return (
            f"{key}: sku={row.sku} {row.name} color={row.color} size={row.size} "
            f"price={format_cents(row.unit_price_cents)}"
        )
    if collection == "inventory":
        return f"{key}: available={row.available_quantity}"
    if collection == "orders":
        return f"{key}: status={row.status.value} total={format_cents(row.total_cents)} items={len(row.item_ids)}"
    if collection == "order_items":
        return (
            f"{key}: order={row.order_id} ordered_sku={row.ordered_sku} fulfilled_sku={row.fulfilled_sku} "
            f"qty={row.quantity} unit={format_cents(row.unit_price_cents)}"
        )
    if collection == "payments":
        return (
            f"{key}: order={row.order_id} amount={format_cents(row.amount_cents)} status={row.status.value}"
        )
    if collection == "shipments":
        return (
            f"{key}: order={row.order_id} status={row.status.value} "
            f"carrier={row.carrier} tracking={row.tracking_number}"
        )
    if collection == "returns":
        return f"{key}: order={row.order_id} item={row.item_id} reason={row.reason.value} status={row.status.value}"
    if collection == "refunds":
        return (
            f"{key}: payment={row.payment_id} amount={format_cents(row.amount_cents)} "
            f"reason={row.reason.value} status={row.status.value}"
        )
    if collection == "delivery_intercepts":
        return f"{key}: order={row.order_id} shipment={row.shipment_id} status={row.status.value}"
    if collection == "notifications":
        return f"{key}: user={row.user_id} kind={row.kind.value} sku={row.sku} status={row.status.value}"
    return f"{key}: {row}"


def _format_payload(payload: dict[str, Any], *, limit: int = 300) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _render_event(event: TrajectoryEvent) -> list[str]:
    if event.kind == "message":
        text = str(event.content.get("text", ""))
        speaker = _SPEAKER_LABEL.get(event.actor, event.actor)
        if event.visible_to_user:
            return [f"{speaker}: {text}"]
        return [f"{INDENT}{HIDDEN_TAG} {speaker} (internal): {text}"]

    if event.kind == "tool_call":
        name = event.content.get("name", "?")
        args = event.content.get("arguments", {})
        severity = f" severity={event.severity.value}" if event.severity is not None else ""
        return [f"{INDENT}{HIDDEN_TAG} tool_call  {name}({_format_payload(args)}){severity}"]

    if event.kind == "tool_result":
        ok = bool(event.content.get("ok"))
        code = event.content.get("code", "")
        message = event.content.get("message", "")
        data = event.content.get("data") or {}
        status = "OK" if ok else "DENIED/FAILED"
        lines = [f"{INDENT}{HIDDEN_TAG} tool_result[{status}] {code}: {message}"]
        if data:
            lines.append(f"{INDENT}{HIDDEN_TAG}            data={_format_payload(data)}")
        return lines

    if event.kind == "invalid_tool_call":
        return [
            f"{INDENT}{HIDDEN_TAG} parse_error {event.content.get('code')}: {event.content.get('message')}"
        ]

    if event.kind == "policy_denial":
        return [
            f"{INDENT}{HIDDEN_TAG} policy_denial tool={event.content.get('name')} code={event.content.get('code')}"
        ]

    return [f"{INDENT}{HIDDEN_TAG} {event.kind}: {_format_payload(dict(event.content))}"]


def _render_termination(trajectory: Trajectory) -> None:
    detail = ""
    for event in trajectory.events:
        if event.kind == "termination":
            detail = str(event.content.get("detail", ""))
            break
    label = trajectory.termination_reason.value if trajectory.termination_reason else "unknown"
    suffix = f" -- {detail}" if detail else ""
    print(f"Termination: {label}{suffix}")


def _render_reward(artifacts: _EpisodeArtifacts) -> None:
    breakdown = compute_reward(
        artifacts.trajectory,
        artifacts.initial_db,
        artifacts.final_db,
        artifacts.task,
        weights=DEFAULT_WEIGHTS,
    )
    print("Reward breakdown (weights_version pinned in explanation):")
    print(f"{INDENT}outcome        {breakdown.outcome:+.3f}  x weight {DEFAULT_WEIGHTS.outcome:.2f}")
    print(f"{INDENT}process        {breakdown.process:+.3f}  x weight {DEFAULT_WEIGHTS.process:.2f}")
    print(
        f"{INDENT}communication  {breakdown.communication:+.3f}  x weight {DEFAULT_WEIGHTS.communication:.2f}"
    )
    print(f"{INDENT}efficiency     {breakdown.efficiency:+.3f}  x weight {DEFAULT_WEIGHTS.efficiency:.2f}")
    if breakdown.penalties:
        print(f"{INDENT}penalties:")
        for name, value in breakdown.penalties.items():
            print(f"{INDENT}{INDENT}- {name}: -{value:.3f}")
    if breakdown.fatal_violations:
        print(f"{INDENT}fatal violations: {', '.join(breakdown.fatal_violations)}")
    print(f"{INDENT}TOTAL: {breakdown.total:+.3f}")
    print(f"{INDENT}explanation:")
    for line in breakdown.explanation:
        print(f"{INDENT}{INDENT}{line}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m turncraft.demo",
        description="Turncraft: offline scripted controls or an optional two-model episode.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print all task ids and titles, then exit.")
    parser.add_argument("--task", metavar="TASK_ID", help="Task id, e.g. CANCEL-READY.")
    parser.add_argument("--mode", choices=MODES, help="How to produce the episode.")
    parser.add_argument("--run", metavar="PATH", help="Recorded run JSON path (required for --mode replay).")
    parser.add_argument("--control", help="Scripted control label, e.g. oracle_refund or null.")
    parser.add_argument("--save", action="store_true", help="Save a scripted episode for offline replay.")
    return parser


def _print_task_list() -> None:
    for task_id, title in list_tasks():
        print(f"{task_id}: {title}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        _print_task_list()
        return 0

    if not args.task or not args.mode:
        parser.error("--task and --mode are required unless --list-tasks is given")
    if args.mode == MODE_REPLAY and not args.run:
        parser.error("--mode replay requires --run <path>")
    if args.control and args.mode not in (MODE_SCRIPTED_GOOD, MODE_SCRIPTED_BAD):
        parser.error("--control is only valid in a scripted mode")

    try:
        task = get_task(args.task)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        if args.mode == MODE_SCRIPTED_GOOD:
            artifacts = _run_scripted(task, MODE_SCRIPTED_GOOD, args.control)
        elif args.mode == MODE_SCRIPTED_BAD:
            artifacts = _run_scripted(task, MODE_SCRIPTED_BAD, args.control)
        elif args.mode == MODE_LIVE:
            artifacts = _run_live(task)
        else:
            artifacts = _load_replay(args.run, args.task)
    except MissingApiKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ProviderError as exc:
        print(f"error: provider call failed: {exc}", file=sys.stderr)
        return 1
    except SystemExit:
        raise
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _render_episode(artifacts)
    if args.save and args.mode in (MODE_SCRIPTED_GOOD, MODE_SCRIPTED_BAD):
        print(f"[saved scripted control -> {_persist_run(artifacts)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
