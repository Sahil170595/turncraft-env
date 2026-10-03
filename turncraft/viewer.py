"""Self-contained HTML trajectory and reward report. No server is required."""

from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from pathlib import Path
from typing import Any

from turncraft.models import DatabaseState, TaskSpec, Trajectory
from turncraft.reward_weights import DEFAULT_WEIGHTS
from turncraft.rewards import compute_reward

CONTROL_LABELS = ("oracle", "near_miss", "null", "forbidden")

# Score bands, mirrored from reward_weights so the colour matches the gate that enforces it.
_BAND_GOOD = DEFAULT_WEIGHTS.oracle_band_floor
_BAND_WEAK = DEFAULT_WEIGHTS.null_band_ceiling


def _load_registry() -> tuple[dict[str, TaskSpec], dict[str, Any], dict[str, Any]]:
    from turncraft.task_registry import ALL_CONTROLS, FINAL_DB, TASKS

    return TASKS, ALL_CONTROLS, FINAL_DB


# --------------------------------------------------------------------------- semantic diff


def _rows(state: DatabaseState) -> dict[str, dict[str, Any]]:
    """Collection name -> {row id -> plain dict}, for comparison."""
    dumped = state.model_dump(mode="json")
    return {k: v for k, v in dumped.items() if isinstance(v, dict)}


def _diff_lines(before: DatabaseState, after: DatabaseState) -> list[tuple[str, str]]:
    """(kind, text) pairs. kind is 'add' | 'del' | 'chg'. Empty list means nothing moved."""
    out: list[tuple[str, str]] = []
    b, a = _rows(before), _rows(after)
    for collection in sorted(set(b) | set(a)):
        rows_b, rows_a = b.get(collection, {}), a.get(collection, {})
        for row_id in sorted(set(rows_b) | set(rows_a)):
            old, new = rows_b.get(row_id), rows_a.get(row_id)
            if old is None:
                out.append(("add", f"{collection}: + {row_id}  {_compact(new)}"))
            elif new is None:
                out.append(("del", f"{collection}: - {row_id}"))
            elif old != new:
                for field in sorted(set(old) | set(new)):
                    if old.get(field) != new.get(field):
                        out.append(
                            (
                                "chg",
                                f"{collection}: {row_id}.{field}: {old.get(field)!r} -> {new.get(field)!r}",
                            )
                        )
    return out


def _compact(row: Any, limit: int = 4) -> str:
    if not isinstance(row, dict):
        return str(row)
    keep = [f"{k}={v}" for k, v in list(row.items())[:limit] if k != "id"]
    return ", ".join(keep)


def _money(cents: Any) -> str:
    return f"${int(cents) / 100:,.2f}" if isinstance(cents, int) else str(cents)


# --------------------------------------------------------------------------- html pieces


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _event_html(event: Any) -> str:
    kind = str(getattr(event, "kind", ""))
    actor = str(getattr(event, "actor", ""))
    visible = bool(getattr(event, "visible_to_user", False))
    content = getattr(event, "content", {}) or {}
    severity = getattr(event, "severity", None)
    sev = f'<span class="sev">{_esc(severity)}</span>' if severity else ""

    if kind == "message" and visible:
        who = "Customer" if actor == "user" else "Assistant"
        return (
            f'<div class="turn {actor}"><div class="who">{who}</div>'
            f'<div class="bubble">{_esc(content.get("text", ""))}</div></div>'
        )

    if kind == "message":  # privileged assistant prose
        return (
            f'<div class="hidden-row"><span class="tag">HIDDEN</span>'
            f'<span class="mono">assistant note: {_esc(content.get("text", ""))}</span></div>'
        )

    if kind == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in (content.get("arguments") or {}).items())
        return (
            f'<div class="hidden-row"><span class="tag">HIDDEN</span>'
            f'<span class="mono call">-> {_esc(content.get("name"))}({_esc(args)})</span>{sev}</div>'
        )

    if kind == "tool_result":
        ok = content.get("ok")
        cls = "ok" if ok else "err"
        mark = "OK" if ok else "DENIED"
        return (
            f'<div class="hidden-row"><span class="tag">HIDDEN</span>'
            f'<span class="mono {cls}">&lt;- {mark} {_esc(content.get("code", ""))} '
            f"{_esc(content.get('message', ''))}</span></div>"
        )

    if kind == "policy_denial":
        return (
            f'<div class="hidden-row"><span class="tag">HIDDEN</span>'
            f'<span class="mono err">!! policy denied {_esc(content.get("name"))}</span></div>'
        )

    if kind == "termination":
        return (
            f'<div class="term">terminated: {_esc(content.get("reason"))}'
            f" &mdash; {_esc(content.get('detail', ''))}</div>"
        )
    return ""


def _bar(label: str, value: float, weight: float | None = None) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    w = f'<span class="wt">x{weight:.2f}</span>' if weight is not None else ""
    return (
        f'<div class="metric"><div class="mlabel">{_esc(label)}{w}'
        f'<span class="mval">{value:.3f}</span></div>'
        f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div></div>'
    )


def _panel(task: TaskSpec, control: str, trajectory: Trajectory, final_db: DatabaseState) -> str:
    breakdown = compute_reward(
        trajectory, task.initial_db.model_copy(deep=True), final_db, task, weights=DEFAULT_WEIGHTS
    )
    total = float(breakdown.total)
    band = "good" if total >= _BAND_GOOD else ("weak" if total <= _BAND_WEAK else "mid")

    events = "".join(_event_html(e) for e in trajectory.events)

    diff = _diff_lines(task.initial_db, final_db)
    if diff:
        diff_html = "".join(f'<div class="d {k}">{_esc(t)}</div>' for k, t in diff)
    else:
        diff_html = (
            '<div class="d none">no state change &mdash; for this task that may be exactly right</div>'
        )

    bars = (
        _bar("outcome", breakdown.outcome, DEFAULT_WEIGHTS.outcome)
        + _bar("process", breakdown.process, DEFAULT_WEIGHTS.process)
        + _bar("communication", breakdown.communication, DEFAULT_WEIGHTS.communication)
        + _bar("efficiency", breakdown.efficiency, DEFAULT_WEIGHTS.efficiency)
    )

    pen = getattr(breakdown, "penalties", {}) or {}
    pen_html = (
        "".join(f'<div class="pen">-{v:.3f} {_esc(k)}</div>' for k, v in sorted(pen.items()))
        or '<div class="pen none">none</div>'
    )
    fatal = list(getattr(breakdown, "fatal_violations", []) or [])
    fatal_html = (
        "".join(f'<div class="fatal">{_esc(f)}</div>' for f in fatal) or '<div class="pen none">none</div>'
    )
    notes = "".join(f"<li>{_esc(x)}</li>" for x in (getattr(breakdown, "explanation", []) or []))

    return f"""
<section class="panel" id="{_esc(task.task_id)}-{_esc(control)}">
  <header class="phead">
    <div><span class="tid">{_esc(task.task_id)}</span> {_esc(task.title)}</div>
    <div class="ctl">{_esc(control)} <span class="score {band}">{total:+.3f}</span></div>
  </header>
  <div class="cols">
    <div class="col conv">
      <h3>Conversation <small>&mdash; inset rows were never seen by the customer</small></h3>
      {events}
    </div>
    <div class="col side">
      <h3>Database diff</h3><div class="diff">{diff_html}</div>
      <h3>Reward</h3>{bars}
      <h3>Deductions</h3>{pen_html}
      <h3>Fatal gates</h3>{fatal_html}
      <h3>Why</h3><ul class="notes">{notes}</ul>
    </div>
  </div>
</section>"""


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#11131a;color:#d8dae3;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:19px;margin:0 0 4px}
h3{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#7d8296;margin:20px 0 8px;font-weight:600}
h3 small{text-transform:none;letter-spacing:0;color:#5d6274;font-weight:400}
.wrap{max-width:1360px;margin:0 auto;padding:26px}
.sub{color:#7d8296;margin:0 0 22px;font-size:13px}
.panel{background:#171a23;border:1px solid #252a37;border-radius:10px;margin-bottom:22px;overflow:hidden}
.phead{display:flex;justify-content:space-between;align-items:center;gap:16px;
       padding:13px 18px;background:#1c202b;border-bottom:1px solid #252a37}
.tid{display:inline-block;background:#2b3142;color:#aab0c4;border-radius:4px;
     padding:1px 7px;margin-right:8px;font-weight:600;font-size:12px}
.ctl{color:#7d8296;font-size:12px;text-transform:uppercase;letter-spacing:.07em}
.score{font-size:17px;font-weight:700;margin-left:10px;font-variant-numeric:tabular-nums}
.score.good{color:#4ec98a}.score.mid{color:#e0b341}.score.weak{color:#e0655f}
.cols{display:grid;grid-template-columns:1fr 400px;gap:26px;padding:18px}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}
.turn{margin:12px 0}
.turn .who{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#6b7089;margin-bottom:3px}
.bubble{background:#212636;border-radius:8px;padding:10px 13px;white-space:pre-wrap}
.turn.assistant .bubble{background:#1b2a2b;border-left:2px solid #3f7f6a}
.hidden-row{display:flex;gap:9px;align-items:baseline;margin:3px 0 3px 26px;padding:4px 10px;
            background:#141721;border-left:2px solid #3a4055;border-radius:0 4px 4px 0}
.tag{font-size:9px;letter-spacing:.1em;color:#5d6274;border:1px solid #2f3547;
     border-radius:3px;padding:0 4px;flex:none}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#8f95ab;word-break:break-word}
.mono.call{color:#8ea9d8}.mono.ok{color:#5f9e7d}.mono.err{color:#c9736d}
.sev{margin-left:auto;font-size:10px;color:#6b7089;flex:none}
.term{margin:14px 0 0;padding-top:10px;border-top:1px dashed #2a3040;color:#6b7089;font-size:12px}
.diff{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.d{padding:3px 8px;border-radius:4px;margin-bottom:3px;word-break:break-word}
.d.add{background:#152a1e;color:#6fca97}.d.del{background:#2a1618;color:#d9807a}
.d.chg{background:#1d2333;color:#93a4cc}.d.none{color:#5d6274;background:#141721}
.metric{margin-bottom:11px}
.mlabel{display:flex;align-items:baseline;gap:7px;font-size:12px;color:#9aa0b4;margin-bottom:4px}
.wt{font-size:10px;color:#5d6274}
.mval{margin-left:auto;font-variant-numeric:tabular-nums;color:#d8dae3}
.track{height:5px;background:#232838;border-radius:3px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,#3f7f6a,#4ec98a)}
.pen{font-size:12px;color:#d9807a;padding:2px 0}
.pen.none{color:#5d6274}
.fatal{font-size:12px;color:#fff;background:#7d2b2b;border-radius:4px;padding:3px 8px;margin-bottom:3px}
.notes{margin:0;padding-left:17px;color:#7d8296;font-size:12px}
.notes li{margin-bottom:3px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:#6b7089;font-size:12px;margin-bottom:20px}
"""


def render(panels: list[str], title: str) -> str:
    return f"""<!doctype html>
<meta charset="utf-8"><title>{_esc(title)}</title><style>{_CSS}</style>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <p class="sub">Multi-agent customer service simulation &mdash; trajectory, world diff, and reward breakdown.</p>
  <div class="legend">
    <span>Full-width bubbles = what the customer saw</span>
    <span>Inset HIDDEN rows = privileged tool traffic they never saw</span>
    <span>Scores: green &ge; {_BAND_GOOD:.2f} &middot; red &le; {_BAND_WEAK:.2f}</span>
  </div>
  {"".join(panels)}
</div>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="viewer", description="Render episodes as a self-contained HTML report."
    )
    ap.add_argument("--task", help="Task id, e.g. CANCEL-READY")
    ap.add_argument("--control", default="oracle", choices=CONTROL_LABELS)
    ap.add_argument("--all", action="store_true", help="Every task, oracle vs near_miss vs forbidden")
    ap.add_argument("--out", default="turncraft_report.html")
    ap.add_argument("--open", action="store_true", help="Open in the default browser when done")
    args = ap.parse_args(argv)

    tasks, controls, final_db = _load_registry()

    pairs: list[tuple[str, str]] = []
    if args.all:
        for tid in sorted(tasks):
            pairs += [(tid, "oracle"), (tid, "near_miss"), (tid, "forbidden")]
        title = "Turncraft synthetic controls: oracle vs near-miss vs forbidden"
    elif args.task:
        if args.task not in tasks:
            print(f"Unknown task {args.task!r}. Known: {sorted(tasks)}", file=sys.stderr)
            return 1
        pairs = [(args.task, args.control)]
        title = f"{args.task} — {args.control}"
    else:
        for label in ("oracle", "near_miss", "forbidden"):
            pairs.append(("CANCEL-READY", label))
        title = "CANCEL-READY: oracle vs near-miss vs forbidden"

    panels = [_panel(tasks[t], c, controls[t][c], final_db[t][c]) for t, c in pairs]
    out = Path(args.out).resolve()
    out.write_text(render(panels, title), encoding="utf-8")
    print(f"wrote {out}  ({len(panels)} panel(s))")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
