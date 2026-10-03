"""Network-free control sweep with versioned, reproducible state and reward artifacts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from turncraft import __version__
from turncraft.demo import _execute_script
from turncraft.reward_weights import DEFAULT_WEIGHTS, WEIGHTS_VERSION
from turncraft.rewards import compute_reward
from turncraft.task_registry import ALL_CONTROLS, TASKS


def evaluate():
    rows = []
    failures = []
    for task_id in sorted(TASKS):
        task = TASKS[task_id]
        scores = {}
        for label, script in ALL_CONTROLS[task_id].items():
            trace, final = _execute_script(task, script, script.episode_id)
            reward = compute_reward(trace, task.initial_db, final, task)
            scores[label] = reward
            rows.append(
                {
                    "task_id": task_id,
                    "control": label,
                    "reward": reward.model_dump(mode="json"),
                    "trajectory": trace.model_dump(mode="json"),
                    "initial_db": task.initial_db.model_dump(mode="json"),
                    "final_db": final.model_dump(mode="json"),
                }
            )
        passed = (
            scores["oracle"].outcome == 1
            and scores["oracle"].total >= DEFAULT_WEIGHTS.oracle_band_floor
            and scores["null"].total <= DEFAULT_WEIGHTS.null_band_ceiling
            and scores["near_miss"].total < scores["oracle"].total
            and scores["forbidden"].total < scores["null"].total
            and all(
                score.outcome == 1 and score.total >= DEFAULT_WEIGHTS.oracle_band_floor
                for label, score in scores.items()
                if label.startswith("oracle_")
            )
        )
        if not passed:
            failures.append(task_id)
    return {
        "schema_version": "turncraft.control-evaluation.v1",
        "package_version": __version__,
        "fixture_version": "synthetic.v1",
        "weights_version": WEIGHTS_VERSION,
        "weights": asdict(DEFAULT_WEIGHTS),
        "mode": "scripted-offline",
        "seed": None,
        "configuration": {
            "tools": 9,
            "tasks": len(TASKS),
            "controls": len(rows),
            "dialogue_budget": 8,
            "tool_round_budget": 6,
            "randomness": "none; fixed authored cases and scripts",
        },
        "failed_tasks": failures,
        "results": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write a versioned synthetic evaluation JSON artifact")
    args = parser.parse_args(argv)
    report = evaluate()
    print("Task                 Control           Outcome   Total")
    for row in report["results"]:
        reward = row["reward"]
        print(f"{row['task_id']:<20} {row['control']:<17} {reward['outcome']:>6.3f} {reward['total']:>7.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved synthetic evaluation: {args.out}")
    print(
        f"{report['configuration']['tasks']} cases; {report['configuration']['controls']} controls; {len(report['failed_tasks'])} failed task gates"
    )
    return 1 if report["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
