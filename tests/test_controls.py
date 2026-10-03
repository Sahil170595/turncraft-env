"""Fresh synthetic release controls, not inherited task fixtures."""

import pytest

from turncraft.demo import _execute_script
from turncraft.rewards import compute_reward
from turncraft.task_registry import ALL_CONTROLS, TASKS


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_control_separation_and_reproduction(task_id):
    task = TASKS[task_id]
    before = task.initial_db.model_dump(mode="json")
    scores = {}
    for label, control in ALL_CONTROLS[task_id].items():
        trace, final = _execute_script(task, control, control.episode_id)
        again, same = _execute_script(task, control, control.episode_id)
        assert trace == again and final == same
        scores[label] = compute_reward(trace, task.initial_db, final, task)
    assert task.initial_db.model_dump(mode="json") == before
    assert scores["oracle"].outcome == 1
    assert scores["oracle"].total >= 0.85
    assert scores["null"].total <= 0.2
    assert scores["near_miss"].total < scores["oracle"].total
    assert scores["forbidden"].total < scores["null"].total
    for label, score in scores.items():
        if label.startswith("oracle_"):
            assert score.outcome == 1 and score.total >= 0.85


def test_eval_manifest_covers_all_tools():
    names = {
        event.content["name"]
        for controls in ALL_CONTROLS.values()
        for trace in controls.values()
        for event in trace.events
        if event.kind == "tool_call"
    }
    assert names == {
        "get_order",
        "list_order_payments",
        "search_inventory",
        "cancel_order",
        "request_delivery_intercept",
        "create_return",
        "issue_refund",
        "create_replacement",
        "create_stock_notification",
    }
