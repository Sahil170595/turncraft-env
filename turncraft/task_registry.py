"""Validated synthetic task registry with computed controls and final states."""

from turncraft.db import check_referential_integrity
from turncraft.offline import execute_plan
from turncraft.rewards import available_predicates
from turncraft.synthetic_cases import build_cases

TASKS = {}
ALL_CONTROLS = {}
FINAL_DB = {}
PLANS = {}

for task_id, (task, plans) in build_cases().items():
    if task_id in TASKS or task_id != task.task_id:
        raise ValueError("Task identifiers must be unique and consistent")
    check_referential_integrity(task.initial_db)
    for branch in task.success_branches:
        for predicate in [*branch.required_state, *branch.preserved_state]:
            if predicate.name not in available_predicates():
                raise ValueError(f"Unknown predicate {predicate.name}")
    TASKS[task_id] = task
    PLANS[task_id] = plans
    ALL_CONTROLS[task_id], FINAL_DB[task_id] = {}, {}
    for label, plan in plans.items():
        result = execute_plan(task, plan, f"synthetic-{task_id}-{label}")
        ALL_CONTROLS[task_id][label] = result.trajectory
        FINAL_DB[task_id][label] = result.final_db


def get_task(task_id):
    if task_id not in TASKS:
        raise KeyError(f"Unknown task {task_id!r}. Known: {', '.join(sorted(TASKS))}")
    return TASKS[task_id].model_copy(deep=True)


def list_tasks():
    return [(key, TASKS[key].title) for key in sorted(TASKS)]
