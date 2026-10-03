"""Explicitly scripted ports that exercise the same runner as the optional model roles."""

import json
from dataclasses import dataclass

from turncraft import tool_protocol
from turncraft.config import Settings
from turncraft.models import TaskSpec, UserTurn
from turncraft.runner import run_episode
from turncraft.tools import build_default_registry


def call(name: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


@dataclass(frozen=True)
class Plan:
    customer: tuple[str, ...]
    assistant: tuple[str, ...]


class ScriptedAssistant:
    def __init__(self, replies):
        self.replies = iter(replies)

    def one_turn(self, history):
        return next(self.replies, "The scripted control has no further actions.")


class ScriptedCustomer:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.visible_replies = []

    def start(self):
        return UserTurn(message=next(self.messages))

    def respond(self, assistant_text):
        self.visible_replies.append(assistant_text)
        message = next(self.messages, None)
        return UserTurn(message=message or "End of scripted control.", done=message is None)


def execute_plan(task: TaskSpec, plan: Plan, episode_id: str):
    return run_episode(
        task,
        assistant=ScriptedAssistant(plan.assistant),
        user=ScriptedCustomer(plan.customer),
        registry=build_default_registry(),
        parser=tool_protocol,
        episode_id=episode_id,
        settings=Settings(),
        sleep=lambda seconds: None,
    )
