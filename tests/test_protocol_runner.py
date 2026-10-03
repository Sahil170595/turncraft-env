import json

import pytest

from turncraft import tool_protocol
from turncraft.agents import SimulatedUser, SupportAssistant
from turncraft.config import Settings
from turncraft.llm import ChatTurn
from turncraft.models import TerminationReason
from turncraft.offline import ScriptedAssistant, ScriptedCustomer, call
from turncraft.runner import run_episode
from turncraft.task_registry import get_task
from turncraft.tools import build_default_registry


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        return reply if isinstance(reply, ChatTurn) else ChatTurn(text=reply, stop_reason="end_turn")


@pytest.mark.parametrize(
    "raw,code",
    [
        ("<tool_call>{bad}</tool_call>", "malformed_json"),
        ('<tool_call>{"name":"get_order","arguments":{}}', "unclosed_tag"),
        (call("absent_tool"), "unknown_tool"),
        ('<tool_call>{"name":"get_order","arguments":[]}</tool_call>', "arguments_not_object"),
        ("<tool_call></tool_call>", "empty_tool_call"),
    ],
)
def test_parser_failures_are_values_and_not_customer_markup(raw, code):
    result = tool_protocol.parse(raw, ["get_order"])
    assert not result.calls and code in {error.code for error in result.errors}
    assert "<tool_call>" not in result.visible_text


def test_parser_nested_strings_fences_multiple_calls_and_recovery():
    valid = call("cancel_order", order_id="ORD-SYN-81", reason='Brace } and quoted "text"')
    text = (
        "An example:\n```json\n"
        + valid
        + "\n```\nActual action:\n"
        + valid
        + call("get_order", order_id="ORD-SYN-81")
    )
    result = tool_protocol.parse(text, ["cancel_order", "get_order"])
    assert [c.name for c in result.calls] == ["cancel_order", "get_order"]
    assert result.calls[0].raw_arguments["reason"] == 'Brace } and quoted "text"'
    assert "<tool_call>" not in result.visible_text
    assert result == tool_protocol.parse(text, ["cancel_order", "get_order"])
    mixed = tool_protocol.parse("<tool_call>{bad}</tool_call>" + valid, ["cancel_order"])
    assert len(mixed.errors) == 1 and len(mixed.calls) == 1


def test_stop_sequence_repair_does_not_guess_truncated_calls():
    raw = call("get_order", order_id="ORD-SYN-81")[: -len("</tool_call>")]
    assert tool_protocol.reattach_stop_sequence(raw, stop_reason="stop_sequence").endswith("</tool_call>")
    assert tool_protocol.reattach_stop_sequence(raw, stop_reason="max_tokens") == raw


def run(task, assistant, customer, **kwargs):
    return run_episode(
        task,
        assistant=assistant,
        user=customer,
        registry=build_default_registry(),
        parser=tool_protocol,
        settings=Settings(),
        sleep=lambda _: None,
        **kwargs,
    )


def test_real_role_classes_preserve_hidden_goal_and_tool_context_boundaries():
    task = get_task("CANCEL-READY")
    task.hidden_user_goal = "Private objective marker: customer prefers a quiet resolution."
    a_backend = ScriptedBackend(
        [
            call("get_order", order_id="ORD-SYN-81"),
            call("cancel_order", order_id="ORD-SYN-81", reason="Customer request"),
            "Order cancelled.",
        ]
    )
    u_backend = ScriptedBackend(
        [
            json.dumps({"message": "Please cancel ORD-SYN-81.", "done": False, "reason": "request"}),
            json.dumps({"message": "Thanks.", "done": True, "reason": "done"}),
        ]
    )
    assistant = SupportAssistant(task, a_backend, registry=build_default_registry(), settings=Settings())
    customer = SimulatedUser(task, u_backend, settings=Settings())
    result = run(task, assistant, customer)
    assert result.trajectory.termination_reason == TerminationReason.USER_DONE
    assert result.final_db.orders["ORD-SYN-81"].status == "cancelled"
    assert "Private objective marker" not in assistant.system_prompt
    assert "Private objective marker" in customer.system_prompt
    assert "<tool_call>" not in customer.audit_text()
    assert "<tool_result>" not in customer.audit_text()
    assert "synthetic-receipt" not in customer.audit_text()
    assert "private_audit" not in str(a_backend.requests)
    assert "<tool_result>" in str(a_backend.requests)
    assert a_backend.requests[0]["stop_sequences"] == ["</tool_call>"]
    assert u_backend.requests[0]["output_schema"]


@pytest.mark.parametrize(
    "replies,reason,attempts",
    [
        (["bad", "bad", "bad"], TerminationReason.INVALID_USER_OUTPUT, 3),
        (["bad", '{"message":"Done","done":true,"reason":"finished"}'], TerminationReason.USER_DONE, 2),
    ],
)
def test_user_json_repair_is_bounded_and_not_success_by_default(replies, reason, attempts):
    task = get_task("CANCEL-READY")
    backend = ScriptedBackend(replies)
    customer = SimulatedUser(task, backend, settings=Settings())
    result = run(task, ScriptedAssistant([]), customer)
    assert result.trajectory.termination_reason == reason
    assert len(backend.requests) == attempts
    assert result.final_db == result.initial_db


def test_transport_retry_budget_is_owned_once_by_runner():
    task = get_task("CANCEL-READY")

    class Failure:
        calls = 0

        def one_turn(self, history):
            self.calls += 1
            raise RuntimeError("Synthetic transport failure")

    assistant = Failure()
    result = run(task, assistant, ScriptedCustomer(["Please cancel ORD-SYN-81."]))
    assert assistant.calls == 3
    assert result.trajectory.termination_reason == TerminationReason.PROVIDER_ERROR
    assert result.final_db == result.initial_db


def test_round_progress_accepts_duplicate_read_plus_new_write():
    task = get_task("CANCEL-READY")
    read = call("get_order", order_id="ORD-SYN-81")
    assistant = ScriptedAssistant(
        [read, read + call("cancel_order", order_id="ORD-SYN-81", reason="request"), "Order cancelled."]
    )
    result = run(task, assistant, ScriptedCustomer(["Please cancel ORD-SYN-81."]))
    assert result.trajectory.termination_reason == TerminationReason.USER_DONE
    assert result.final_db.orders["ORD-SYN-81"].status == "cancelled"


def test_duplicate_read_loop_stops():
    task = get_task("CANCEL-READY")
    read = call("get_order", order_id="ORD-SYN-81")
    result = run(task, ScriptedAssistant([read, read, read]), ScriptedCustomer(["Please help."]))
    assert result.trajectory.termination_reason == TerminationReason.REPEATED_NO_PROGRESS


def test_parser_recovery_respects_tool_budget_and_hides_prose():
    task = get_task("CANCEL-READY")
    task.max_tool_rounds_per_turn = 2
    result = run(
        task, ScriptedAssistant(["<tool_call>{bad}</tool_call>"] * 3), ScriptedCustomer(["Please help."])
    )
    assert result.trajectory.termination_reason == TerminationReason.MAX_TOOL_ROUNDS
    assert len([e for e in result.trajectory.events if e.kind == "invalid_tool_call"]) == 2
