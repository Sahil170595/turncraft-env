import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from turncraft import llm
from turncraft.config import MissingApiKeyError, Settings

PACKAGE = Path(__file__).resolve().parents[1] / "turncraft"


def test_runtime_dependency_and_sdk_boundary():
    sdk_importers = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in sys.stdlib_module_names | {"turncraft", "anthropic", "pydantic"}
                if root == "anthropic":
                    sdk_importers.append(path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not any(keyword.arg == "tools" for keyword in node.keywords)
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"tool_runner", "beta_tool"}
    assert sdk_importers == ["llm.py"]


def test_live_requires_credentials_and_explicit_model_without_network():
    with pytest.raises(MissingApiKeyError):
        llm.AnthropicBackend(model="offline-test-model", settings=Settings())
    with pytest.raises(ValueError, match="Configure"):
        llm.AnthropicBackend(model="", settings=Settings(anthropic_api_key="offline-placeholder"))


def test_provider_payload_uses_our_protocol_not_native_tools(monkeypatch):
    constructor_args, requests = [], []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="offline response")],
            usage=None,
            model="offline-test-model",
            stop_reason="end_turn",
        )

    def fake_client(**kwargs):
        constructor_args.append(kwargs)
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(llm.anthropic, "Anthropic", fake_client)
    backend = llm.AnthropicBackend(
        model="offline-test-model", settings=Settings(anthropic_api_key="offline-placeholder"), max_retries=9
    )
    turn = backend.complete(
        system="Synthetic policy",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=50,
        stop_sequences=["</tool_call>"],
    )
    assert turn.text == "offline response"
    assert constructor_args[0]["max_retries"] == 0 and backend._max_retries == 0
    assert requests[0]["stop_sequences"] == ["</tool_call>"]
    assert not {"tools", "temperature", "top_p", "top_k"} & requests[0].keys()


def test_provider_failure_does_not_retry_in_adapter(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        raise llm.anthropic.APIError("Synthetic adapter failure", request=SimpleNamespace(), body=None)

    monkeypatch.setattr(
        llm.anthropic, "Anthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    backend = llm.AnthropicBackend(
        model="offline-test-model", settings=Settings(anthropic_api_key="offline-placeholder")
    )
    with pytest.raises(llm.ProviderError):
        backend.complete(system="policy", messages=[{"role": "user", "content": "hello"}], max_tokens=50)
    assert len(requests) == 1
