"""Provider-agnostic LLM abstraction tests (Phase 3).

These tests exercise the multi-mode provider layer WITHOUT any network calls:
the shared HTTP transport (``http_post_json``) is monkeypatched so we can assert
on the exact request payload each provider mode builds and on how it parses the
mode-specific response schema into the uniform ``ModelCallTrace``.

They also cover the registry/factory dispatch, single-provider config files, and
API-key resolution (including fail-closed behaviour when a required key is
missing).
"""
from __future__ import annotations

import pytest

from retracemem.providers import (
    AnthropicMessagesProvider,
    OllamaChatProvider,
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderConfigError,
    get_provider,
    load_provider_config_file,
    provider_config_from_registry,
    provider_from_config,
)
from retracemem.providers import anthropic_messages as anthropic_mod
from retracemem.providers import ollama_chat as ollama_mod


# --------------------------------------------------------------------------- #
# Registry / factory dispatch (backward compatibility preserved)
# --------------------------------------------------------------------------- #
def test_registry_dispatches_each_mode_to_correct_class(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "an")
    assert isinstance(get_provider("siliconflow"), OpenAICompatibleProvider)
    assert isinstance(get_provider("anthropic"), AnthropicMessagesProvider)
    assert isinstance(get_provider("ollama"), OllamaChatProvider)


def test_unknown_provider_fails_closed():
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("nope-not-registered")


def test_mock_provider_still_resolves():
    from retracemem.providers.base import MockLLMProvider

    assert isinstance(get_provider("mock"), MockLLMProvider)


# --------------------------------------------------------------------------- #
# Config files + API-key resolution
# --------------------------------------------------------------------------- #
def test_config_file_round_trip_and_mode(tmp_path):
    cfg = load_provider_config_file("configs/providers/anthropic.yaml.example")
    assert cfg.mode == "anthropic-messages"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.extra_headers.get("anthropic-version") == "2023-06-01"


def test_resolve_api_key_missing_raises():
    cfg = ProviderConfig(name="x", mode="openai-chat", api_key_env="DEFINITELY_UNSET_KEY")
    with pytest.raises(ProviderConfigError, match="requires an API key"):
        cfg.resolve_api_key()


def test_ollama_mode_needs_no_key():
    cfg = ProviderConfig(name="ollama", mode="ollama-chat", api_key_env=None)
    assert cfg.resolve_api_key() is None
    assert isinstance(provider_from_config(cfg), OllamaChatProvider)


def test_unknown_mode_rejected():
    with pytest.raises(ProviderConfigError, match="unknown mode"):
        ProviderConfig(name="x", mode="telepathy")


def test_registry_config_carries_mode():
    cfg = provider_config_from_registry("deepseek")
    assert cfg is not None and cfg.mode == "openai-chat"
    assert provider_config_from_registry("does-not-exist") is None


# --------------------------------------------------------------------------- #
# Anthropic Messages: payload shape + response parsing (no network)
# --------------------------------------------------------------------------- #
def test_anthropic_builds_messages_payload_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, *, timeout, max_retries=0, redactor=None, **kw):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return (
            {
                "content": [{"type": "text", "text": "hello world"}],
                "usage": {"input_tokens": 11, "output_tokens": 2},
            },
            1,
        )

    monkeypatch.setattr(anthropic_mod, "http_post_json", fake_post)
    prov = AnthropicMessagesProvider(api_key="secret-key", base_url="https://api.anthropic.com/v1/messages")
    trace = prov.generate(prompt="hi", model_id="claude-3-5-sonnet-latest", provider="anthropic", temperature=0.0, max_tokens=64)

    assert trace.status == "success"
    assert trace.response == "hello world"
    assert trace.prompt_tokens == 11 and trace.completion_tokens == 2
    # Anthropic schema: top-level max_tokens + messages array; x-api-key header.
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["payload"]["max_tokens"] == 64
    assert captured["headers"]["x-api-key"] == "secret-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_missing_key_fails_closed_without_network(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("network must not be attempted without a key")

    monkeypatch.setattr(anthropic_mod, "http_post_json", boom)
    prov = AnthropicMessagesProvider(api_key=None)
    trace = prov.generate(prompt="hi", model_id="m", provider="anthropic")
    assert trace.status == "failure"
    assert "API key missing" in (trace.error_message or "")


# --------------------------------------------------------------------------- #
# Ollama: payload shape + response parsing (no network)
# --------------------------------------------------------------------------- #
def test_ollama_builds_chat_payload_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, *, timeout, max_retries=0, redactor=None, **kw):
        captured["payload"] = payload
        return (
            {
                "message": {"role": "assistant", "content": "pong"},
                "prompt_eval_count": 5,
                "eval_count": 1,
            },
            1,
        )

    monkeypatch.setattr(ollama_mod, "http_post_json", fake_post)
    prov = OllamaChatProvider(base_url="http://localhost:11434/api/chat")
    trace = prov.generate(prompt="ping", model_id="llama3.1", provider="ollama", temperature=0.0, max_tokens=32)

    assert trace.status == "success"
    assert trace.response == "pong"
    assert trace.prompt_tokens == 5 and trace.completion_tokens == 1
    # Ollama forces stream=False and nests sampling under "options".
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_predict"] == 32
