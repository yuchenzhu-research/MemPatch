"""Tests for SiliconFlowProvider transient-error retry / graceful degradation."""

from __future__ import annotations

import json
import sys
import types

from benchmark.retrace_bench.llm_providers.siliconflow_provider import SiliconFlowProvider


class _Msg:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


def _install_fake_openai(monkeypatch, behaviors):
    """Install a fake ``openai`` module whose create() pops from ``behaviors``.

    Each behavior is either an Exception (raised) or a string (returned).
    """
    calls = {"n": 0}

    class _Completions:
        def create(self, **kwargs):
            item = behaviors[calls["n"]]
            calls["n"] += 1
            if isinstance(item, Exception):
                raise item
            return _Resp(item)

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)
    return calls


def test_retries_then_succeeds(monkeypatch):
    calls = _install_fake_openai(monkeypatch, [RuntimeError("alb 400"), RuntimeError("alb 400"), '{"answer": "ok"}'])
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real backoff delay
    provider = SiliconFlowProvider(api_key="k", model="deepseek-ai/DeepSeek-V4-Flash")
    out = provider.generate("hello")
    assert out == '{"answer": "ok"}'
    assert calls["n"] == 3


def test_persistent_failure_returns_error_json(monkeypatch):
    _install_fake_openai(monkeypatch, [RuntimeError("alb 400")] * 10)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = SiliconFlowProvider(api_key="k")
    out = provider.generate("hello")
    payload = json.loads(out)
    assert "error" in payload  # degrades instead of raising -> run continues
    assert "RuntimeError" in payload["error"]


def test_missing_key_short_circuits(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    provider = SiliconFlowProvider(api_key=None)
    payload = json.loads(provider.generate("hello"))
    assert payload["error"] == "SILICONFLOW_API_KEY is not set"
