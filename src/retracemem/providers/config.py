"""Provider-agnostic LLM configuration.

A :class:`ProviderConfig` captures everything the evaluation runner needs to
talk to *any* standard LLM API style without hard-coding provider-specific
request logic. The same config shape covers:

* ``openai-chat``              - OpenAI ``/v1/chat/completions`` and any
                                  OpenAI-compatible server (SiliconFlow,
                                  DeepSeek, vLLM, SGLang, LM Studio, ...).
* ``custom-openai-compatible`` - alias of ``openai-chat`` for self-hosted
                                  OpenAI-compatible endpoints (kept distinct so
                                  configs can document intent).
* ``anthropic-messages``       - Anthropic ``/v1/messages`` (Claude schema).
* ``ollama-chat``              - Ollama native ``/api/chat`` (local, no auth).

Secrets are *never* stored in config; only the *name* of the environment
variable holding the key (``api_key_env``) is. Resolution happens at call time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

OPENAI_CHAT = "openai-chat"
CUSTOM_OPENAI_COMPATIBLE = "custom-openai-compatible"
ANTHROPIC_MESSAGES = "anthropic-messages"
OLLAMA_CHAT = "ollama-chat"

VALID_MODES = (
    OPENAI_CHAT,
    CUSTOM_OPENAI_COMPATIBLE,
    ANTHROPIC_MESSAGES,
    OLLAMA_CHAT,
)

# Modes that resolve to the OpenAI-compatible chat-completions request shape.
OPENAI_FAMILY_MODES = (OPENAI_CHAT, CUSTOM_OPENAI_COMPATIBLE)

# Modes that do not require an API key (local servers).
_NO_AUTH_MODES = (OLLAMA_CHAT,)

# Legacy ``provider_type`` (configs/providers.yaml registry) -> mode.
_LEGACY_PROVIDER_TYPE_TO_MODE = {
    "openai_compatible": OPENAI_CHAT,
    "anthropic_messages": ANTHROPIC_MESSAGES,
    "ollama_chat": OLLAMA_CHAT,
}


class ProviderConfigError(ValueError):
    """Raised when a provider config is malformed or missing a required field."""


@dataclass(frozen=True)
class ProviderConfig:
    """Provider-agnostic LLM call configuration.

    ``base_url`` is the *full* endpoint URL (including the path, e.g.
    ``https://api.siliconflow.cn/v1/chat/completions``) so a single field works
    across all provider modes without mode-specific path assembly.
    """

    name: str
    mode: str = OPENAI_CHAT
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    timeout: float = 60.0
    max_retries: int = 0
    temperature: float | None = 0.0
    max_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    reasoning: bool = False
    stream: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ProviderConfigError(
                f"provider '{self.name}': unknown mode '{self.mode}'. "
                f"Valid modes: {', '.join(VALID_MODES)}."
            )

    @property
    def requires_api_key(self) -> bool:
        return self.mode not in _NO_AUTH_MODES

    def resolve_api_key(self, explicit: str | None = None) -> str | None:
        """Resolve the API key from an explicit value or ``api_key_env``.

        Raises :class:`ProviderConfigError` only when a key is *required* by the
        mode and neither an explicit value nor a populated ``api_key_env`` is
        available. Local modes (e.g. Ollama) return ``None`` happily.
        """
        if explicit:
            return explicit
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        if self.requires_api_key:
            hint = self.api_key_env or "<api_key_env not set in config>"
            raise ProviderConfigError(
                f"provider '{self.name}' (mode '{self.mode}') requires an API key, "
                f"but environment variable '{hint}' is not set. "
                f"Set it (e.g. in a .env file) or pass an explicit api_key."
            )
        return None

    def with_overrides(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "ProviderConfig":
        """Return a copy with non-None CLI overrides applied."""
        updates: dict[str, Any] = {}
        if model is not None:
            updates["model"] = model
        if base_url is not None:
            updates["base_url"] = base_url
        return replace(self, **updates) if updates else self


def _coerce(data: dict[str, Any], name: str) -> ProviderConfig:
    mode = data.get("mode")
    if mode is None:
        legacy = data.get("provider_type")
        mode = _LEGACY_PROVIDER_TYPE_TO_MODE.get(legacy, OPENAI_CHAT)
    base_url = data.get("base_url") or data.get("default_base_url")
    extra_headers = data.get("extra_headers") or {}
    if not isinstance(extra_headers, dict):
        raise ProviderConfigError(f"provider '{name}': extra_headers must be a mapping.")
    return ProviderConfig(
        name=str(data.get("name", name)),
        mode=str(mode),
        base_url=base_url,
        api_key_env=data.get("api_key_env"),
        model=data.get("model"),
        timeout=float(data.get("timeout", 60.0)),
        max_retries=int(data.get("max_retries", 0)),
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens"),
        extra_headers={str(k): str(v) for k, v in extra_headers.items()},
        reasoning=bool(data.get("reasoning", False)),
        stream=bool(data.get("stream", False)),
    )


def load_provider_config_file(path: str | Path) -> ProviderConfig:
    """Load a single-provider YAML/JSON config file.

    The file is a flat mapping of :class:`ProviderConfig` fields, e.g.::

        name: siliconflow
        mode: openai-chat
        base_url: https://api.siliconflow.cn/v1/chat/completions
        api_key_env: SILICONFLOW_API_KEY
        model: deepseek-ai/DeepSeek-V3
    """
    p = Path(path)
    if not p.exists():
        raise ProviderConfigError(f"provider config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ProviderConfigError(f"provider config file must be a mapping: {p}")
    # Tolerate a single-key {providers: {name: {...}}} wrapper too.
    if "providers" in data and isinstance(data["providers"], dict) and "mode" not in data:
        providers = data["providers"]
        if len(providers) != 1:
            raise ProviderConfigError(
                f"{p}: a single-provider config file must define exactly one provider."
            )
        (only_name, only_cfg), = providers.items()
        return _coerce(dict(only_cfg), str(only_name))
    return _coerce(data, p.stem)


def provider_config_from_registry(name: str) -> ProviderConfig | None:
    """Build a :class:`ProviderConfig` from the ``configs/providers.yaml`` registry."""
    from retracemem.providers.provider_factory import load_providers_config

    registry = load_providers_config()
    key = name.lower()
    if key not in registry:
        return None
    return _coerce(dict(registry[key]), key)
