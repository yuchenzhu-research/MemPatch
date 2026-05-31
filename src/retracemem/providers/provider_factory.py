from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from retracemem.providers.base import BaseLLMProvider, MockLLMProvider
from retracemem.providers.openai_compatible import OpenAICompatibleProvider

def load_providers_config() -> dict[str, Any]:
    """Loads configs/providers.yaml from repo root."""
    # Find repository root. We can traverse up from this file's location.
    current_file = Path(__file__).resolve()
    # Path is src/retracemem/providers/provider_factory.py, repo root is 4 parents up
    repo_root = current_file.parents[3]
    config_path = repo_root / "configs" / "providers.yaml"
    
    if not config_path.exists():
        return {}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "providers" in data:
                return data["providers"]
    except Exception:
        pass
    return {}


def _build_provider_for_mode(
    mode: str,
    *,
    api_key: str | None,
    base_url: str | None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Construct the concrete provider class for a resolved ``mode``.

    Keeps the evaluation runner agnostic to provider style: it only ever sees a
    :class:`BaseLLMProvider`. New API styles are added here, not in the runner.
    """
    # Imported lazily to avoid import cost when only OpenAI-compatible is used.
    from retracemem.providers.config import (
        ANTHROPIC_MESSAGES,
        OLLAMA_CHAT,
        OPENAI_FAMILY_MODES,
    )
    from retracemem.providers.anthropic_messages import AnthropicMessagesProvider
    from retracemem.providers.ollama_chat import OllamaChatProvider

    headers = dict(extra_headers or {})
    if mode == ANTHROPIC_MESSAGES:
        anthropic_version = headers.pop("anthropic-version", None)
        kw: dict[str, Any] = dict(
            api_key=api_key,
            base_url=base_url,
            extra_headers=headers,
            max_retries=max_retries,
        )
        if timeout is not None:
            kw["timeout"] = timeout
        if anthropic_version is not None:
            kw["anthropic_version"] = anthropic_version
        return AnthropicMessagesProvider(**kw)
    if mode == OLLAMA_CHAT:
        kw = dict(
            api_key=api_key,
            base_url=base_url,
            extra_headers=headers,
            max_retries=max_retries,
        )
        if timeout is not None:
            kw["timeout"] = timeout
        return OllamaChatProvider(**kw)
    # Default: OpenAI-compatible chat completions (openai-chat / custom-openai-compatible).
    if mode not in OPENAI_FAMILY_MODES:
        # Unknown/legacy value still defaults to OpenAI-compatible for safety.
        pass
    kw = dict(api_key=api_key, base_url=base_url, extra_headers=headers, max_retries=max_retries)
    if timeout is not None:
        kw["timeout"] = timeout
    kw.update(kwargs)
    return OpenAICompatibleProvider(**kw)


def get_provider(
    provider_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """
    Factory to retrieve an LLM provider instance.
    Loads provider metadata dynamically from configs/providers.yaml.
    Resolves the provider *mode* (openai-chat / anthropic-messages / ollama-chat /
    custom-openai-compatible) and constructs the matching concrete provider, so
    the evaluation runner never hard-codes provider-specific request logic.
    Fails closed if the provider is unknown (unless 'mock').
    """
    provider_name_lower = provider_name.lower()
    if provider_name_lower == "mock":
        return MockLLMProvider(**kwargs)
        
    config = load_providers_config()
    if provider_name_lower not in config:
        raise ValueError(f"Unsupported provider: '{provider_name}'. Must be registered in configs/providers.yaml.")
        
    prov_meta = config[provider_name_lower]
    api_key_env = prov_meta.get("api_key_env")
    resolved_api_key = api_key
    
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.environ.get(api_key_env)
        
    # If key is still missing, but we are running in live/non-mock mode, we will let the provider raise ValueError on generate
    resolved_base_url = base_url or prov_meta.get("default_base_url")

    # Resolve the request mode (new 'mode' field, else legacy 'provider_type').
    from retracemem.providers.config import _LEGACY_PROVIDER_TYPE_TO_MODE, OPENAI_CHAT

    mode = prov_meta.get("mode")
    if mode is None:
        mode = _LEGACY_PROVIDER_TYPE_TO_MODE.get(prov_meta.get("provider_type"), OPENAI_CHAT)

    provider_inst = _build_provider_for_mode(
        mode,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        extra_headers=prov_meta.get("extra_headers"),
        max_retries=int(prov_meta.get("max_retries", 0)),
        **kwargs,
    )
    # Inject metadata properties for tracing
    provider_inst._api_key_env = api_key_env
    provider_inst._provider_type = prov_meta.get("provider_type", "openai_compatible")
    provider_inst._mode = mode
    
    return provider_inst


def provider_from_config(
    cfg: "Any",
    *,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Construct a provider from a :class:`ProviderConfig` (single-provider file).

    Resolves the API key from ``cfg`` (explicit ``api_key`` wins, else the
    ``api_key_env`` environment variable) and raises a clear error when a key is
    required by the mode but missing.
    """
    resolved_api_key = cfg.resolve_api_key(api_key)
    provider_inst = _build_provider_for_mode(
        cfg.mode,
        api_key=resolved_api_key,
        base_url=cfg.base_url,
        extra_headers=cfg.extra_headers,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    provider_inst._api_key_env = cfg.api_key_env
    provider_inst._mode = cfg.mode
    return provider_inst
