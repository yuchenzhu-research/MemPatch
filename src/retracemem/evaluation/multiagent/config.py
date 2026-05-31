"""Evaluation run configuration and live client construction (shared A/B/C)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers import (
    ProviderConfigError,
    get_provider,
    load_provider_config_file,
    provider_config_from_registry,
    provider_from_config,
)
from retracemem.providers.cached_client import CachedLLMClient


@dataclass(frozen=True)
class EvalRunConfig:
    live: bool = False
    dry_run: bool = False
    mock: bool = False
    max_cases: int | None = None
    resume: bool = False
    provider: str = "siliconflow"
    model: str = "deepseek-ai/DeepSeek-V3"
    api_key: str | None = None
    base_url: str | None = None
    provider_config_path: str | None = None
    output_dir: str = "outputs/runs/stageab_dev70"
    dataset: str = "dev_expansion"
    constrained: bool = False
    stage_a_variant: str = "default"
    diagnostic: bool = False
    method: str | None = None
    allow_fallback_to_zeroshot: bool = False
    repair_on_parse_error: bool = False
    max_repair_rounds: int = 0


def make_live_client(
    output_dir: str,
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    provider_config_path: str | None = None,
) -> CachedLLMClient:
    """Initialize a live ``CachedLLMClient`` from either a single-provider config
    file (``provider_config_path``) or the ``configs/providers.yaml`` registry
    (``provider`` name).

    The API key is resolved from the resolved provider's ``api_key_env`` (or an
    explicit ``api_key``); it is never hard-coded to a single provider's env var.
    A clear error is raised when a key is required by the mode but missing.
    """
    if not model:
        raise ValueError("Live mode requires a valid --model ID.")

    # Resolve the provider / API key FIRST so a fail-closed missing-key error does
    # not create an empty run directory or cache file as a side effect.
    try:
        if provider_config_path:
            cfg = load_provider_config_file(provider_config_path).with_overrides(base_url=base_url)
            provider_client = provider_from_config(cfg, api_key=api_key)
            label = f"{cfg.name} (mode={cfg.mode})"
        else:
            if not provider or provider == "mock":
                raise ValueError("Live mode requires a valid non-mock --provider or --provider-config.")
            cfg = provider_config_from_registry(provider)
            if cfg is not None:
                cfg = cfg.with_overrides(base_url=base_url)
                provider_client = provider_from_config(cfg, api_key=api_key)
                label = f"{provider} (mode={cfg.mode})"
            else:
                # Unregistered provider: get_provider fails closed with a clear error.
                provider_client = get_provider(provider_name=provider, api_key=api_key, base_url=base_url)
                label = provider
    except ProviderConfigError as exc:
        # Preserve the historical contract ("Live mode requires API key ...") while
        # surfacing the specific env var the resolved provider expects.
        raise ValueError(f"Live mode requires API key: {exc}") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / "api_cache.jsonl"
    cache = JSONLCache(str(cache_file))

    client = CachedLLMClient(cache=cache, provider_client=provider_client)
    print(f"✓ Initialized live API client [{label}] with cache at: {cache_file}")
    return client
