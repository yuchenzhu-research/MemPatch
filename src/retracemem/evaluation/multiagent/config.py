"""Evaluation run configuration and live client construction (shared A/B/C)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers import get_provider
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
    output_dir: str = "outputs/runs/stageab_dev70"
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
) -> CachedLLMClient:
    """Initializes and returns a live CachedLLMClient."""
    actual_api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    if not actual_api_key:
        raise ValueError("Live mode requires API key to be set via --api-key or SILICONFLOW_API_KEY env var.")
    if not provider or provider == "mock":
        raise ValueError("Live mode requires a valid non-mock --provider.")
    if not model:
        raise ValueError("Live mode requires a valid --model ID.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / "api_cache.jsonl"
    cache = JSONLCache(str(cache_file))
    provider_client = get_provider(provider_name=provider, api_key=actual_api_key, base_url=base_url)
    client = CachedLLMClient(cache=cache, provider_client=provider_client)
    print(f"✓ Initialized live API client with cache at: {cache_file}")
    return client
