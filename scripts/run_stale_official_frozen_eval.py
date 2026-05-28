#!/usr/bin/env python3
"""Offline wiring runner for the official frozen STALE benchmark dataset.

This script does not make any live API call. It uses MockLLMProvider so the
wiring against the official frozen dataset can be validated without leaking
gold fields or contacting external services.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retracemem.adapters.stale_official_runner import (
    StaleOfflineRunConfig,
    run_offline_wiring_demo,
)
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient


_DEFAULT_RESPONSE = json.dumps({
    "verdicts": [],
    "edges": [],
    "answer": "Offline wiring demo: no model answer was produced.",
})


def make_offline_client(cache_path: Path) -> CachedLLMClient:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = JSONLCache(str(cache_path))
    provider = MockLLMProvider(default_response=_DEFAULT_RESPONSE)
    return CachedLLMClient(cache=cache, provider_client=provider)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline STALE official frozen wiring demo (no live calls).",
    )
    parser.add_argument(
        "--dataset-path",
        default="data_external/stale_official_frozen/T1_T2_400_FULL.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stale_official_frozen_wiring_demo",
    )
    parser.add_argument("--limit-t1", type=int, default=2)
    parser.add_argument("--limit-t2", type=int, default=2)
    args = parser.parse_args()

    cache_dir = Path(args.output_dir) / "caches"
    client_a = make_offline_client(cache_dir / "stage_a.jsonl")
    client_b = make_offline_client(cache_dir / "stage_b.jsonl")

    config = StaleOfflineRunConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        limit_t1=args.limit_t1,
        limit_t2=args.limit_t2,
    )
    result = run_offline_wiring_demo(config, client_a, client_b)
    print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
