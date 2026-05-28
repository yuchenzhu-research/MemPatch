#!/usr/bin/env python3
"""Runner for official frozen STALE offline wiring and approved live subsets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retracemem.adapters.stale_official_runner import (
    StaleLiveRunConfig,
    StaleOfflineRunConfig,
    run_live_stageab_generation,
    run_offline_wiring_demo,
    run_official_evaluator,
)
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


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
        description="Official frozen STALE Stage A/B runner.",
    )
    parser.add_argument("--mode", choices=("replay", "live-dev", "evaluate"), default="replay")
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
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--judge-provider", default="siliconflow")
    parser.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--evaluator-concurrency", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "live-dev":
        if load_dotenv is not None:
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        config = StaleLiveRunConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            limit_t1=args.limit_t1,
            limit_t2=args.limit_t2,
            provider=args.provider,
            model=args.model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            http_timeout_seconds=args.http_timeout_seconds,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
            evaluator_concurrency=args.evaluator_concurrency,
        )
        result = run_live_stageab_generation(config)
        manifest = result["manifest"]
        if manifest["errors"]:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        run_official_evaluator(
            answers_path=manifest["stage_a_export"],
            dataset_path=manifest["selected_subset_path"],
            output_path=str(Path(args.output_dir) / "stage_a_official_eval.json"),
            model_method="stage_a_retrace",
            conflict_type="stage_a",
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            concurrency=args.evaluator_concurrency,
            timeout=args.http_timeout_seconds,
        )
        run_official_evaluator(
            answers_path=manifest["stage_b_export"],
            dataset_path=manifest["selected_subset_path"],
            output_path=str(Path(args.output_dir) / "stage_b_official_eval.json"),
            model_method="stage_b_directjudge",
            conflict_type="stage_b",
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            concurrency=args.evaluator_concurrency,
            timeout=args.http_timeout_seconds,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    if args.mode == "evaluate":
        raise SystemExit("--mode evaluate is reserved; use --mode live-dev to run generation plus evaluator.")

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
