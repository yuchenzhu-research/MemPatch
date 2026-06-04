#!/usr/bin/env python3
"""Run an LLM provider on ReTrace-Bench and write predictions.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.model_runner import PROVIDERS, run_model_predictions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a provider/model over ReTrace-Bench scenarios and write prediction JSONL.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="scenarios.jsonl file or a directory containing scenarios.jsonl",
    )
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--model", required=True, help="Provider model name")
    parser.add_argument(
        "--out-predictions",
        required=True,
        help="Path to write canonical predictions JSONL",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of new scenarios to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append and skip scenario_ids already present in --out-predictions",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after provider or JSON-parse failures without writing malformed rows",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Override the environment variable used for the provider API key",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for openai_compatible providers, or to override a provider default",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Maximum completion tokens for provider calls",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between provider calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = run_model_predictions(
            data=args.data,
            provider=args.provider,
            model=args.model,
            out_predictions=args.out_predictions,
            max_cases=args.max_cases,
            resume=args.resume,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            sleep_seconds=args.sleep_seconds,
            continue_on_error=args.continue_on_error,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {count} prediction(s) to {args.out_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
