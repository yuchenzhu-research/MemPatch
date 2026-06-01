#!/usr/bin/env python3
"""AB-1C controlled A/B development runner with live provider capability and manifest generation.

WARNING: This is an INTERNAL DEVELOPMENT PROTOCOL CHECK ONLY.
- Default: Replay/mock execution only.
- Live API calls allowed ONLY with --live flag.
- Enforces hard call and token caps for live safety.
- NOT an official benchmark.
- NOT strict call-budget matched.
- NO claim that ReTrace outperforms DirectJudge.

Usage:
    .venv/bin/python scripts/run_controlled_ab_dev.py [--live] [--provider openai] [--model gpt-4o]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.controlled_ab import (
    compute_metrics,
    format_report,
    load_cases,
    run_case,
)
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.providers.base import BaseLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider

_CASES_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "retrace_learn", "v1", "internal_dev", "controlled_ab_cases.json"
)
_PROMPT_EDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "retrace_llm", "evidence_edge_prediction_v0.txt"
)
_PROMPT_JUDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "directjudge", "direct_usability_v1.txt"
)


class CappedProviderWrapper(BaseLLMProvider):
    """Safety wrapper that enforces hard call and token caps on a provider."""

    def __init__(self, inner: BaseLLMProvider, max_calls: int = 5, max_tokens: int = 10000) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls_made = 0
        self.tokens_used = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"Hard call cap of {self.max_calls} reached. Aborting execution.")
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached. Aborting execution.")
            
        trace = self.inner.generate(*args, **kwargs)
        self.calls_made += 1
        self.tokens_used += trace.total_tokens
        
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached during call. Aborting execution.")
        return trace


def _get_file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AB-1C controlled A/B development runner with live provider capability."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live API calls to LLM provider (otherwise replay/mock).",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("RETRACE_LIVE_PROVIDER", "openai"),
        help="LLM provider name (e.g. openai, google).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RETRACE_LIVE_MODEL", "gpt-4o"),
        help="Model ID to use for live generation.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key. If omitted, falls back to env vars.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for the HTTP LLM provider.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "controlled_ab_dev"),
        help="Output directory for results (gitignored).",
    )
    parser.add_argument(
        "--cases",
        default=_CASES_PATH,
        help="Path to internal dev cases JSON file.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=1000,
        help="Hard call cap across the entire run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000000,
        help="Hard token cap across the entire run.",
    )
    args = parser.parse_args()

    cases_path = os.path.normpath(args.cases)
    output_dir = os.path.normpath(args.output_dir)
    run_id = f"run-{uuid.uuid4()}"

    print("=" * 70)
    print("AB-1C CONTROLLED A/B DEVELOPMENT RUNNER")
    print("=" * 70)
    print()
    print("  WARNING: Internal development protocol check only.")
    print("  NOT an official benchmark.")
    print("  NOT strict call-budget matched.")
    print("  NO claim that ReTrace outperforms DirectJudge.")
    print()
    print(f"  Run ID:  {run_id}")
    print(f"  Mode:    {'LIVE API CALLS' if args.live else 'REPLAY/MOCK'}")
    if args.live:
        print(f"  Config:  provider={args.provider}, model={args.model}")
    print(f"  Cases:   {cases_path}")
    print(f"  Output:  {output_dir}")
    print()

    # Load cases
    cases = load_cases(cases_path)
    print(f"Loaded {len(cases)} internal development cases.")

    # 1. Prompt template hashes
    prompt_hashes = {
        "evidence_edge_prediction": _get_file_hash(_PROMPT_EDGE),
        "direct_usability": _get_file_hash(_PROMPT_JUDGE),
    }

    # 2. Setup Provider Client & Cache if running live
    client_a = None
    client_b = None
    cache_path = ""
    
    if args.live:
        cache_path = os.path.join(output_dir, "controlled_ab_live_cache.jsonl")
        cache = JSONLCache(cache_path)
        
        # Instantiate HTTP provider and wrap with safety cap
        http_provider = HTTPLLMProvider(api_key=args.api_key, base_url=args.base_url)
        capped_provider = CappedProviderWrapper(http_provider, max_calls=args.max_calls, max_tokens=args.max_tokens)
        
        cost_accountant_a = CostAccounting()
        cost_accountant_b = CostAccounting()
        client_a = CachedLLMClient(cache=cache, provider_client=capped_provider, cost_accountant=cost_accountant_a)
        client_b = CachedLLMClient(cache=cache, provider_client=capped_provider, cost_accountant=cost_accountant_b)
        print(f"  ✓ Setup live provider client with cache and safety caps (max {args.max_calls} calls, {args.max_tokens} tokens).")
        print()

    # Execute all cases
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] {case.case_id} ({case.case_type})")
            try:
                res = run_case(
                    case,
                    tmp_dir if not args.live else output_dir,
                    client_a=client_a,
                    client_b=client_b,
                    model_id=args.model if args.live else "mock",
                    provider=args.provider if args.live else "mock",
                )
                if res.stage_a_error:
                    print(f"        Stage A ERROR: {res.stage_a_error}")
                if res.stage_b_error:
                    print(f"        Stage B ERROR: {res.stage_b_error}")
                results.append(res)
            except Exception as e:
                print(f"        FATAL Case Execution Failure: {e}")
                raise

    # Compute metrics
    metrics = compute_metrics(cases, results)

    # Format report
    report = format_report(metrics, results)

    # Print compact summary
    print()
    print("-" * 70)
    print("AGGREGATE SUMMARY")
    print("-" * 70)
    agg = report["aggregate"]
    print(f"  Total cases:              {agg['total_cases']}")
    print(f"  Total belief decisions:   {agg['total_belief_decisions']}")
    print(f"  Stage A accuracy:         {agg['stage_a_accuracy']}")
    print(f"  Stage B accuracy:         {agg['stage_b_accuracy']}")
    print(f"  Stage A Ablation accuracy:{agg['stage_a_ablation_accuracy']}")
    print(f"  Stage A coverage:         {agg['stage_a_coverage']}")
    print(f"  Stage B coverage:         {agg['stage_b_coverage']}")
    print(f"  Stage A Ablation coverage:{agg['stage_a_ablation_coverage']}")
    print(f"  Stage A status breakdown: {agg['stage_a_status_breakdown']}")
    print(f"  Stage B verdict breakdown:{agg['stage_b_verdict_breakdown']}")
    print(f"  Stage A Ablation breakdown:{agg['stage_a_ablation_status_breakdown']}")
    print(f"  Obsolete misuse:          {agg['obsolete_misuse']}")
    print(f"  Protected preserved:      {agg['protected_belief_preserved']}")
    print(f"  Rollback recovery:        {agg['rollback_recovery']}")
    print(f"  Unsupported revision:     {agg['unsupported_revision_rate']}")
    print(f"  Execution errors:         {agg['execution_errors']}")
    print(f"  Parse errors:             {agg.get('parse_errors', 0)}")
    print()
    print("  Observed cost (NOT matched):")
    for stage, label in [("stage_a", "Stage A"), ("stage_b", "Stage B"), ("stage_a_ablation", "Stage A Ablation")]:
        c = agg["observed_cost"][stage]
        print(f"    {label}: calls={c['calls']}, tokens={c['tokens']}, "
              f"cache_hits={c['cache_hits']}, latency_ms={c['latency_ms']}")
    print()

    # Write output report
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "controlled_ab_dev_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report written to: {output_path}")

    # Generate and save run manifest
    config = RunConfiguration(
        run_id=run_id,
        stage_and_method_name="AB-1C_StageAB_controlled",
        provider_name=args.provider if args.live else "mock",
        model_id=args.model if args.live else "mock",
        prompt_hashes=prompt_hashes,
        parser_schema_versions={
            "evidence_edge_prediction": "evidence_edge_response_v0",
            "direct_usability": "direct_usability_response_v1",
        },
        cache_path=cache_path,
        dataset_checksum=compute_file_sha256(cases_path),
        comparison_regime="controlled_candidate_view",
        scientific_status="exploratory_pilot" if args.live else "development_live",
        not_for_main_table=True if args.live else False,
    )
    
    # Capture final cost accounting from client accountants if they exist
    aggregate_cost = {
        "stage_a": client_a.cost_accountant.to_dict() if client_a else {},
        "stage_b": client_b.cost_accountant.to_dict() if client_b else {},
    }
    
    manifest = RunManifest(
        config=config,
        aggregate_cost=aggregate_cost,
        instance_count=len(cases),
        output_path=output_path,
    )
    
    manifest_path = os.path.join(output_dir, f"run_manifest_{run_id}.json")
    manifest.save(manifest_path)
    
    # Stable symlink/static name for the manifest
    static_manifest_path = os.path.join(output_dir, "controlled_ab_dev_manifest.json")
    manifest.save(static_manifest_path)
    
    print(f"Run manifest written to: {static_manifest_path}")
    print()
    print("=" * 70)
    print("DONE. Controlled A/B check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
