#!/usr/bin/env python3
"""Ambiguity-and-Scope Stage A/B feasibility runner.

Internal development diagnostic only.
- Not an official benchmark.
- Replay/mock mode is for runner correctness.
- Live mode is exploratory development-only via the validated provider/cache/accounting boundary.
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
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from retracemem.evaluation.ambiguity_scope import (
    compute_metrics,
    format_report,
    load_dataset,
    run_cases,
    validate_dataset_balance,
)
from retracemem.evaluation.manifest import (
    RunConfiguration,
    RunManifest,
    compute_file_sha256,
)
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.providers.base import BaseLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider

DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "internal_dev",
    "ambiguity_scope_controlled_v0.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "outputs", "ambiguity_scope_dev"
)
PILOT_CASE_IDS = (
    "as_supersession_03",
    "as_prereq_block_02",
    "as_protected_02",
    "as_temp_vs_persistent_01",
    "as_current_vs_historical_01",
    "as_tentative_intention_01",
    "as_insufficient_evidence_02",
    "as_scope_trap_01",
)
_PROMPT_EDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "retrace_llm", "evidence_edge_prediction_v0.txt"
)
_PROMPT_JUDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "directjudge", "direct_usability_v1.txt"
)


class CappedProviderWrapper(BaseLLMProvider):
    def __init__(self, inner: BaseLLMProvider, max_calls: int, max_tokens: int) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls_made = 0
        self.tokens_used = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"Hard call cap of {self.max_calls} reached.")
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached.")
        trace = self.inner.generate(*args, **kwargs)
        self.calls_made += 1
        self.tokens_used += trace.total_tokens
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached during call.")
        return trace


def _file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return hashlib.sha256(f.read().encode("utf-8")).hexdigest()


def _select_cases(cases: list[Any], case_ids: tuple[str, ...]) -> list[Any]:
    by_id = {case.case.case_id: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown case id(s): {missing}")
    return [by_id[case_id] for case_id in case_ids]


def _old_evidence_texts(case: Any) -> list[str]:
    new_id = case.raw_record["new_evidence_id"]
    return [e["text"] for e in case.raw_record["evidence_context"] if e["evidence_id"] != new_id]


def _new_evidence_text(case: Any) -> str:
    new_id = case.raw_record["new_evidence_id"]
    return next(e["text"] for e in case.raw_record["evidence_context"] if e["evidence_id"] == new_id)


def _possible_bias_or_ambiguity(case: Any) -> str:
    if case.abstention_required_belief_ids:
        return "Reviewer should confirm that evidence is genuinely insufficient rather than weakly decisive."
    if case.scope_trap:
        return "Reviewer should inspect whether protected beliefs are truly outside the new evidence scope."
    if case.category == "temporary_constraint_vs_persistent_preference":
        return "Reviewer should confirm the temporary constraint does not imply loss of persistent preference."
    if case.category == "current_state_vs_historical_fact":
        return "Reviewer should confirm historical facts remain protected despite current-state change."
    return "No obvious drafting ambiguity beyond normal human-review verification."


def write_review_table(cases: list[Any], output_path: str) -> None:
    pilot = set(PILOT_CASE_IDS)
    lines = [
        "# Ambiguity-and-Scope Dataset Human Review Table",
        "",
        "Generated artifact only; not a tracked canonical document.",
        "",
        "| case id | category | pilot_review_required | old evidence | new evidence | candidate belief(s) | expected Stage A fine-grained status | expected comparable Stage B verdict | annotations | annotation rationale | possible_bias_or_ambiguity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        raw = case.raw_record
        beliefs = "<br>".join(f"{b['belief_id']}: {b['proposition']}" for b in raw["candidate_beliefs"])
        annotations = (
            f"protected={list(case.protected_belief_ids)}; "
            f"stale_target={list(case.stale_target_belief_ids)}; "
            f"abstention={list(case.abstention_required_belief_ids)}"
        )
        row = [
            case.case.case_id,
            case.category,
            "yes" if case.case.case_id in pilot else "no",
            "<br>".join(_old_evidence_texts(case)).replace("|", "\\|"),
            _new_evidence_text(case).replace("|", "\\|"),
            beliefs.replace("|", "\\|"),
            json.dumps(raw["expected_stage_a_status"], ensure_ascii=False),
            json.dumps(raw["expected_comparable_status"], ensure_ascii=False),
            annotations,
            case.rationale.replace("|", "\\|"),
            _possible_bias_or_ambiguity(case).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A/B Ambiguity-and-Scope feasibility runner (dev-only).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the internal Ambiguity-and-Scope dataset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the report/manifest (gitignored).",
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "live-dev"),
        default="replay",
        help="Execution mode. 'live-dev' is exploratory development only.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Live provider name (used only with --mode live-dev).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Live model id (used only with --mode live-dev).",
    )
    parser.add_argument("--api-key", default=None, help="Explicit API key for live-dev; otherwise provider env var is used.")
    parser.add_argument("--base-url", default=None, help="Optional custom HTTP endpoint for live-dev.")
    parser.add_argument("--live-approved", action="store_true", help="Required explicit approval flag for live-dev API calls.")
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids to run without editing the dataset.")
    parser.add_argument("--pilot-only", action="store_true", help="Run the fixed 8-case human-review pilot subset.")
    parser.add_argument("--max-calls", type=int, default=24, help="Hard live-dev total call cap across both stages.")
    parser.add_argument("--max-tokens", type=int, default=60000, help="Hard live-dev total token cap across both stages.")
    parser.add_argument("--write-review-table", action="store_true", help="Write the compact human-review markdown artifact and exit.")
    parser.add_argument(
        "--review-output",
        default=os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "outputs",
            "ambiguity_scope_review",
            "ambiguity_scope_review_table.md",
        ),
        help="Path for the generated human-review table.",
    )
    parser.add_argument(
        "--skip-balance-check",
        action="store_true",
        help="Skip the strict 4-per-category balance check (development bypass).",
    )
    args = parser.parse_args()

    dataset_path = os.path.normpath(args.dataset)
    output_dir = os.path.normpath(args.output_dir)
    run_id = f"run-ambiguity-scope-{uuid.uuid4()}"

    print("=" * 70)
    print("AMBIGUITY-AND-SCOPE STAGE A/B FEASIBILITY RUNNER")
    print("=" * 70)
    print("  Disclaimer: internal development feasibility study only.")
    print(f"  Run ID: {run_id}")
    print(f"  Mode:   {args.mode.upper()}")
    print(f"  Cases:  {dataset_path}")
    print(f"  Output: {output_dir}")
    print()

    cases = load_dataset(dataset_path)
    if not args.skip_balance_check:
        validate_dataset_balance(cases)
    if args.write_review_table:
        write_review_table(cases, os.path.normpath(args.review_output))
        print(f"Review table saved to: {os.path.normpath(args.review_output)}")
        return
    selected_ids: tuple[str, ...] = ()
    if args.pilot_only:
        selected_ids = PILOT_CASE_IDS
    elif args.case_ids:
        selected_ids = tuple(item.strip() for item in args.case_ids.split(",") if item.strip())
    if selected_ids:
        cases = _select_cases(cases, selected_ids)
        print(f"Selected {len(cases)} case(s): {', '.join(selected_ids)}")
    print(f"Loaded {len(cases)} cases.")

    os.makedirs(output_dir, exist_ok=True)

    client_a = None
    client_b = None
    cache_path = ""
    if args.mode == "live-dev":
        if not args.live_approved:
            print("Refusing live execution without --live-approved.")
            sys.exit(2)
        if not selected_ids:
            print("Refusing live execution without --pilot-only or --case-ids.")
            sys.exit(2)
        cache_path = str(Path(output_dir) / "ambiguity_scope_live_cache.jsonl")
        cache = JSONLCache(cache_path)
        provider = CappedProviderWrapper(
            HTTPLLMProvider(api_key=args.api_key, base_url=args.base_url),
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
        )
        client_a = CachedLLMClient(cache=cache, provider_client=provider, cost_accountant=CostAccounting())
        client_b = CachedLLMClient(cache=cache, provider_client=provider, cost_accountant=CostAccounting())
        print(f"Live provider configured: provider={args.provider}, model={args.model}")
        print(f"Safety caps: max_calls={args.max_calls}, max_tokens={args.max_tokens}")

    with tempfile.TemporaryDirectory() as tmp:
        results = run_cases(
            cases,
            tmp if args.mode == "replay" else output_dir,
            client_a=client_a,
            client_b=client_b,
            model_id="mock" if args.mode == "replay" else args.model,
            provider="mock" if args.mode == "replay" else args.provider,
        )
        metrics = compute_metrics(cases, results)
        report = format_report(cases, results, metrics)

    report_path = Path(output_dir) / "ambiguity_scope_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    config = RunConfiguration(
        run_id=run_id,
        stage_and_method_name="AmbiguityScope_StageAB_dev",
        provider_name="mock" if args.mode == "replay" else args.provider,
        model_id="mock" if args.mode == "replay" else args.model,
        temperature=0.0,
        max_tokens=args.max_tokens if args.mode == "live-dev" else None,
        prompt_hashes={
            "evidence_edge_prediction": _file_hash(_PROMPT_EDGE),
            "direct_usability": _file_hash(_PROMPT_JUDGE),
        },
        cache_path=cache_path,
        dataset_checksum=compute_file_sha256(dataset_path),
        comparison_regime="ambiguity_scope_controlled",
        metadata={
            "max_calls": args.max_calls if args.mode == "live-dev" else None,
            "selected_case_ids": [case.case.case_id for case in cases],
            "live_requires_explicit_approval": True,
        },
    )
    manifest = RunManifest(
        config=config,
        aggregate_cost={
            "stage_a": {
                "calls": metrics.stage_a_calls,
                "tokens": metrics.stage_a_tokens,
            },
            "stage_b": {
                "calls": metrics.stage_b_calls,
                "tokens": metrics.stage_b_tokens,
            },
        },
        instance_count=len(cases),
        output_path=str(report_path),
    )
    manifest_path = Path(output_dir) / "ambiguity_scope_manifest.json"
    manifest.save(str(manifest_path))
    print(f"Manifest saved to: {manifest_path}")

    print()
    print("-" * 70)
    print("Summary")
    print("-" * 70)
    agg = report["aggregate"]
    print(f"  Overall accuracy A/B:               "
          f"{agg['overall_comparable_accuracy']['stage_a']} vs "
          f"{agg['overall_comparable_accuracy']['stage_b']}")
    print(f"  Stale blocking A/B:                 "
          f"{agg['stale_blocking_accuracy']['stage_a']} vs "
          f"{agg['stale_blocking_accuracy']['stage_b']}")
    print(f"  Protected preservation A/B:         "
          f"{agg['protected_belief_preservation']['stage_a']} vs "
          f"{agg['protected_belief_preservation']['stage_b']}")
    print(f"  Abstention accuracy A/B:            "
          f"{agg['abstention_accuracy']['stage_a']} vs "
          f"{agg['abstention_accuracy']['stage_b']}")
    print(f"  Unsupported confident revision A/B: "
          f"{agg['unsupported_confident_revision_rate']['stage_a']} vs "
          f"{agg['unsupported_confident_revision_rate']['stage_b']}")
    print(f"  Execution / parse errors:           "
          f"{agg['execution_errors']} / {agg['parse_errors']}")
    print()
    print("Reminder: this output is an internal feasibility diagnostic, not a benchmark result.")


if __name__ == "__main__":
    main()
