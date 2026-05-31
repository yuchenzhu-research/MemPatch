"""Stage A/B orchestration over the shared evaluation engine.

Thin glue: load cases -> run Stage A (typed proposer -> commit -> DPA) and
Stage B (DirectJudge) -> compute metrics -> write artifacts.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from retracemem.proposers.typed_revision_policy import (
    ClosedAPIZeroShotProposer,
    ClosedAPIZeroShotConstrainedProposer,
    ConflictAwareConstrainedProposer,
)
from collections import Counter

from retracemem.evaluation.multiagent.config import EvalRunConfig, make_live_client
from retracemem.evaluation.multiagent.cases import load_eval_cases
from retracemem.evaluation.multiagent.data.paper1_balanced import GENERATOR_VERSION as PAPER1_GENERATOR_VERSION
from retracemem.evaluation.multiagent.pipeline import run_retrace_variant_on_episode
from retracemem.evaluation.multiagent.directjudge import run_directjudge_on_episode
from retracemem.evaluation.multiagent.metrics import compute_eval_metrics
from retracemem.evaluation.multiagent.artifacts import write_run_outputs

load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)


def run_stageab_eval(config: EvalRunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Uniform orchestration entry point for evaluation of Stage A and Stage B."""
    live = config.live
    dry_run = config.dry_run
    mock = config.mock
    max_cases = config.max_cases
    resume = config.resume
    provider = config.provider
    model = config.model
    api_key = config.api_key
    base_url = config.base_url
    output_dir = config.output_dir
    constrained = config.constrained
    stage_a_variant = config.stage_a_variant
    diagnostic = config.diagnostic
    dataset = config.dataset

    method = config.method
    if method == "StageC-ICL":
        allow_fallback = config.allow_fallback_to_zeroshot
        if not allow_fallback:
            raise ValueError("StageC-ICL requires approved exemplars, but none were loaded (default fail-closed).")
        method = "zero_shot_fallback"

    if output_dir.startswith("artifacts/"):
        print("\n⚠ WARNING: output_dir starts with 'artifacts/'. Please consider using 'outputs/runs/' instead.\n")

    constrained_path = method == "StageA-Constrained" or constrained
    if stage_a_variant == "conflict_aware" and not constrained_path:
        print("\n⚠ WARNING: --stage-a-variant conflict_aware has no effect without --constrained; using default zero-shot proposer.\n")

    print("=" * 80)
    print("STAGE A VS STAGE B API EVALUATION RUNNER (CORE)")
    print("=" * 80)
    print(f"Mode: {'LIVE API' if live else 'DRY RUN' if dry_run else 'MOCK REPLAY'}")
    print(f"Provider: {provider}, Model: {model}")
    print(f"Output Directory: {output_dir}")
    print()

    # Load Cases
    processed_cases = load_eval_cases(max_cases, dataset=dataset)

    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize live client if live
    client = None
    if live:
        client = make_live_client(
            output_dir, provider, model, api_key, base_url,
            provider_config_path=config.provider_config_path,
        )

    # Resume capability: load already processed cases
    stage_a_raw_rows = []
    stage_b_raw_rows = []
    stage_a_parsed_rows = []
    stage_b_parsed_rows = []
    dpa_trace_rows = []
    resumed_episodes = set()
    if resume:
        a_raw_file = output_path / "stage_a_raw.jsonl"
        b_raw_file = output_path / "stage_b_raw.jsonl"
        a_parsed_file = output_path / "stage_a_parsed.jsonl"
        b_parsed_file = output_path / "stage_b_parsed.jsonl"
        dpa_traces_file = output_path / "dpa_traces.jsonl"

        if a_raw_file.exists() and b_raw_file.exists():
            print("✓ Resuming from existing results...")
            def load_jsonl(p: Path):
                rows = []
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                return rows

            try:
                stage_a_raw_rows = load_jsonl(a_raw_file)
                stage_b_raw_rows = load_jsonl(b_raw_file)
                stage_a_parsed_rows = load_jsonl(a_parsed_file)
                stage_b_parsed_rows = load_jsonl(b_parsed_file)
                dpa_trace_rows = load_jsonl(dpa_traces_file)

                finished_a = {r["episode_id"] for r in stage_a_parsed_rows}
                finished_b = {r["episode_id"] for r in stage_b_parsed_rows}
                resumed_episodes = finished_a & finished_b
                print(f"✓ Resumed {len(resumed_episodes)} fully completed cases.")
            except Exception as e:
                print(f"⚠ Failed to resume cleanly: {e}. Will run from scratch.")
                resumed_episodes = set()
                stage_a_raw_rows = []
                stage_b_raw_rows = []
                stage_a_parsed_rows = []
                stage_b_parsed_rows = []
                dpa_trace_rows = []

    # Proposer setup
    prov_a = provider if live else "mock"
    mid_a = model if live else None
    cli_a = client if live else None

    if method == "zero_shot_fallback":
        from retracemem.proposers.typed_revision_policy import ClosedAPIICLProposer
        proposer_a = ClosedAPIICLProposer(
            provider_kind=prov_a,
            model_id=mid_a,
            client=cli_a,
            allow_fallback_to_zeroshot=True,
            repair_on_parse_error=config.repair_on_parse_error,
            max_repair_rounds=config.max_repair_rounds,
        )
    elif method == "StageA-Constrained" or constrained:
        constrained_cls = (
            ConflictAwareConstrainedProposer
            if stage_a_variant == "conflict_aware"
            else ClosedAPIZeroShotConstrainedProposer
        )
        proposer_a = constrained_cls(
            provider_kind=prov_a,
            model_id=mid_a,
            client=cli_a,
            diagnostic_mode=diagnostic,
            repair_on_parse_error=config.repair_on_parse_error,
            max_repair_rounds=config.max_repair_rounds,
        )
    else:
        proposer_a = ClosedAPIZeroShotProposer(
            provider_kind=prov_a,
            model_id=mid_a,
            client=cli_a,
            diagnostic_mode=diagnostic,
            repair_on_parse_error=config.repair_on_parse_error,
            max_repair_rounds=config.max_repair_rounds,
        )

    # Main Orchestration Loop
    for idx, (episode, gold) in enumerate(processed_cases):
        ep_id = episode.episode_id
        if ep_id in resumed_episodes:
            print(f"[{idx+1}/{len(processed_cases)}] Skipping (resumed) {ep_id}")
            continue

        print(f"[{idx+1}/{len(processed_cases)}] Evaluating {ep_id} ({episode.failure_type_public_or_controlled})")

        # Run Stage A
        raw_a, parsed_a, final_dpa_statuses, trace_dict = run_retrace_variant_on_episode(
            episode, gold, proposer_a, mock
        )
        stage_a_raw_rows.append({"episode_id": ep_id, "submissions": raw_a})
        stage_a_parsed_rows.append({
            "episode_id": ep_id,
            "submissions": parsed_a,
            "final_belief_statuses": final_dpa_statuses,
        })
        dpa_trace_rows.append({
            "episode_id": ep_id,
            "dpa_trace": trace_dict["dpa_trace"],
            "final_belief_statuses": trace_dict["final_belief_statuses"],
        })

        # Run Stage B
        raw_b, parsed_b, strict_statuses, canonical_statuses = run_directjudge_on_episode(
            episode, gold, client, model, provider, dry_run, mock
        )
        stage_b_raw_rows.append({"episode_id": ep_id, "submissions": raw_b})
        stage_b_parsed_rows.append({
            "episode_id": ep_id,
            "submissions": parsed_b,
            "strict_final_belief_statuses": strict_statuses,
            "canonicalized_final_belief_statuses": canonical_statuses,
            "final_belief_statuses": canonical_statuses,
        })

    # Metrics calculation
    print("\nCalculating metrics...")
    global_metrics, failure_breakdown_rows = compute_eval_metrics(
        processed_cases, stage_a_parsed_rows, stage_b_parsed_rows, stage_a_raw_rows, stage_b_raw_rows
    )

    print("\n" + "-" * 40)
    print("STAGE A (Decomposition + DPA) METRICS:")
    print("-" * 40)
    for k, v in global_metrics["stage_a"].items():
        print(f"  {k:28s}: {v:.4f}")

    print("\n" + "-" * 40)
    print("STAGE B (Direct usability judge) METRICS:")
    print("-" * 40)
    for k, v in global_metrics["stage_b"].items():
        print(f"  {k:28s}: {v:.4f}")
    print("-" * 40)

    # Save metrics and failure breakdown files
    manifest = {}
    if not dry_run:
        # Save failure breakdown csv
        csv_file = output_path / "failure_breakdown.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=failure_breakdown_rows[0].keys())
            writer.writeheader()
            writer.writerows(failure_breakdown_rows)

        # Calculate prompt template hash
        from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy
        temp_policy = PromptTypedRevisionPolicy()
        sys_prompt = temp_policy.build_system_prompt()
        prompt_template_hash = hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()

        manifest = {
            "run_identifier": "development_live_api_run / not_final_paper_result" if live else "development_run",
            "executed_at": datetime.datetime.now().isoformat(),
            "run_mode": "live" if live else "mock" if mock else "dry-run",
            "is_live_api_result": live,
            "mock_default_used": mock,
            "provider": provider,
            "model": model,
            "resume_mode": resume,
            "temperature": 0.0,
            "seed": 42,
            "decoding_parameters": {
                "temperature": 0.0,
                "seed": 42,
            },
            "cases_evaluated": len(processed_cases),
            "dataset_name": dataset,
            "dataset_size": len(processed_cases),
            "failure_type_counts": dict(Counter(ep.failure_type_public_or_controlled for ep, _ in processed_cases)),
            "domain_counts": dict(Counter(ep.domain for ep, _ in processed_cases)),
            "generator_version": PAPER1_GENERATOR_VERSION if dataset == "paper1_balanced" else "dev_expansion_v1",
            "constrained": constrained_path,
            "stage_a_variant": stage_a_variant,
            "output_directory": output_dir,
            "git_commit_sha": "unknown",
            "code_commit_sha": "unknown",
            "prompt_template_hash": prompt_template_hash,
            "parser_version": "PromptTypedRevisionPolicy_v1",
            "response_schema_version": "v1_canonical",
        }
        if live:
            manifest["warning"] = "development_live_api_run / not_final_paper_result"

        try:
            res_git = os.popen("git rev-parse HEAD").read().strip()
            if res_git:
                manifest["git_commit_sha"] = res_git
                manifest["code_commit_sha"] = res_git
        except Exception:
            pass

        write_run_outputs(
            output_dir,
            stage_a_raw_rows,
            stage_b_raw_rows,
            stage_a_parsed_rows,
            stage_b_parsed_rows,
            dpa_trace_rows,
            global_metrics,
            manifest,
            dry_run,
        )

    print("\nEvaluation Completed successfully!")
    return global_metrics, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A vs Stage B API Evaluation Runner")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="Enable live API calls")
    mode_group.add_argument("--dry-run", action="store_true", help="Dry run prompt generation and dataset checking")
    mode_group.add_argument("--mock", action="store_true", help="Explicit mock replay mode")
    
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases to evaluate")
    parser.add_argument("--resume", action="store_true", help="Resume from cached files")
    parser.add_argument("--provider", default="siliconflow", help="API Provider name")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Model ID")
    parser.add_argument("--api-key", default=None, help="Explicit API key")
    parser.add_argument("--base-url", default=None, help="Explicit base URL")
    parser.add_argument("--provider-config", default=None, help="Path to a single-provider config file (configs/providers/*.yaml).")
    parser.add_argument("--output-dir", default="outputs/runs/stageab_dev70", help="Output directory")
    parser.add_argument("--constrained", action="store_true", help="Use constrained zero-shot proposer")
    parser.add_argument("--diagnostic", action="store_true", help="Enable diagnostic mode (decision audit)")
    parser.add_argument("--repair-on-parse-error", action="store_true", help="Enable multi-round parse error repair")
    parser.add_argument("--max-repair-rounds", type=int, default=0, help="Maximum number of repair rounds")
    args = parser.parse_args()

    config = EvalRunConfig(
        live=args.live,
        dry_run=args.dry_run,
        mock=args.mock,
        max_cases=args.max_cases,
        resume=args.resume,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        provider_config_path=args.provider_config,
        output_dir=args.output_dir,
        constrained=args.constrained,
        diagnostic=args.diagnostic,
        method=None,
        repair_on_parse_error=args.repair_on_parse_error,
        max_repair_rounds=args.max_repair_rounds,
    )
    run_stageab_eval(config)


if __name__ == "__main__":
    main()
