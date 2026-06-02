#!/usr/bin/env python3
"""Stage C (ReTrace-Learn) evaluation runner.

Stage C evaluates an adapter / local / SFT-style typed revision proposer on the
same dev70 episode structure used by Stage A / Stage B, and routes its typed
action proposals through the exact same deterministic kernel:

    Stage C proposer -> RevisionGate -> deterministic DPA / authorize(...)
                     -> SharedMemoryCommitResult

To keep the evaluation deterministic and free of external API calls, the heavy
decoding step (a local adapter / SFT checkpoint, e.g. MLX or transformers +
LoRA) is expected to run *offline* and dump one decoded completion per
submission. This runner replays those decoded generations through the canonical
constrained post-validation parser and the shared commit / DPA path, then
reuses Stage A's metric computation so Stage C numbers are directly comparable.

Sources of decoded generations:
  --generations-dir DIR   read ``{submission_id}.txt`` per submission (real run)
  --smoke / --mock        offline NO_REVISION generations (no files, no API)

Stage C plugs into the same shared evaluation engine as Stage A/B; it imports
reusable modules (case loading, the per-episode commit/DPA pipeline, and metric
computation) rather than any Stage runner script.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
)
from retracemem.evaluation.multiagent.cases import load_eval_cases
from retracemem.evaluation.multiagent.pipeline import run_retrace_variant_on_episode
from retracemem.evaluation.multiagent.metrics import (
    _STATUS_MAP_A_TO_COMPARABLE,
    compute_eval_metrics,
    compute_stage_a_action_metrics,
)
from retracemem.proposers.replay import (
    CANONICAL_ACTIONS,
    LocalAdapterReplayProposer,
    build_replay_proposer,
)
from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy


@dataclass(frozen=True)
class StageCEvalConfig:
    proposer_source: str = "replay"  # replay | mock
    generations_dir: str | None = None
    generation_suffix: str = ".txt"
    policy_variant: str = "adapter_replay"
    backbone_model: str | None = None
    checkpoint_id: str | None = None
    allowed_actions: tuple[str, ...] = CANONICAL_ACTIONS
    constrained_postvalidation: bool = True
    max_cases: int | None = None
    smoke: bool = False
    output_dir: str = "outputs/runs/stagec_dev70"


def build_stagec_proposer(config: StageCEvalConfig) -> LocalAdapterReplayProposer:
    """Construct the Stage C proposer based on the configured decoded source."""
    if config.proposer_source == "mock":
        return build_replay_proposer(
            mock=True,
            policy_variant=config.policy_variant,
            backbone_model=config.backbone_model,
            checkpoint_id=config.checkpoint_id,
            allowed_actions=config.allowed_actions,
            constrained_postvalidation=config.constrained_postvalidation,
        )
    if config.proposer_source == "replay":
        if not config.generations_dir:
            raise ValueError(
                "proposer_source='replay' requires --generations-dir pointing at "
                "a directory of decoded '{submission_id}.txt' generations."
            )
        return build_replay_proposer(
            generations_dir=config.generations_dir,
            suffix=config.generation_suffix,
            policy_variant=config.policy_variant,
            backbone_model=config.backbone_model,
            checkpoint_id=config.checkpoint_id,
            allowed_actions=config.allowed_actions,
            constrained_postvalidation=config.constrained_postvalidation,
        )
    raise ValueError(f"Unknown proposer_source: {config.proposer_source!r}")


def _belief_status_correctness(
    pred_statuses: dict[str, str],
    gold_statuses: dict[str, str],
) -> dict[str, Any]:
    """Per-belief comparison using the same comparable mapping as Stage A."""
    per_belief = {}
    correct = 0
    for bid, gold_status in gold_statuses.items():
        gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
        pred_raw = pred_statuses.get(bid, "UNRESOLVED")
        pred_comp = _STATUS_MAP_A_TO_COMPARABLE.get(pred_raw, "UNCERTAIN")
        is_correct = pred_comp == gold_comp
        if is_correct:
            correct += 1
        per_belief[bid] = {
            "gold_status": gold_status,
            "gold_comparable": gold_comp,
            "pred_status": pred_raw,
            "pred_comparable": pred_comp,
            "correct": is_correct,
        }
    total = len(gold_statuses)
    return {
        "per_belief": per_belief,
        "correct_beliefs": correct,
        "total_beliefs": total,
        "belief_status_accuracy": correct / total if total else 1.0,
        "episode_exact_match": correct == total,
    }


def build_case_record(
    episode: FixedCandidateInputEpisode,
    gold: FixedCandidateGoldRecord,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    final_dpa_statuses: dict[str, str],
    trace_dict: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a rich per-case Stage C record (requirement 6).

    Contains: case/episode id, method-visible submissions, raw proposer output,
    parsed typed actions, commit / decision audit + gate decisions, final DPA
    belief statuses, gold labels, correctness metrics, and failure reasons.
    """
    raw_by_sub = {r["submission_id"]: r for r in raw_rows}
    parsed_by_sub = {r["submission_id"]: r for r in parsed_rows}

    submission_records = []
    for sub in episode.submissions:
        sub_id = sub.submission_id
        raw = raw_by_sub.get(sub_id, {})
        parsed = parsed_by_sub.get(sub_id, {})
        pred_actions = parsed.get("actions", [])
        gold_actions = tuple(
            t for t in gold.gold_typed_targets if t.submission_id == sub_id
        )
        action_metrics = compute_stage_a_action_metrics(pred_actions, gold_actions, sub)
        parse_error = raw.get("parse_error") or parsed.get("parse_error")
        submission_records.append(
            {
                "submission_id": sub_id,
                "method_visible_input": sub.to_dict(),
                "raw_proposer_output": {
                    "prompt": raw.get("prompt", ""),
                    "raw_response": raw.get("raw_response", ""),
                },
                "parsed_actions": pred_actions,
                "proposal_edges": parsed.get("proposal_edges", []),
                "decision_audit": parsed.get("decision_audit"),
                "gold_typed_targets": [t.to_dict() for t in gold_actions],
                "action_metrics": action_metrics,
                "parse_or_validation_error": parse_error,
            }
        )

    status_correctness = _belief_status_correctness(
        final_dpa_statuses, gold.gold_snapshot.belief_statuses
    )

    return {
        "episode_id": episode.episode_id,
        "case_id": episode.episode_id,
        "domain": episode.domain,
        "failure_type": episode.failure_type_public_or_controlled,
        "submissions": submission_records,
        "commit_result": {
            "gate_decisions": trace_dict["dpa_trace"].get("gate_decisions", []),
            "cumulative_beliefs": trace_dict["dpa_trace"].get("cumulative_beliefs", []),
            "cumulative_evidence": trace_dict["dpa_trace"].get("cumulative_evidence", []),
        },
        "final_dpa_belief_statuses": final_dpa_statuses,
        "gold_belief_statuses": gold.gold_snapshot.belief_statuses,
        "gold_rationale": gold.gold_snapshot.rationale,
        "correctness": status_correctness,
        "any_parse_error": any(
            s["parse_or_validation_error"] is not None for s in submission_records
        ),
    }


def _relabel_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename Stage A failure-breakdown columns to Stage C for clarity.

    Stage C occupies the Stage A computation slot inside ``compute_eval_metrics``
    (Stage B is intentionally empty), so the ``*_a`` columns hold Stage C values.
    """
    relabeled = []
    for r in rows:
        relabeled.append(
            {
                "episode_id": r["episode_id"],
                "failure_type": r["failure_type"],
                "domain": r["domain"],
                "total_beliefs": r["total_beliefs"],
                "correct_beliefs_c": r["correct_beliefs_a"],
                "accuracy_c": r["accuracy_a"],
                "has_parse_error_c": r["has_parse_error_a"],
                "has_grounding_error_c": r["has_grounding_error_a"],
            }
        )
    return relabeled


def write_stagec_outputs(
    output_dir: str,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    dpa_trace_rows: list[dict[str, Any]],
    case_records: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def save_jsonl(p: Path, data: list[dict[str, Any]]) -> None:
        with open(p, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    save_jsonl(output_path / "stagec_raw.jsonl", raw_rows)
    save_jsonl(output_path / "stagec_parsed.jsonl", parsed_rows)
    save_jsonl(output_path / "dpa_traces.jsonl", dpa_trace_rows)
    save_jsonl(output_path / "stagec_records.jsonl", case_records)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if failure_rows:
        with open(output_path / "failure_breakdown.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=failure_rows[0].keys())
            writer.writeheader()
            writer.writerows(failure_rows)

    print(f"\u2713 Wrote Stage C raw, parsed, trace, record, and metric files to {output_dir}")


def run_stagec_eval(config: StageCEvalConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Orchestrate Stage C evaluation over the dev70 set."""
    max_cases = config.max_cases
    if config.smoke and max_cases is None:
        max_cases = 3

    print("=" * 80)
    print("STAGE C (ReTrace-Learn) EVALUATION RUNNER")
    print("=" * 80)
    print(f"Proposer source : {config.proposer_source}")
    print(f"Policy variant  : {config.policy_variant}")
    print(f"Generations dir : {config.generations_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Max cases       : {max_cases}")
    print()

    proposer = build_stagec_proposer(config)
    processed_cases = load_eval_cases(max_cases)

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    dpa_trace_rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []

    for idx, (episode, gold) in enumerate(processed_cases):
        ep_id = episode.episode_id
        print(
            f"[{idx + 1}/{len(processed_cases)}] Evaluating {ep_id} "
            f"({episode.failure_type_public_or_controlled})"
        )

        raw_a, parsed_a, final_dpa_statuses, trace_dict = run_retrace_variant_on_episode(
            episode, gold, proposer, mock=False
        )

        raw_rows.append({"episode_id": ep_id, "submissions": raw_a})
        parsed_rows.append(
            {
                "episode_id": ep_id,
                "submissions": parsed_a,
                "final_belief_statuses": final_dpa_statuses,
            }
        )
        dpa_trace_rows.append(
            {
                "episode_id": ep_id,
                "dpa_trace": trace_dict["dpa_trace"],
                "final_belief_statuses": trace_dict["final_belief_statuses"],
            }
        )
        case_records.append(
            build_case_record(
                episode, gold, raw_a, parsed_a, final_dpa_statuses, trace_dict
            )
        )

    # Reuse Stage A's aggregate metric computation. Stage C results occupy the
    # Stage A slot; Stage B is intentionally empty and discarded below.
    print("\nCalculating metrics (reusing Stage A computation)...")
    global_metrics, failure_breakdown_rows = compute_eval_metrics(
        processed_cases, parsed_rows, [], raw_rows, []
    )
    stagec_metrics = {"stage_c": global_metrics["stage_a"]}
    failure_rows = _relabel_failure_rows(failure_breakdown_rows)

    print("\n" + "-" * 40)
    print("STAGE C (Adaptive proposal + DPA) METRICS:")
    print("-" * 40)
    for k, v in stagec_metrics["stage_c"].items():
        print(f"  {k:30s}: {v:.4f}")
    print("-" * 40)

    git_sha = "unknown"
    try:
        git_sha = os.popen("git rev-parse HEAD").read().strip() or "unknown"
    except Exception:
        pass

    temp_policy = PromptTypedRevisionPolicy(allowed_actions=config.allowed_actions)
    prompt_template_hash = hashlib.sha256(
        temp_policy.build_system_prompt().encode("utf-8")
    ).hexdigest()

    manifest = {
        "run_identifier": "stagec_development_run / not_final_paper_result",
        "stage": "C",
        "method_family": "ReTrace-Learn",
        "executed_at": datetime.datetime.now().isoformat(),
        "proposer_source": config.proposer_source,
        "policy_variant": config.policy_variant,
        "backbone_model": config.backbone_model,
        "checkpoint_id": config.checkpoint_id,
        "generations_dir": config.generations_dir,
        "constrained_postvalidation": config.constrained_postvalidation,
        "allowed_actions": list(config.allowed_actions),
        "is_live_api_result": False,
        "smoke": config.smoke,
        "cases_evaluated": len(processed_cases),
        "output_directory": config.output_dir,
        "git_commit_sha": git_sha,
        "code_commit_sha": git_sha,
        "prompt_template_hash": prompt_template_hash,
        "parser_version": "PromptTypedRevisionPolicy_v1",
        "response_schema_version": "v1_canonical",
    }

    write_stagec_outputs(
        config.output_dir,
        raw_rows,
        parsed_rows,
        dpa_trace_rows,
        case_records,
        failure_rows,
        stagec_metrics,
        manifest,
    )

    print("\nStage C evaluation completed successfully!")
    return stagec_metrics, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C (ReTrace-Learn) evaluation runner"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--generations-dir",
        default=None,
        help="Directory of decoded '{submission_id}.txt' generations (replay mode).",
    )
    source_group.add_argument(
        "--mock",
        action="store_true",
        help="Use offline NO_REVISION generations (no files, no API).",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: offline mock generations on the first few cases.",
    )
    parser.add_argument("--generation-suffix", default=".txt", help="Generation filename suffix.")
    parser.add_argument(
        "--policy-variant",
        default="adapter_replay",
        help="Label for the proposer policy variant (e.g. adapter_replay, open_lora).",
    )
    parser.add_argument("--backbone-model", default=None, help="Backbone model id for provenance.")
    parser.add_argument("--checkpoint-id", default=None, help="Adapter/SFT checkpoint id for provenance.")
    parser.add_argument(
        "--allowed-actions",
        default=None,
        help="Comma-separated subset of the canonical action vocabulary.",
    )
    parser.add_argument(
        "--no-constrained-postvalidation",
        action="store_true",
        help="Record that constrained post-validation was disabled (parser stays strict).",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases.")
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/stagec_dev70",
        help="Output directory for Stage C run artifacts.",
    )
    args = parser.parse_args()

    if args.allowed_actions:
        allowed_actions = tuple(
            a.strip() for a in args.allowed_actions.split(",") if a.strip()
        )
    else:
        allowed_actions = CANONICAL_ACTIONS

    use_mock = args.mock or (args.smoke and not args.generations_dir)
    proposer_source = "mock" if use_mock else "replay"

    config = StageCEvalConfig(
        proposer_source=proposer_source,
        generations_dir=args.generations_dir,
        generation_suffix=args.generation_suffix,
        policy_variant=args.policy_variant,
        backbone_model=args.backbone_model,
        checkpoint_id=args.checkpoint_id,
        allowed_actions=allowed_actions,
        constrained_postvalidation=not args.no_constrained_postvalidation,
        max_cases=args.max_cases,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )
    run_stagec_eval(config)


if __name__ == "__main__":
    main()
