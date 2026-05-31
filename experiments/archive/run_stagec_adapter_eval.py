#!/usr/bin/env python3
"""
Formal Stage C adapter evaluation runner.
Compares base Qwen3-4B prompt-only output with Qwen3-4B-4bit MLX LoRA adapter output.
Computes fine-grained metrics including valid_json, action_type_match, target_grounding,
evidence_grounding, exact_match, and DPA final status accuracy.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission, TypedRevisionTarget
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes
from retracemem.evaluation.multiagent.data.silver_compositional import make_example
from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy
from retracemem.authorization import authorize, SharedCandidateView
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
from retracemem.multiagent.utils import rename_submission, format_assistant_response

VALID_ACTION_TYPES = {
    "SUPERSEDES",
    "BLOCKS",
    "RELEASES",
    "UNCERTAIN",
    "REAFFIRMS",
    "NO_REVISION",
}


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def get_test_source() -> List[Any]:
    """Reconstruct the exact test_source sequence using fixed RNG_SEED=42."""
    episodes_with_gold = generate_expanded_episodes()
    labeled_examples = []
    for ep, gold in episodes_with_gold:
        targets_by_submission = defaultdict(list)
        for target in gold.gold_typed_targets:
            targets_by_submission[target.submission_id].append(target)
        for submission in ep.submissions:
            targets = tuple(targets_by_submission.get(submission.submission_id, ()))
            if not targets:
                continue
            labeled_examples.append(make_example(ep, submission, targets))

    heldout_source = [ex for ex in labeled_examples if ex.episode_id.endswith("_v5")]
    rng = random.Random(42)
    rng.shuffle(heldout_source)
    midpoint = max(1, len(heldout_source) // 2)
    valid_source = heldout_source[:midpoint]
    test_source = heldout_source[midpoint:]
    if not test_source:
        test_source = valid_source[-1:]
    return test_source





def extract_first_json_array(text: str) -> List[Any]:
    """
    Extracts the first valid JSON array from model output.
    Strips Qwen-style <think>...</think> content and tolerates leading/trailing prose.
    """
    # Strip <think>...</think> block cleanly
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed <think> to the end
    if "<think>" in cleaned:
        cleaned = cleaned[:cleaned.find("<think>")]

    start_idx = cleaned.find("[")
    if start_idx == -1:
        raise ValueError("No JSON array start '[' found")

    end_indices = [i for i, char in enumerate(cleaned) if char == "]" and i > start_idx]
    if not end_indices:
        raise ValueError("No JSON array end ']' found")

    # Backtrack from the last ']' to find the first parseable array (longest match first)
    for end_idx in reversed(end_indices):
        candidate = cleaned[start_idx:end_idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse a valid JSON array")


def normalize_action(a: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic normalization for action comparison."""
    ev_ids = a.get("evidence_ids")
    if isinstance(ev_ids, list):
        ev_tuple = tuple(sorted(str(e) for e in ev_ids))
    elif isinstance(ev_ids, tuple):
        ev_tuple = tuple(sorted(str(e) for e in ev_ids))
    elif isinstance(ev_ids, str):
        ev_tuple = (ev_ids,)
    else:
        ev_tuple = ()

    return {
        "action_type": a.get("action_type"),
        "target_belief_id": a.get("target_belief_id"),
        "target_condition_id": a.get("target_condition_id"),
        "replacement_belief_id": a.get("replacement_belief_id"),
        "evidence_ids": ev_tuple,
    }


def normalize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [normalize_action(a) for a in actions]
    return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True))


def get_target_ids(actions: List[Dict[str, Any]]) -> List[str]:
    res = []
    for a in actions:
        if a.get("target_belief_id"):
            res.append(a["target_belief_id"])
        if a.get("target_condition_id"):
            res.append(a["target_condition_id"])
    return res


def get_evidence_ids(actions: List[Dict[str, Any]]) -> List[str]:
    res = []
    for a in actions:
        evs = a.get("evidence_ids") or []
        if isinstance(evs, str):
            res.append(evs)
        else:
            res.extend(evs)
    return res


def evaluate_predictions(
    test_source: List[Any],
    adapter_gen_dir: Path,
    base_gen_dir: Path,
    policy: PromptTypedRevisionPolicy,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluates base vs adapter predictions across test source examples."""
    per_case_results = []
    parse_errors = []
    dpa_traces = []

    for i, ex in enumerate(test_source):
        case_id = f"test_{i:02d}"
        sub = ex.method_visible_input
        
        # Rename namespace to match SFT augmented dataset
        old_ns = ex.episode_id
        new_ns = f"{ex.episode_id}__heldout_base"
        sub_renamed = rename_submission(sub, old_ns, new_ns)

        # 1. Setup Shared Candidate View
        new_ev = next(e for e in sub_renamed.evidence_context if e.evidence_id == sub_renamed.new_evidence_id)
        view = SharedCandidateView(
            instance_id=sub_renamed.instance_id,
            query_id=sub_renamed.query_id,
            query=sub_renamed.query,
            evidence_context=sub_renamed.evidence_context,
            new_evidence=new_ev,
            candidate_beliefs=sub_renamed.candidate_beliefs,
            candidate_replacement_beliefs=sub_renamed.candidate_replacement_beliefs,
            candidate_conditions_by_belief=sub_renamed.candidate_conditions_by_belief,
            dependency_edges_by_belief=sub_renamed.dependency_edges_by_belief,
        )

        # 2. Get Gold actions and Gold DPA
        gold_json_str = format_assistant_response(ex).replace(old_ns, new_ns)
        gold_actions = json.loads(gold_json_str)
        gold_out = policy.parse_response(gold_json_str, example_id=ex.example_id, submission=sub_renamed)
        gold_result = authorize(view, gold_out.proposal_batches)
        gold_statuses = gold_result.trace["fine_grained_statuses"]

        # Record gold trace
        dpa_traces.append({
            "case_id": case_id,
            "variant": "gold",
            "fine_grained_statuses": gold_statuses,
            "trace": gold_result.trace,
        })

        case_summary: Dict[str, Any] = {
            "case": case_id,
            "gold_actions": gold_actions,
            "adapter": {},
            "base": {},
        }

        # 3. Evaluate each variant (adapter vs base)
        variants = [
            ("adapter", adapter_gen_dir / f"test_{i:02d}_adapter.txt"),
            ("base", base_gen_dir / f"test_{i:02d}_base.txt"),
        ]

        for var_name, file_path in variants:
            raw_output = ""
            if file_path.exists():
                raw_output = file_path.read_text(encoding="utf-8")

            # Parse array
            valid_json = False
            pred_actions = []
            parse_err_msg = ""
            try:
                pred_actions = extract_first_json_array(raw_output)
                
                # Check for array of dicts type correctness
                if not isinstance(pred_actions, list):
                    raise ValueError("Parsed output is not a JSON list/array")
                for idx, act in enumerate(pred_actions):
                    if not isinstance(act, dict):
                        raise ValueError(f"Array item at index {idx} is not an object")

                valid_json = True
            except Exception as e:
                parse_err_msg = str(e)
                parse_errors.append({
                    "case_id": case_id,
                    "variant": var_name,
                    "raw_output_snippet": raw_output[:300] + ("..." if len(raw_output) > 300 else ""),
                    "parse_error": parse_err_msg,
                })

            # Check action schema validation constraints
            pred_out = None
            if valid_json:
                pred_json_str = json.dumps(pred_actions)
                pred_out = policy.parse_response(pred_json_str, example_id=ex.example_id, submission=sub_renamed)
                # If there are schema verification errors, record them
                if not pred_out.parsing_valid:
                    parse_err_msg = "; ".join(pred_out.errors)
                    parse_errors.append({
                        "case_id": case_id,
                        "variant": var_name,
                        "raw_output_snippet": raw_output[:300] + ("..." if len(raw_output) > 300 else ""),
                        "parse_error": f"Schema validation errors: {parse_err_msg}",
                    })

            # Replay through DPA
            proposal_batches = ()
            if valid_json and pred_out and pred_out.parsing_valid:
                proposal_batches = pred_out.proposal_batches

            pred_result = authorize(view, proposal_batches)
            pred_statuses = pred_result.trace["fine_grained_statuses"]

            # Record DPA trace
            dpa_traces.append({
                "case_id": case_id,
                "variant": var_name,
                "fine_grained_statuses": pred_statuses,
                "trace": pred_result.trace,
            })

            # Metrics calculations
            action_type_match = False
            target_grounding = False
            evidence_grounding = False
            exact_match = False
            dpa_accuracy = False

            if valid_json:
                pred_types = [a.get("action_type") for a in pred_actions]
                gold_types = [a.get("action_type") for a in gold_actions]
                action_type_match = (Counter(pred_types) == Counter(gold_types))

                pred_targets = get_target_ids(pred_actions)
                gold_targets = get_target_ids(gold_actions)
                target_grounding = (Counter(pred_targets) == Counter(gold_targets))

                pred_ev = get_evidence_ids(pred_actions)
                gold_ev = get_evidence_ids(gold_actions)
                evidence_grounding = (Counter(pred_ev) == Counter(gold_ev))

                exact_match = (normalize_actions(pred_actions) == normalize_actions(gold_actions))

            dpa_accuracy = (pred_statuses == gold_statuses)

            case_summary[var_name] = {
                "valid_json": valid_json,
                "action_type_match": action_type_match,
                "target_grounding": target_grounding,
                "evidence_grounding": evidence_grounding,
                "exact_match": exact_match,
                "dpa_accuracy": dpa_accuracy,
                "parse_error": parse_err_msg if not valid_json or (pred_out and not pred_out.parsing_valid) else None,
                "pred_actions": pred_actions,
            }

        per_case_results.append(case_summary)

    return per_case_results, parse_errors, dpa_traces


def get_case_subset(gold_actions: List[Dict[str, Any]]) -> str:
    """Classify the gold actions into one of the mutually exclusive evaluation subsets."""
    if not gold_actions:
        return "no_revision_only"
    action_types = [a.get("action_type") for a in gold_actions]
    if len(action_types) == 1:
        act_type = action_types[0]
        if act_type == "SUPERSEDES":
            return "supersedes_only"
        elif act_type == "BLOCKS":
            return "blocks_only"
        elif act_type == "RELEASES":
            return "releases_only"
        elif act_type == "UNCERTAIN":
            return "uncertain_only"
        elif act_type == "REAFFIRMS":
            return "reaffirms_only"
        elif act_type == "NO_REVISION":
            return "no_revision_only"
    # Multi-action scenarios
    if set(action_types) == {"SUPERSEDES", "BLOCKS"}:
        return "multi_action_supersedes_blocks"
    return "multi_action_other"


def compute_aggregate_metrics(
    per_case_results: List[Dict[str, Any]],
    test_source: List[Any],
) -> Dict[str, Any]:
    n = len(per_case_results)
    
    # Map case ID to SFT example to inspect domain/metadata
    case_to_ex = {}
    for i, ex in enumerate(test_source):
        case_to_ex[f"test_{i:02d}"] = ex

    summary: Dict[str, Any] = {
        "adapter": {k: 0 for k in ["valid_json", "action_type_match", "target_grounding", "evidence_grounding", "exact_match", "dpa_accuracy"]},
        "base": {k: 0 for k in ["valid_json", "action_type_match", "target_grounding", "evidence_grounding", "exact_match", "dpa_accuracy"]},
        "total": n,
        "subsets": {}
    }

    subset_names = [
        "supersedes_only", "blocks_only", "releases_only", "uncertain_only",
        "reaffirms_only", "no_revision_only", "multi_action_supersedes_blocks",
        "multi_action_other", "domain_software", "domain_research", "heldout_base"
    ]
    for sub_name in subset_names:
        summary["subsets"][sub_name] = {
            "adapter": {k: 0 for k in ["valid_json", "action_type_match", "target_grounding", "evidence_grounding", "exact_match", "dpa_accuracy"]},
            "base": {k: 0 for k in ["valid_json", "action_type_match", "target_grounding", "evidence_grounding", "exact_match", "dpa_accuracy"]},
            "total": 0
        }

    for case in per_case_results:
        case_id = case["case"]
        ex = case_to_ex[case_id]
        gold_actions = case["gold_actions"]
        
        assigned_subsets = ["heldout_base"]
        assigned_subsets.append(get_case_subset(gold_actions))
        
        dom = getattr(ex, "domain", "unknown").lower()
        if "soft" in dom or "code" in dom:
            assigned_subsets.append("domain_software")
        elif "research" in dom or "paper" in dom:
            assigned_subsets.append("domain_research")
            
        for sub_name in assigned_subsets:
            for var in ["adapter", "base"]:
                res = case[var]
                summary["subsets"][sub_name][var]["valid_json"] += int(res["valid_json"])
                summary["subsets"][sub_name][var]["action_type_match"] += int(res["action_type_match"])
                summary["subsets"][sub_name][var]["target_grounding"] += int(res["target_grounding"])
                summary["subsets"][sub_name][var]["evidence_grounding"] += int(res["evidence_grounding"])
                summary["subsets"][sub_name][var]["exact_match"] += int(res["exact_match"])
                summary["subsets"][sub_name][var]["dpa_accuracy"] += int(res["dpa_accuracy"])
            summary["subsets"][sub_name]["total"] += 1

        for var in ["adapter", "base"]:
            res = case[var]
            summary[var]["valid_json"] += int(res["valid_json"])
            summary[var]["action_type_match"] += int(res["action_type_match"])
            summary[var]["target_grounding"] += int(res["target_grounding"])
            summary[var]["evidence_grounding"] += int(res["evidence_grounding"])
            summary[var]["exact_match"] += int(res["exact_match"])
            summary[var]["dpa_accuracy"] += int(res["dpa_accuracy"])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ReTrace Stage C Adapter Evaluation Runner")
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="outputs/local_training/stagec_qwen3_4b_silver/data/test.jsonl",
        help="Path to the test.jsonl file",
    )
    parser.add_argument(
        "--adapter-gen-dir",
        type=str,
        default="outputs/local_training/stagec_qwen3_4b_silver/generations",
        help="Directory containing adapter predictions",
    )
    parser.add_argument(
        "--base-gen-dir",
        type=str,
        default="outputs/local_training/stagec_qwen3_4b_silver/generations_base",
        help="Directory containing base model predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/local_training/stagec_qwen3_4b_silver/evaluation_report",
        help="Directory to save evaluation report files",
    )
    parser.add_argument(
        "--model-base-id",
        type=str,
        default="Qwen3-4B-4bit",
        help="Base model identifier",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="outputs/local_training/stagec_qwen3_4b_silver/adapters",
        help="LoRA adapter checkpoint path",
    )
    parser.add_argument(
        "--decoding-params",
        type=str,
        default="temp=0.0",
        help="LLM generation decoding parameter string",
    )

    args = parser.parse_args()

    test_data_path = Path(args.test_data_path)
    adapter_gen_dir = Path(args.adapter_gen_dir)
    base_gen_dir = Path(args.base_gen_dir)
    output_dir = Path(args.output_dir)

    print(f"Starting Stage C Adapter Evaluation...")
    print(f"  Test data path: {test_data_path}")
    print(f"  Adapter outputs: {adapter_gen_dir}")
    print(f"  Base outputs: {base_gen_dir}")
    print(f"  Output directory: {output_dir}\n")

    # Build evaluation test set
    test_source = get_test_source()
    if not test_source:
        raise ValueError("Reconstructed test set source is empty.")

    policy = PromptTypedRevisionPolicy()

    # Evaluate
    per_case, errors, traces = evaluate_predictions(test_source, adapter_gen_dir, base_gen_dir, policy)
    metrics_summary = compute_aggregate_metrics(per_case, test_source)

    # Write files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics.json
    (output_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    # Write per_case_results.jsonl
    with open(output_dir / "per_case_results.jsonl", "w", encoding="utf-8") as f:
        for c in per_case:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write parse_errors.jsonl
    with open(output_dir / "parse_errors.jsonl", "w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    # Write dpa_traces.jsonl
    with open(output_dir / "dpa_traces.jsonl", "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Write manifest.json
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_sha(),
        "dataset_path": str(test_data_path),
        "model_base_identifier": args.model_base_id,
        "adapter_path": str(args.adapter_path),
        "decoding_parameters": args.decoding_params,
        "metric_summary": metrics_summary,
        "warning": "silver_synthetic_training_only / not_for_paper_main_results",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Report to console
    print(f"Evaluation complete. Reports written to: {output_dir}")
    print("\n================== ACCURACY SUMMARY ==================")
    print(f"{'Metric':<22} | {'Adapter (Score)':<18} | {'Base Prompt (Score)':<18}")
    print("-" * 66)
    
    total = metrics_summary["adapter"]["total"]
    
    for m in ["valid_json", "action_type_match", "target_grounding", "evidence_grounding", "exact_match", "dpa_accuracy"]:
        ad_count = metrics_summary["adapter"][m]
        base_count = metrics_summary["base"][m]
        ad_pct = (ad_count / total) * 100.0
        base_pct = (base_count / total) * 100.0
        ad_str = f"{ad_count}/{total} ({ad_pct:.1f}%)"
        base_str = f"{base_count}/{total} ({base_pct:.1f}%)"
        print(f"{m:<22} | {ad_str:<18} | {base_str:<18}")

    print("======================================================")


if __name__ == "__main__":
    main()
