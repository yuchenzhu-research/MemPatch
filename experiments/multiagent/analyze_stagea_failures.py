#!/usr/bin/env python3
"""Stage A Diagnostics and Failure Analysis Script.

It loads a Stage A/B evaluation output directory and computes detailed diagnostics:
- Action type confusion matrix (gold vs pred)
- Target and evidence grounding error breakdowns
- False NO_REVISION bias details
- Per-failure type performance metrics
- Representative failure classification Markdown
- summary.json overview
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure src and repo root are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.contracts import FixedCandidateSubmission, TypedRevisionTarget

_STATUS_MAP_A_TO_COMPARABLE = {
    "AUTHORIZED": "USABLE",
    "BLOCKED": "NOT_USABLE",
    "SUPERSEDED": "NOT_USABLE",
    "UNRESOLVED": "UNCERTAIN",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Stage A performance failures.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the directory containing stage_a_parsed.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    stage_a_parsed_file = run_dir / "stage_a_parsed.jsonl"
    
    if not stage_a_parsed_file.exists():
        print(f"Error: Could not find {stage_a_parsed_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Stage A parsed results from {stage_a_parsed_file}...")
    stage_a_rows = load_jsonl(stage_a_parsed_file)
    stage_a_map = {r["episode_id"]: r for r in stage_a_rows}

    print("Loading dev dataset episodes and gold expectations...")
    ep_gold_pairs = generate_expanded_episodes()
    print(f"Loaded {len(ep_gold_pairs)} evaluation cases.")

    # Structures for stats
    confusion = defaultdict(int)
    evidence_grounding_errors = []
    no_revision_bias = []
    
    # Per failure type metrics counters
    # failure_type -> {metric: list_of_values}
    failure_type_metrics = defaultdict(lambda: defaultdict(list))
    
    representative_failures = defaultdict(list)
    
    # Global diagnostic counts
    total_cases = 0
    final_status_accuracy_sum = 0.0
    action_type_match_sum = 0.0
    exact_action_match_sum = 0.0
    
    false_no_revision_count = 0
    false_positive_revision_count = 0
    correct_status_wrong_action_count = 0
    correct_action_wrong_grounding_count = 0
    missing_multi_action_count = 0
    total_submissions_count = 0

    all_actions = ["SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"]

    for ep, gold in ep_gold_pairs:
        ep_id = ep.episode_id
        if ep_id not in stage_a_map:
            # Try renamed namespace fallback just in case
            renamed_ep_id = f"{ep_id}__heldout_base"
            if renamed_ep_id in stage_a_map:
                ep_id = renamed_ep_id
            else:
                continue

        total_cases += 1
        run_record = stage_a_map[ep_id]
        
        gold_statuses = gold.gold_snapshot.belief_statuses
        pred_statuses = run_record.get("final_belief_statuses", {})
        
        # 1. Final usability status accuracy
        correct_beliefs_count = 0
        total_beliefs = len(gold_statuses)
        for bid, gold_status in gold_statuses.items():
            gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
            pred_raw = pred_statuses.get(bid, "UNRESOLVED")
            pred_comp = _STATUS_MAP_A_TO_COMPARABLE.get(pred_raw, "UNCERTAIN")
            if pred_comp == gold_comp:
                correct_beliefs_count += 1
                
        status_accuracy = correct_beliefs_count / total_beliefs if total_beliefs > 0 else 1.0
        final_status_accuracy_sum += status_accuracy
        
        # Track submission level action matches
        episode_action_matches = []
        episode_exact_matches = []
        episode_has_action_error = False
        episode_has_grounding_error = False

        for sub_run in run_record.get("submissions", []):
            sub_id = sub_run["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            pred_actions = sub_run.get("actions", [])
            gold_actions = tuple(t for t in gold.gold_typed_targets if t.submission_id == sub_id)
            total_submissions_count += 1

            # 2. Check missing multi-action
            is_gold_multi = len(gold_actions) > 1
            is_pred_multi = len(pred_actions) > 1
            if is_gold_multi and not is_pred_multi:
                missing_multi_action_count += 1
                representative_failures["missing_multi_action"].append({
                    "episode_id": ep_id,
                    "submission_id": sub_id,
                    "gold": [a.to_dict() for a in gold_actions],
                    "pred": pred_actions,
                    "new_evidence": next((e.text for e in orig_sub.evidence_context if e.evidence_id == orig_sub.new_evidence_id), ""),
                })

            # 3. Check NO_REVISION bias
            gold_has_no_rev = any(a.action_type == "NO_REVISION" for a in gold_actions) or not gold_actions
            pred_has_no_rev = any(a["action_type"] == "NO_REVISION" for a in pred_actions) or not pred_actions
            
            if not gold_has_no_rev and pred_has_no_rev:
                false_no_revision_count += 1
                no_revision_bias.append({
                    "episode_id": ep_id,
                    "submission_id": sub_id,
                    "gold_actions": ",".join(a.action_type for a in gold_actions),
                    "predicted_actions": ",".join(a["action_type"] for a in pred_actions),
                    "bias_type": "false_no_revision",
                })
                representative_failures["false_no_revision"].append({
                    "episode_id": ep_id,
                    "submission_id": sub_id,
                    "gold": [a.to_dict() for a in gold_actions],
                    "pred": pred_actions,
                    "new_evidence": next((e.text for e in orig_sub.evidence_context if e.evidence_id == orig_sub.new_evidence_id), ""),
                })
            elif gold_has_no_rev and not pred_has_no_rev:
                false_positive_revision_count += 1
                no_revision_bias.append({
                    "episode_id": ep_id,
                    "submission_id": sub_id,
                    "gold_actions": ",".join(a.action_type for a in gold_actions) if gold_actions else "NO_REVISION",
                    "predicted_actions": ",".join(a["action_type"] for a in pred_actions),
                    "bias_type": "false_positive_revision",
                })

            # 4. Target level Action type matching (Confusion Matrix)
            # Find all visible targets in submission context
            targets = []
            for b in orig_sub.candidate_beliefs:
                targets.append((b.belief_id, "belief"))
            for bid, conds in orig_sub.candidate_conditions_by_belief:
                for c in conds:
                    targets.append((c.condition_id, "condition"))

            for target_id, target_kind in targets:
                # Find gold action
                g_act = "NO_REVISION"
                g_obj = None
                for a in gold_actions:
                    if target_kind == "belief" and a.target_belief_id == target_id:
                        g_act = a.action_type
                        g_obj = a
                        break
                    elif target_kind == "condition" and a.target_condition_id == target_id:
                        g_act = a.action_type
                        g_obj = a
                        break
                
                # Find pred action
                p_act = "NO_REVISION"
                p_obj = None
                for a in pred_actions:
                    if target_kind == "belief" and a.get("target_belief_id") == target_id:
                        p_act = a["action_type"]
                        p_obj = a
                        break
                    elif target_kind == "condition" and a.get("target_condition_id") == target_id:
                        p_act = a["action_type"]
                        p_obj = a
                        break

                confusion[(g_act, p_act)] += 1

                # 5. Wrong actions & Grounding checks
                if g_act != "NO_REVISION" and p_act != "NO_REVISION":
                    if g_act != p_act:
                        episode_has_action_error = True
                        representative_failures["wrong_action_type"].append({
                            "episode_id": ep_id,
                            "submission_id": sub_id,
                            "target_id": target_id,
                            "gold_type": g_act,
                            "pred_type": p_act,
                            "new_evidence": next((e.text for e in orig_sub.evidence_context if e.evidence_id == orig_sub.new_evidence_id), ""),
                        })
                    else:
                        # Grounding checks when action type matches
                        g_ev = set(g_obj.evidence_ids) if g_obj else set()
                        p_ev = set(p_obj.get("evidence_ids", [])) if p_obj else set()
                        
                        g_targ = g_obj.replacement_belief_id if g_obj else None
                        p_targ = p_obj.get("replacement_belief_id") if p_obj else None
                        
                        if g_targ != p_targ:
                            episode_has_action_error = True
                        elif g_ev != p_ev:
                            correct_action_wrong_grounding_count += 1
                            episode_has_grounding_error = True
                            evidence_grounding_errors.append({
                                "episode_id": ep_id,
                                "submission_id": sub_id,
                                "action_type": g_act,
                                "target_id": target_id,
                                "predicted_evidence_ids": ",".join(p_ev),
                                "gold_evidence_ids": ",".join(g_ev),
                                "error_kind": "evidence_grounding_mismatch",
                            })
                            representative_failures["grounding_error"].append({
                                "episode_id": ep_id,
                                "submission_id": sub_id,
                                "target_id": target_id,
                                "gold_ev": list(g_ev),
                                "pred_ev": list(p_ev),
                                "new_evidence": next((e.text for e in orig_sub.evidence_context if e.evidence_id == orig_sub.new_evidence_id), ""),
                            })

            # Calculate submission-level action metrics (matching run_stageab_api_eval.py logic)
            from experiments.multiagent.run_stageab_api_eval import compute_stage_a_action_metrics
            act_metrics = compute_stage_a_action_metrics(pred_actions, gold_actions, orig_sub)
            episode_action_matches.append(act_metrics.get("action_type_match", 0.0))
            episode_exact_matches.append(act_metrics.get("exact_action_match", 0.0))

        sub_action_type_match = sum(episode_action_matches) / len(episode_action_matches) if episode_action_matches else 0.0
        sub_exact_action_match = sum(episode_exact_matches) / len(episode_exact_matches) if episode_exact_matches else 0.0
        
        action_type_match_sum += sub_action_type_match
        exact_action_match_sum += sub_exact_action_match
        
        # Check: status correct but action wrong
        if status_accuracy == 1.0 and (sub_exact_action_match < 1.0 or episode_has_action_error):
            correct_status_wrong_action_count += 1
            representative_failures["correct_status_wrong_action"].append({
                "episode_id": ep_id,
                "gold_snapshot": gold_statuses,
                "pred_snapshot": pred_statuses,
                "gold_actions": [[a.action_type, a.target_belief_id or a.target_condition_id] for a in gold.gold_typed_targets],
                "pred_actions": [[a["action_type"], a.get("target_belief_id") or a.get("target_condition_id")] for s in run_record.get("submissions", []) for a in s.get("actions", [])],
            })

        # Failure type tracking
        ft = gold.failure_type or "unknown"
        failure_type_metrics[ft]["final_status_accuracy"].append(status_accuracy)
        failure_type_metrics[ft]["action_type_match"].append(sub_action_type_match)
        failure_type_metrics[ft]["exact_action_match"].append(sub_exact_action_match)
        failure_type_metrics[ft]["grounding_errors"].append(1.0 if episode_has_grounding_error else 0.0)

    # 1. Output summary.json
    summary = {
        "total_cases": total_cases,
        "final_status_accuracy": final_status_accuracy_sum / total_cases if total_cases > 0 else 0.0,
        "action_type_match": action_type_match_sum / total_cases if total_cases > 0 else 0.0,
        "exact_action_match": exact_action_match_sum / total_cases if total_cases > 0 else 0.0,
        "false_no_revision_count": false_no_revision_count,
        "false_positive_revision_count": false_positive_revision_count,
        "correct_status_wrong_action_count": correct_status_wrong_action_count,
        "correct_action_wrong_grounding_count": correct_action_wrong_grounding_count,
        "missing_multi_action_count": missing_multi_action_count,
        "total_submissions_evaluated": total_submissions_count,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 2. Output action_confusion.csv
    with open(run_dir / "action_confusion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold_action_type"] + all_actions)
        for g in all_actions:
            row = [g]
            for p in all_actions:
                row.append(confusion[(g, p)])
            writer.writerow(row)

    # 3. Output evidence_grounding_errors.csv
    if evidence_grounding_errors:
        with open(run_dir / "evidence_grounding_errors.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=evidence_grounding_errors[0].keys())
            writer.writeheader()
            writer.writerows(evidence_grounding_errors)
    else:
        # write empty template
        with open(run_dir / "evidence_grounding_errors.csv", "w", newline="", encoding="utf-8") as f:
            f.write("episode_id,submission_id,action_type,target_id,predicted_evidence_ids,gold_evidence_ids,error_kind\n")

    # 4. Output no_revision_bias.csv
    if no_revision_bias:
        with open(run_dir / "no_revision_bias.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=no_revision_bias[0].keys())
            writer.writeheader()
            writer.writerows(no_revision_bias)
    else:
        with open(run_dir / "no_revision_bias.csv", "w", newline="", encoding="utf-8") as f:
            f.write("episode_id,submission_id,gold_actions,predicted_actions,bias_type\n")

    # 5. Output per_failure_type_metrics.csv
    with open(run_dir / "per_failure_type_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_type", "case_count", "final_status_accuracy", "action_type_match", "exact_action_match", "grounding_error_rate"])
        for ft, metrics in failure_type_metrics.items():
            count = len(metrics["final_status_accuracy"])
            fs_acc = sum(metrics["final_status_accuracy"]) / count
            at_match = sum(metrics["action_type_match"]) / count
            ea_match = sum(metrics["exact_action_match"]) / count
            ge_rate = sum(metrics["grounding_errors"]) / count
            writer.writerow([ft, count, f"{fs_acc:.4f}", f"{at_match:.4f}", f"{ea_match:.4f}", f"{ge_rate:.4f}"])

    # 6. Output representative_failures.md
    with open(run_dir / "representative_failures.md", "w", encoding="utf-8") as f:
        f.write("# Representative Stage A Revision Action Failures\n\n")
        f.write("This file highlights typical diagnostic failure modes identified in Stage A proposer output.\n\n")
        
        # Section A: False NO_REVISION
        f.write("## 1. False NO_REVISION Bias\n")
        f.write("Cases where gold required specific memory revision actions, but Stage A proposed NO_REVISION.\n\n")
        for i, item in enumerate(representative_failures["false_no_revision"][:3]):
            f.write(f"### Example {i+1} (Episode: {item['episode_id']})\n")
            f.write(f"- **New Evidence text**: {item['new_evidence']}\n")
            f.write(f"- **Gold Actions**: `{item['gold']}`\n")
            f.write(f"- **Predicted Actions**: `{item['pred']}`\n\n")

        # Section B: Missing Multi-Action
        f.write("## 2. Missing Multi-Action Composition\n")
        f.write("Cases where multiple concurrent actions were required but Stage A only proposed one (or none).\n\n")
        for i, item in enumerate(representative_failures["missing_multi_action"][:3]):
            f.write(f"### Example {i+1} (Episode: {item['episode_id']})\n")
            f.write(f"- **New Evidence text**: {item['new_evidence']}\n")
            f.write(f"- **Gold Actions**: `{item['gold']}`\n")
            f.write(f"- **Predicted Actions**: `{item['pred']}`\n\n")

        # Section C: Wrong Action Type
        f.write("## 3. Incorrect Action Type Choice\n")
        f.write("Cases where the target was correctly identified but the action type was wrong (e.g. SUPERSEDES vs UNCERTAIN).\n\n")
        for i, item in enumerate(representative_failures["wrong_action_type"][:3]):
            f.write(f"### Example {i+1} (Episode: {item['episode_id']})\n")
            f.write(f"- **New Evidence text**: {item['new_evidence']}\n")
            f.write(f"- **Target ID**: {item['target_id']}\n")
            f.write(f"- **Gold Type**: {item['gold_type']}\n")
            f.write(f"- **Predicted Type**: {item['pred_type']}\n\n")

        # Section D: Correct status but wrong action
        f.write("## 4. Final Usability Correct but Proposer Actions Wrong\n")
        f.write("Cases where the proposer proposed incorrect DPA edges, but DPA evaluation still computed correct final status by coincidence.\n\n")
        for i, item in enumerate(representative_failures["correct_status_wrong_action"][:3]):
            f.write(f"### Example {i+1} (Episode: {item['episode_id']})\n")
            f.write(f"- **Gold snapshot**: `{item['gold_snapshot']}`\n")
            f.write(f"- **Pred snapshot**: `{item['pred_snapshot']}`\n")
            f.write(f"- **Gold Actions**: `{item['gold_actions']}`\n")
            f.write(f"- **Predicted Actions**: `{item['pred_actions']}`\n\n")

    print(f"✓ Diagnostic analysis completed. Output files generated under {run_dir}/")


if __name__ == "__main__":
    main()
