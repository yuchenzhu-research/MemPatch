from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from typing import Any, Dict, List
from experiments.multiagent.contracts import ExperimentRunManifest
from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes
from experiments.multiagent.methods import (
    NaiveLastWriteWinsFixedCandidateMethod,
    AppendOnlyLexicalTopKMethod,
    DirectJudgeReplayMethod,
    ReTraceStageAReplayMethod,
)
from experiments.multiagent.metrics import (
    compute_fixed_candidate_metrics,
    aggregate_fixed_candidate_metrics,
)


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


def run_fc_comparison(mode: str) -> Dict[str, Any]:
    """Run fixed-candidate method comparison evaluation."""
    run_id = f"fc_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    episodes = get_fc_dev_episodes()
    methods = [
        NaiveLastWriteWinsFixedCandidateMethod(),
        AppendOnlyLexicalTopKMethod(k=5),
        DirectJudgeReplayMethod(),
        ReTraceStageAReplayMethod(),
    ]

    results_rows: List[Dict[str, Any]] = []
    run_results = []

    for ep in episodes:
        for method in methods:
            res = method.run_fixed_episode(ep)
            run_results.append((ep, res))

            ep_metrics = compute_fixed_candidate_metrics(ep, res)

            conflict_density = ep.stress_factors.get("conflict_density", 0.0)
            delay_depth = ep.stress_factors.get("delay_depth", 0)
            trace_available = method.method_name in ("ReTrace_StageA_Replay", "DirectJudge_Replay")
            num_subagents = len(set(s.producer_id for s in ep.submissions))
            num_submissions = len(ep.submissions)
            role_diversity = len(set(ep.subagent_roles))
            recovery_present = ep.failure_type == "temporary_blocker_recovery"

            for m_name, m_val in ep_metrics.items():
                results_rows.append({
                    "run_id": run_id,
                    "episode_id": ep.episode_id,
                    "domain": ep.domain,
                    "failure_type": ep.failure_type,
                    "protocol_mode": ep.protocol_mode,
                    "scientific_status": "pipeline_validation_only",
                    "split": ep.split,
                    "method_name": method.method_name,
                    "backbone_model": None,
                    "proposal_source": ep.proposal_source,
                    "candidate_source": "fixed_candidate",
                    "number_of_subagents": num_subagents,
                    "number_of_submissions": num_submissions,
                    "role_diversity": role_diversity,
                    "conflict_density": conflict_density,
                    "delay_depth": delay_depth,
                    "recovery_present": recovery_present,
                    "metric_name": m_name,
                    "metric_value": m_val,
                    "trace_available": trace_available,
                    "calls": res.metadata.get("calls", 0) if res.metadata else 0,
                    "tokens": res.metadata.get("tokens", None) if res.metadata else None,
                    "latency_ms": res.metadata.get("latency_ms", None) if res.metadata else None,
                })

    aggregated = aggregate_fixed_candidate_metrics(run_results)

    manifest = ExperimentRunManifest(
        run_id=run_id,
        split="development_only",
        methods=tuple(m.method_name for m in methods),
        episode_ids=tuple(ep.episode_id for ep in episodes),
        model_config={"api": "offline_replay", "protocol_mode": "oracle_edge_replay"},
        prompt_hashes={},
        code_commit_sha=get_git_commit_sha(),
        created_at=datetime.datetime.now().isoformat(),
        mode=mode,
    )

    os.makedirs("outputs", exist_ok=True)

    jsonl_path = "outputs/fc_method_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")

    summary_path = "outputs/fc_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    manifest_dict = manifest.to_dict()
    manifest_dict.update({
        "dataset_version": "fc_dev_v1",
        "episode_factory_hash": "hash_placeholder",
        "method_config": {},
        "random_seed": 42,
        "notes": "Fixed-candidate development-only offline replay."
    })

    manifest_path = "outputs/fc_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    details_path = "outputs/fc_run_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump([res.to_dict() for _, res in run_results], f, indent=2)

    return {
        "results_count": len(results_rows),
        "aggregated": aggregated,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "details_path": details_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fixed-Candidate Method Comparison Evaluation (Packet 4B)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline_replay",
        choices=["offline_replay", "smoke_live", "official_frozen"],
        help="Comparison execution mode",
    )
    args = parser.parse_args()

    if args.mode != "offline_replay":
        print(f"Error: Mode '{args.mode}' is not runnable offline in this packet.")
        return

    res = run_fc_comparison(args.mode)
    print("Fixed-Candidate Method Comparison completed successfully.")
    print(f"Results: {res['jsonl_path']}, {res['summary_path']}, {res['manifest_path']}")
    print(f"Total metric rows: {res['results_count']}")

    # Print per-method overall accuracy
    for key, vals in sorted(res["aggregated"].items()):
        if "__overall__all" in key:
            acc = vals.get("authorization_accuracy", "N/A")
            method = key.split("__")[0]
            print(f"  {method} overall auth accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
