from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from typing import Any, Dict, List
from experiments.multiagent.contracts import ExperimentRunManifest
from experiments.multiagent.episodes_dev import get_dev_episodes
from experiments.multiagent.methods import NaiveLastWriteWinsMethod, ReTraceMethod
from experiments.multiagent.metrics import compute_episode_metrics, aggregate_metrics


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


def run_comparison(mode: str) -> Dict[str, Any]:
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    episodes = get_dev_episodes()
    methods = [NaiveLastWriteWinsMethod(), ReTraceMethod()]

    results_rows: List[Dict[str, Any]] = []
    run_results = []

    for ep in episodes:
        for method in methods:
            res = method.run_episode(ep)
            run_results.append((ep, res))

            # Compute individual episode metrics
            ep_metrics = compute_episode_metrics(ep, res)
            
            # Count trace available
            trace_avail = any(ev.trace_available for ev in res.revision_events)
            num_subagents = len(ep.subagent_roles)
            conflict_density = ep.stress_factors.get("conflict_density", 0.0)
            delay_depth = ep.stress_factors.get("delay_depth", 0)

            # Export one row per metric for plot-readiness
            for m_name, m_val in ep_metrics.items():
                results_rows.append({
                    "run_id": run_id,
                    "episode_id": ep.episode_id,
                    "domain": ep.domain,
                    "failure_type": ep.failure_type,
                    "method_name": method.method_name,
                    "number_of_subagents": num_subagents,
                    "conflict_density": conflict_density,
                    "delay_depth": delay_depth,
                    "metric_name": m_name,
                    "metric_value": m_val,
                    "trace_available": trace_avail,
                    "calls": 0,
                    "tokens": 0,
                    "latency_ms": 0.0,
                })

    # Compute aggregate metrics
    aggregated = aggregate_metrics(run_results)

    # Build manifest
    manifest = ExperimentRunManifest(
        run_id=run_id,
        split="development_only",
        methods=tuple(m.method_name for m in methods),
        episode_ids=tuple(ep.episode_id for ep in episodes),
        model_config={"api": "offline_replay"},
        prompt_hashes={},
        code_commit_sha=get_git_commit_sha(),
        created_at=datetime.datetime.now().isoformat(),
        mode=mode,
    )

    os.makedirs("outputs", exist_ok=True)
    
    # Save Results JSONL
    jsonl_path = "outputs/multiagent_method_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")

    # Save Summary JSON
    summary_path = "outputs/multiagent_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Save Manifest JSON
    manifest_path = "outputs/multiagent_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return {
        "results_count": len(results_rows),
        "aggregated": aggregated,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Multi-Agent Method Comparison Evaluation")
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

    res = run_comparison(args.mode)
    print("Multi-Agent Method Comparison Evaluation completed successfully.")
    print(f"Results written to {res['jsonl_path']}, {res['summary_path']}, and {res['manifest_path']}")
    print("\nSample Aggregate Metrics:")
    # Print ReTrace overall accuracy vs Naive overall accuracy
    print("ReTrace Overall Accuracy:", res["aggregated"].get("ReTrace__overall__all", {}).get("authorization_accuracy"))
    print("Naive LWW Overall Accuracy:", res["aggregated"].get("Naive_LWW__overall__all", {}).get("authorization_accuracy"))


if __name__ == "__main__":
    main()
