#!/usr/bin/env python3
"""Run a small ReTrace-Bench general-schema ablation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


OFFLINE_BASELINES = (
    "latest_only",
    "retrieve_all",
    "rag_lexical",
    "crud_memory",
    "mem0_style",
    "retrace_oracle_engine",
    "heuristic_memory_state",
)


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/retrace_bench/sample_40_en/scenarios.jsonl")
    parser.add_argument("--out-dir", default="outputs/retrace_bench/ablation_sample40")
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    baselines = list(OFFLINE_BASELINES)
    if args.include_llm:
        baselines.append("llm_json_answerer")

    for baseline in baselines:
        out = out_dir / f"{baseline}.jsonl"
        cmd = [
            sys.executable,
            "scripts/run_retrace_bench_baseline.py",
            "--data",
            args.data,
            "--baseline",
            baseline,
            "--max-cases",
            str(args.max_cases),
            "--out",
            str(out),
        ]
        if baseline == "llm_json_answerer":
            cmd.extend(
                [
                    "--provider",
                    args.provider,
                    "--model",
                    args.model,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--progress",
                    "--append",
                ]
            )
            if args.disable_thinking:
                cmd.append("--disable-thinking")
        run_cmd(cmd)
        metrics = load_metrics(out.with_suffix(".metrics.json"))
        rows.append(
            {
                "baseline": baseline,
                "black_box_decision_accuracy": metrics.get("black_box_decision_accuracy", metrics.get("decision_accuracy", 0.0)),
                "answer_key_fact_accuracy": metrics.get("answer_key_fact_accuracy", 0.0),
                "decision_accuracy": metrics.get("decision_accuracy", 0.0),
                "memory_state_accuracy": metrics.get("memory_state_accuracy", 0.0),
                "evidence_f1": metrics.get("evidence_f1", 0.0),
                "failure_diagnosis_accuracy": metrics.get("failure_diagnosis_accuracy", 0.0),
                "stale_reuse_rate": metrics.get("stale_reuse_rate", 0.0),
            }
        )

    summary = {"data": args.data, "max_cases": args.max_cases, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nBaseline matrix")
    print("baseline,black_box_decision,answer_key_fact,memory_state,evidence_f1,diagnosis,stale_reuse")
    for row in rows:
        print(
            f"{row['baseline']},{row['black_box_decision_accuracy']:.3f},"
            f"{row['answer_key_fact_accuracy']:.3f},{row['memory_state_accuracy']:.3f},"
            f"{row['evidence_f1']:.3f},"
            f"{row['failure_diagnosis_accuracy']:.3f},{row['stale_reuse_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
