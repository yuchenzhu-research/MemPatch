#!/usr/bin/env python3
"""Memora Oracle-Conditioned Authorization Diagnostic runner.

MEMORA ORACLE-CONDITIONED AUTHORIZATION DIAGNOSTIC ONLY —
NOT OFFICIAL END-TO-END MEMORA RESULT.

Candidate beliefs originate from Memora evaluation annotations
(memory_evidence / forgetting_evidence), not from end-to-end memory
extraction.  Do not interpret output as a paper result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

from retracemem.adapters.memora_oracle_diagnostic import (
    MemoraDiagnosticConfig,
    run_memora_oracle_diagnostic,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memora Oracle-Conditioned Authorization Diagnostic.",
    )
    parser.add_argument("--mode", choices=("replay", "live-dev"), default="replay")
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--period", default="weekly")
    parser.add_argument("--persona", default="academic_researcher")
    parser.add_argument("--all-personas", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=5)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument(
        "--stage-a-execution", choices=("per-belief", "batched"),
        default="batched",
    )
    parser.add_argument(
        "--stage-a-prompt-version",
        default="evidence_edge_prediction_batch_v1",
    )
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output-dir", default="outputs/memora_oracle_diag",
    )
    args = parser.parse_args()

    if args.mode == "live-dev" and not args.live_approved:
        raise SystemExit("Refusing live execution without --live-approved")

    config = MemoraDiagnosticConfig(
        mode=args.mode,
        reference_root=args.reference_root,
        period=args.period,
        persona=args.persona,
        all_personas=args.all_personas,
        limit_questions=args.limit_questions,
        provider=args.provider,
        model=args.model,
        stage_a_execution=args.stage_a_execution,
        stage_a_prompt_version=args.stage_a_prompt_version,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        http_timeout_seconds=args.http_timeout_seconds,
        output_dir=args.output_dir,
    )
    report_path, manifest_path, summary = run_memora_oracle_diagnostic(config)
    print(
        f"[DONE] report={report_path} manifest={manifest_path} "
        f"errors={summary['errors']} "
        f"calls={summary['calls']} tokens={summary['tokens']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
