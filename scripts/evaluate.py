#!/usr/bin/env python3
"""Public evaluation entrypoint for ReTrace Paper 1.

Thin CLI over the shared multi-agent evaluation engine
(`retracemem.evaluation.multiagent`). All three method families plug into the
same pipeline:

    python3 scripts/evaluate.py stage-a --mock        # ReTrace-API-ZeroShot (typed -> gate -> DPA)
    python3 scripts/evaluate.py stage-b --mock        # DirectJudge-API baseline (final status only)
    python3 scripts/evaluate.py stage-c --smoke       # ReTrace-AdaptiveProposer (adaptive typed proposer)

Stage A and Stage B are evaluated *jointly* on the same fixed-candidate cases
for a fair comparison (experiment E1), so `stage-a` and `stage-b` are entry
aliases into the same A/B runner, which always reports both. `stage-c` runs the
adaptive typed-proposer evaluation over the identical commit / DPA path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from retracemem.evaluation.multiagent.config import EvalRunConfig
from retracemem.evaluation.multiagent.runner import run_stageab_eval
from retracemem.evaluation.multiagent.stagec import StageCEvalConfig, run_stagec_eval
from retracemem.proposers.replay import CANONICAL_ACTIONS

load_dotenv(REPO_ROOT / ".env", override=False)


def _add_stageab_args(p: argparse.ArgumentParser) -> None:
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Enable live API calls.")
    mode.add_argument("--dry-run", action="store_true", help="Dry run prompt/dataset checking.")
    mode.add_argument("--mock", action="store_true", help="Offline mock replay mode.")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases.")
    p.add_argument("--resume", action="store_true", help="Resume from cached files.")
    p.add_argument("--provider", default="siliconflow", help="API provider name.")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Model ID.")
    p.add_argument("--api-key", default=None, help="Explicit API key.")
    p.add_argument("--base-url", default=None, help="Explicit base URL.")
    p.add_argument("--output-dir", default="outputs/runs/stageab_dev70", help="Output directory.")
    p.add_argument("--constrained", action="store_true", help="Use constrained zero-shot proposer.")
    p.add_argument(
        "--stage-a-variant",
        choices=("default", "conflict_aware"),
        default="default",
        help="Constrained Stage A policy variant ('conflict_aware' adds conflict-handling affordances; requires --constrained).",
    )
    p.add_argument("--diagnostic", action="store_true", help="Enable diagnostic decision audit.")
    p.add_argument("--repair-on-parse-error", action="store_true", help="Enable parse-error repair rounds.")
    p.add_argument("--max-repair-rounds", type=int, default=0, help="Maximum number of repair rounds.")


def _run_stageab(args: argparse.Namespace) -> None:
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
        output_dir=args.output_dir,
        constrained=args.constrained,
        stage_a_variant=args.stage_a_variant,
        diagnostic=args.diagnostic,
        method=None,
        repair_on_parse_error=args.repair_on_parse_error,
        max_repair_rounds=args.max_repair_rounds,
    )
    run_stageab_eval(config)


def _add_stagec_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group()
    src.add_argument("--generations-dir", default=None, help="Directory of decoded '{submission_id}.txt' generations.")
    src.add_argument("--mock", action="store_true", help="Offline NO_REVISION generations (no files, no API).")
    p.add_argument("--smoke", action="store_true", help="Smoke mode: offline mock generations on the first few cases.")
    p.add_argument("--generation-suffix", default=".txt", help="Generation filename suffix.")
    p.add_argument("--policy-variant", default="adapter_replay", help="Proposer policy variant label.")
    p.add_argument("--backbone-model", default=None, help="Backbone model id for provenance.")
    p.add_argument("--checkpoint-id", default=None, help="Adapter/SFT checkpoint id for provenance.")
    p.add_argument("--allowed-actions", default=None, help="Comma-separated subset of the canonical action vocabulary.")
    p.add_argument("--no-constrained-postvalidation", action="store_true", help="Record constrained post-validation disabled.")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases.")
    p.add_argument("--output-dir", default="outputs/runs/stagec_dev70", help="Output directory for Stage C artifacts.")


def _run_stagec(args: argparse.Namespace) -> None:
    if args.allowed_actions:
        allowed_actions = tuple(a.strip() for a in args.allowed_actions.split(",") if a.strip())
    else:
        allowed_actions = CANONICAL_ACTIONS
    use_mock = args.mock or (args.smoke and not args.generations_dir)
    config = StageCEvalConfig(
        proposer_source="mock" if use_mock else "replay",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ReTrace Paper 1 evaluation entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    # Stage A and Stage B share the joint A-vs-B runner.
    for name in ("stage-a", "stage-b", "stage-ab"):
        sp = sub.add_parser(name, help="Run the joint Stage A (typed) vs Stage B (DirectJudge) evaluation.")
        _add_stageab_args(sp)
        sp.set_defaults(func=_run_stageab)

    sc = sub.add_parser("stage-c", help="Run the Stage C ReTrace-AdaptiveProposer evaluation.")
    _add_stagec_args(sc)
    sc.set_defaults(func=_run_stagec)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
