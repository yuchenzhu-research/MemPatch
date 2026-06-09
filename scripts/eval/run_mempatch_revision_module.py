#!/usr/bin/env python3
"""Run the MemPatch Revision Module full method path and write predictions JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__, src=True)

from benchmark.api import load_scenarios  # noqa: E402
from benchmark.model_runner import call_model  # noqa: E402
from mempatch_learn.runtime.learned_proposer import (  # noqa: E402
    LearnedTypedRevisionProposer,
    ScriptedProposer,
    build_proposer_prompt,
)
from mempatch_learn.runtime.revision_module import run_revision_module_on_scenario  # noqa: E402
from mempatch_learn.runtime.scenario_revision import build_scenario_revision_view  # noqa: E402
from mempatch_learn.schemas import RevisionAction  # noqa: E402


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row.get("scenario_id")
                if sid:
                    done.add(str(sid))
    return done


def _load_scripted_actions(path: Path) -> list[RevisionAction]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--scripted-actions must be a JSON array of action objects")
    return [RevisionAction.from_dict(item) for item in payload]


def _build_proposer(args: argparse.Namespace) -> LearnedTypedRevisionProposer | ScriptedProposer | None:
    if args.policy == "noop":
        return None
    if args.policy == "scripted":
        if not args.scripted_actions:
            raise ValueError("--policy scripted requires --scripted-actions")
        actions = _load_scripted_actions(Path(args.scripted_actions))
        for action in actions:
            action.validate()
        return ScriptedProposer(actions)
    if args.policy == "prompt":
        if not args.provider or not args.model:
            raise ValueError("--policy prompt requires --provider and --model")

        def generate_fn(prompt: str) -> str:
            return call_model(
                provider=args.provider,
                model=args.model,
                prompt=prompt,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                json_mode=False,
                disable_thinking=args.disable_thinking,
            )

        return LearnedTypedRevisionProposer(generate_fn)
    raise ValueError(f"unsupported policy: {args.policy}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MemPatch Revision Module: scenario → view → policy → "
            "DPA projection → benchmark response."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="scenarios.jsonl file or directory containing scenarios.jsonl",
    )
    parser.add_argument(
        "--out-predictions",
        required=True,
        help="Path to write canonical predictions JSONL",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of new scenarios to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenario_ids already present in --out-predictions",
    )
    parser.add_argument(
        "--policy",
        choices=("noop", "scripted", "prompt"),
        default="noop",
        help="Revision Response Policy variant (default: noop smoke policy)",
    )
    parser.add_argument(
        "--scripted-actions",
        default=None,
        help="JSON file with a fixed action list for --policy scripted",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider for --policy prompt",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name for --policy prompt",
    )
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print-prompt-stats",
        action="store_true",
        help="Print proposer prompt size for the first planned case and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = load_scenarios(args.data)
    out_path = Path(args.out_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume and out_path.exists() else "w"
    planned = [s for s in scenarios if s["scenario_id"] not in done]
    if args.max_cases is not None:
        planned = planned[: args.max_cases]

    if args.print_prompt_stats:
        if planned:
            view = build_scenario_revision_view(planned[0])
            prompt = build_proposer_prompt(view)
            print(f"prompt_chars={len(prompt)}", flush=True)
            print(f"scenario_id={planned[0]['scenario_id']}", flush=True)
        else:
            print("prompt_chars=0", flush=True)
            print("scenario_id=", flush=True)
        return 0

    proposer = _build_proposer(args)
    print(
        f"MemPatch Revision Module runner | policy={args.policy} | "
        f"planned={len(planned)} | resume={args.resume}"
    )

    with out_path.open(mode, encoding="utf-8") as out_f:
        for scenario in planned:
            prediction = run_revision_module_on_scenario(scenario, proposer=proposer)
            out_f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"wrote {len(planned)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
