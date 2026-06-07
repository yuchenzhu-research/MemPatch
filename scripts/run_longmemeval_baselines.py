#!/usr/bin/env python3
"""Run LongMemEval QA baselines with a local MLX model.

Policies:
  direct        — feed the selected history and question directly
  retrieve-all  — alias of direct (all haystack sessions)
  latest-only   — keep only the last N haystack sessions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_utils import (  # noqa: E402
    POLICIES,
    build_chat_messages,
    filter_examples,
    load_longmemeval,
    write_jsonl,
)


def prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    examples = filter_examples(
        load_longmemeval(args.data),
        question_type=args.filter_type,
        limit=args.limit,
        offset=args.offset,
    )
    if not examples:
        raise SystemExit("No examples selected; check --filter-type/--limit/--offset.")

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load(
        str(args.model),
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=args.temp)

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        question_id = example["question_id"]
        messages = build_chat_messages(
            example,
            policy=args.policy,
            latest_sessions=args.latest_sessions,
            user_only=args.user_only,
        )
        tokens = prompt_tokens(tokenizer, messages)
        output = generate(
            model,
            tokenizer,
            tokens,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        hypothesis = output.strip()
        row = {
            "question_id": question_id,
            "hypothesis": hypothesis,
            "question_type": example.get("question_type"),
            "policy": args.policy,
            "latest_sessions": args.latest_sessions if args.policy == "latest-only" else None,
            "prompt_tokens": len(tokens),
        }
        rows.append(row)
        print(
            f"[{index}/{len(examples)}] {question_id} "
            f"({example.get('question_type')}) prompt_tokens={len(tokens)}",
            file=sys.stderr,
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "local/longmemeval/data/longmemeval_oracle.json",
    )
    parser.add_argument("--policy", choices=POLICIES, default="direct")
    parser.add_argument("--latest-sessions", type=int, default=3)
    parser.add_argument(
        "--filter-type",
        default=None,
        help="Optional question_type filter, e.g. knowledge-update",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "local/models/Qwen3-14B-MLX-4bit",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "local/longmemeval/results/lme_direct_oracle.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--user-only",
        action="store_true",
        help="Drop assistant turns from the history block.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_predictions(args)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} predictions -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
