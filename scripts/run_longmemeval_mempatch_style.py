#!/usr/bin/env python3
"""Run MemPatch-style LongMemEval baselines: convert → revision module → QA answer."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_mempatch_convert import (  # noqa: E402
    build_answer_messages,
    build_ku_supersede_actions_text,
    convert_knowledge_update,
)
from longmemeval_utils import filter_examples, load_longmemeval, write_jsonl  # noqa: E402
from retrace_learn.runtime.learned_proposer import build_proposer_prompt  # noqa: E402
from retrace_learn.runtime.revision_module import run_revision_module_on_scenario  # noqa: E402
from retrace_learn.runtime.scenario_revision import build_scenario_revision_view  # noqa: E402


def strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def mlx_generate(model: Any, tokenizer: Any, sampler: Any, messages: list[dict[str, str]], *, max_tokens: int) -> str:
    from mlx_lm import generate

    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return strip_thinking(
        generate(
            model,
            tokenizer,
            tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
    ).strip()


def resolve_actions_text(
    scenario: dict[str, Any],
    *,
    policy: str,
    generate_fn: Any | None,
) -> tuple[str, dict[str, Any]]:
    view = build_scenario_revision_view(scenario)
    meta: dict[str, Any] = {"policy": policy}

    if policy == "ku-heuristic":
        actions_text = build_ku_supersede_actions_text(view)
        if actions_text is None:
            actions_text = json.dumps(
                [
                    {
                        "action_type": "NO_REVISION",
                        "target_belief_id": None,
                        "target_condition_id": None,
                        "replacement_belief_id": None,
                        "evidence_ids": [view.new_evidence.evidence_id],
                        "rationale": "no replacement candidate found",
                    }
                ],
                ensure_ascii=False,
            )
            meta["fallback"] = "no_revision"
        return actions_text, meta

    if policy == "mlx-proposer":
        if generate_fn is None:
            raise ValueError("mlx-proposer requires a loaded MLX model")
        prompt = build_proposer_prompt(view)
        raw_text = generate_fn(prompt)
        meta["proposer_raw"] = raw_text[:500]
        return raw_text, meta

    raise ValueError(f"unsupported policy: {policy}")


def run_one(
    example: dict[str, Any],
    *,
    policy: str,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    answer_max_tokens: int,
    proposer_max_tokens: int,
) -> dict[str, Any]:
    scenario = convert_knowledge_update(example)
    view = build_scenario_revision_view(scenario)

    def proposer_generate(prompt: str) -> str:
        return mlx_generate(
            model,
            tokenizer,
            sampler,
            [{"role": "user", "content": prompt}],
            max_tokens=proposer_max_tokens,
        )

    generate_fn = proposer_generate if policy == "mlx-proposer" else None
    actions_text, action_meta = resolve_actions_text(scenario, policy=policy, generate_fn=generate_fn)

    module_out = run_revision_module_on_scenario(scenario, actions_text=actions_text)
    response = module_out["response"]
    question = str(example.get("question") or "")
    answer_messages = build_answer_messages(question, view, response)
    hypothesis = mlx_generate(
        model,
        tokenizer,
        sampler,
        answer_messages,
        max_tokens=answer_max_tokens,
    )

    return {
        "question_id": example["question_id"],
        "hypothesis": hypothesis,
        "question_type": example.get("question_type"),
        "policy": f"mempatch-{policy}",
        "mempatch_response": response,
        "mempatch_action_meta": action_meta,
    }


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mlx_lm import load
    from mlx_lm.generate import make_sampler

    examples = filter_examples(
        load_longmemeval(args.data),
        question_type=args.filter_type,
        limit=args.limit,
        offset=args.offset,
    )
    if not examples:
        raise SystemExit("No examples selected.")

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load(
        str(args.model),
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=args.temp)

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        if example.get("question_id", "").endswith("_abs"):
            print(f"[{index}/{len(examples)}] skip abstention {example['question_id']}", file=sys.stderr)
            continue
        try:
            row = run_one(
                example,
                policy=args.policy,
                model=model,
                tokenizer=tokenizer,
                sampler=sampler,
                answer_max_tokens=args.answer_max_tokens,
                proposer_max_tokens=args.proposer_max_tokens,
            )
        except ValueError as exc:
            print(f"[{index}/{len(examples)}] skip {example.get('question_id')}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        print(
            f"[{index}/{len(examples)}] {row['question_id']} "
            f"decision={row['mempatch_response'].get('decision')}",
            file=sys.stderr,
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "local/longmemeval/data/longmemeval_oracle.json",
    )
    parser.add_argument(
        "--filter-type",
        default="knowledge-update",
        help="LongMemEval question_type filter (default: knowledge-update)",
    )
    parser.add_argument(
        "--policy",
        choices=("ku-heuristic", "mlx-proposer"),
        default="ku-heuristic",
        help="Revision policy: deterministic KU supersede, or MLX typed-action proposer",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "local/models/Qwen3-14B-MLX-4bit",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "local/longmemeval/results/lme_mempatch_ku_oracle78.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--proposer-max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_batch(args)
    slim_rows = [
        {
            "question_id": row["question_id"],
            "hypothesis": row["hypothesis"],
            "question_type": row.get("question_type"),
            "policy": row.get("policy"),
        }
        for row in rows
    ]
    write_jsonl(args.out, slim_rows)
    detail_path = args.out.with_suffix(".detail.jsonl")
    write_jsonl(detail_path, rows)
    print(f"Wrote {len(rows)} predictions -> {args.out}", file=sys.stderr)
    print(f"Wrote detail log -> {detail_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
