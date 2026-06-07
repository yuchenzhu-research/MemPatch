#!/usr/bin/env python3
"""Evaluate LongMemEval predictions with local heuristic + MLX judge."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_utils import (  # noqa: E402
    get_anscheck_prompt,
    heuristic_label,
    load_longmemeval,
    parse_yes_no,
    read_jsonl,
)


def judge_with_mlx(
    model: Any,
    tokenizer: Any,
    sampler: Any,
    prompt: str,
    *,
    max_tokens: int,
) -> str:
    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()


def aggregate(labels: list[int]) -> float | None:
    if not labels:
        return None
    return round(sum(labels) / len(labels), 4)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    references = load_longmemeval(args.ref)
    ref_by_id = {row["question_id"]: row for row in references}
    hypotheses = read_jsonl(args.hyp)

    model = tokenizer = sampler = None
    if not args.heuristic_only:
        from mlx_lm import load
        from mlx_lm.generate import make_sampler

        print(f"Loading judge model: {args.model}", file=sys.stderr)
        model, tokenizer = load(
            str(args.model),
            tokenizer_config={"trust_remote_code": True},
        )
        sampler = make_sampler(temp=0.0)

    logs: list[dict[str, Any]] = []
    heuristic_by_type: dict[str, list[int]] = defaultdict(list)
    judge_by_type: dict[str, list[int]] = defaultdict(list)

    for index, entry in enumerate(hypotheses, start=1):
        question_id = entry["question_id"]
        ref = ref_by_id.get(question_id)
        if ref is None:
            print(f"skip missing ref: {question_id}", file=sys.stderr)
            continue

        question_type = ref["question_type"]
        question = ref["question"]
        answer = ref["answer"]
        hypothesis = entry.get("hypothesis", "")
        abstention = question_id.endswith("_abs")

        heur = heuristic_label(question_type, question, answer, hypothesis)
        heuristic_by_type[question_type].append(1 if heur else 0)

        judge_label: bool | None = None
        judge_raw: str | None = None
        if not args.heuristic_only:
            prompt = get_anscheck_prompt(
                question_type,
                question,
                answer,
                hypothesis,
                abstention=abstention,
            )
            judge_raw = judge_with_mlx(
                model,
                tokenizer,
                sampler,
                prompt,
                max_tokens=args.judge_max_tokens,
            )
            judge_label = parse_yes_no(judge_raw)
            judge_by_type[question_type].append(1 if judge_label else 0)

        log_row = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "hypothesis": hypothesis,
            "heuristic_label": heur,
            "judge_label": judge_label,
            "judge_raw": judge_raw,
        }
        logs.append(log_row)
        print(
            f"[{index}/{len(hypotheses)}] {question_id} "
            f"heur={heur} judge={judge_label}",
            file=sys.stderr,
        )

    metrics = {
        "count": len(logs),
        "heuristic_accuracy": aggregate([row["heuristic_label"] for row in logs]),
        "judge_accuracy": aggregate(
            [1 if row["judge_label"] else 0 for row in logs if row["judge_label"] is not None]
        ),
        "heuristic_by_type": {
            key: aggregate(values) for key, values in sorted(heuristic_by_type.items())
        },
        "judge_by_type": {
            key: aggregate(values) for key, values in sorted(judge_by_type.items())
        },
        "note": (
            "Local MLX judge is for development only; not comparable to official GPT-4o metrics."
            if not args.heuristic_only
            else "Heuristic-only evaluation."
        ),
    }

    args.out_log.parent.mkdir(parents=True, exist_ok=True)
    with args.out_log.open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote log -> {args.out_log}", file=sys.stderr)
    print(f"Wrote metrics -> {args.out_metrics}", file=sys.stderr)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hyp", type=Path, help="Predictions JSONL with question_id/hypothesis.")
    parser.add_argument(
        "--ref",
        type=Path,
        default=root / "local/longmemeval/data/longmemeval_oracle.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "local/models/Qwen3-14B-MLX-4bit",
    )
    parser.add_argument(
        "--out-log",
        type=Path,
        default=root / "local/longmemeval/results/longmemeval_eval.log.jsonl",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=root / "local/longmemeval/results/longmemeval_eval_metrics.json",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=8)
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip MLX judge and only compute heuristic accuracy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
