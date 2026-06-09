#!/usr/bin/env python3
"""Run Path B LoRA on test split and score with benchmark.api."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__)

from benchmark.api import evaluate_predictions, load_scenarios  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_scenario_id(messages: list[dict[str, str]]) -> str:
    user = next((m for m in messages if m.get("role") == "user"), None)
    if user is None:
        raise ValueError("SFT row has no user message")
    payload = json.loads(user["content"])
    scenario_id = payload.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("SFT user payload has no scenario_id")
    return scenario_id


def strip_thinking(text: str) -> str:
    from scripts.mlx_support.mlx_chat_utils import strip_thinking as _strip

    return _strip(text)


def extract_json_object(text: str, *, json_brace_prefill: bool = False) -> dict[str, Any]:
    from scripts.mlx_support.mlx_chat_utils import extract_json_object as _extract

    return _extract(text, json_brace_prefill=json_brace_prefill)


def prediction_from_output(
    scenario_id: str,
    output: str,
    *,
    json_brace_prefill: bool = False,
) -> dict[str, Any]:
    try:
        response = extract_json_object(output, json_brace_prefill=json_brace_prefill)
        return {"scenario_id": scenario_id, "response": response}
    except ValueError as exc:
        return {
            "scenario_id": scenario_id,
            "response": {},
            "raw_output": output,
            "parse_error": str(exc),
        }


def prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], dict[str, Any]]:
    from scripts.mlx_support.mlx_chat_utils import apply_chat_template_no_think

    prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]
    return apply_chat_template_no_think(tokenizer, prompt_messages)


def run_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    rows = read_jsonl(args.data)
    selected = rows[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"Loading model: {args.model}", file=sys.stderr)
    if args.adapter_path is not None:
        print(f"Loading adapter: {args.adapter_path}", file=sys.stderr)
    else:
        print("Loading without adapter", file=sys.stderr)
    model, tokenizer = load(
        str(args.model),
        adapter_path=str(args.adapter_path) if args.adapter_path is not None else None,
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=args.temp)

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=args.offset + 1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{args.data}: row {index} has no messages list")
        scenario_id = extract_scenario_id(messages)
        tokens, gen_meta = prompt_tokens(tokenizer, messages)
        output = generate(
            model,
            tokenizer,
            tokens,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        pred = prediction_from_output(
            scenario_id,
            output,
            json_brace_prefill=bool(gen_meta.get("json_brace_prefill")),
        )
        predictions.append(pred)
        status = "ok" if pred.get("response") else "parse_error"
        print(f"[{len(predictions)}/{len(selected)}] {scenario_id}: {status}", file=sys.stderr)

    return predictions


def write_metrics(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> None:
    if args.eval_data is None or args.out_metrics is None:
        return
    scenario_ids = {p.get("scenario_id") for p in predictions}
    scenarios = [s for s in load_scenarios(args.eval_data) if s.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(
        scenarios,
        predictions,
        strict=False,
        allow_missing=True,
    )
    payload = {
        "path": getattr(args, "path_tag", "B"),
        "model": getattr(args, "model_tag", None),
        "variant": getattr(args, "variant_tag", None),
        "split": getattr(args, "split_tag", None),
        "count": result["count"],
        "headline_metrics": result["headline_metrics"],
        "auxiliary_metrics": result["auxiliary_metrics"],
        "all_metrics": result["all_metrics"],
        "warnings": result["warnings"],
        "errors": result["errors"],
        "missing_prediction_count": result["missing_prediction_count"],
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote metrics -> {args.out_metrics}", file=sys.stderr)
    print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True), file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "local/train_data/mempatch_v13_smoke/valid.jsonl",
        help="SFT JSONL with messages rows.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "local/models/Qwen3-14B-MLX-4bit",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=root / "local/adapters/qwen3_14b_mempatch_v13_smoke",
        help="Adapter directory. Use --no-adapter to evaluate the base model.",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base model without loading LoRA adapter weights.",
    )
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=root / "local/results/qwen3_14b_mempatch_v13_smoke_valid_predictions.jsonl",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=root / "hf_release/mempatch/test/scenarios.jsonl",
        help="Scenario JSONL used for scoring. Use --eval-data '' to skip metrics.",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=root / "local/results/qwen3_14b_mempatch_v13_smoke_valid_metrics.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--variant-tag", default=None)
    parser.add_argument("--split-tag", default=None)
    args = parser.parse_args(argv)
    if args.no_adapter:
        args.adapter_path = None
    if args.eval_data == Path(""):
        args.eval_data = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = run_predictions(args)
    write_jsonl(args.out_predictions, predictions)
    print(f"Wrote predictions -> {args.out_predictions}", file=sys.stderr)
    write_metrics(args, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
