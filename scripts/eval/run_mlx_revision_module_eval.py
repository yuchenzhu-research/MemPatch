#!/usr/bin/env python3
"""Path A: MLX typed-action proposer + DPA + benchmark projection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__, src=True)

from benchmark.api import evaluate_predictions, load_scenarios  # noqa: E402
from mempatch_learn.runtime.learned_proposer import LearnedTypedRevisionProposer  # noqa: E402
from mempatch_learn.runtime.revision_module import run_revision_module_on_scenario  # noqa: E402


def strip_thinking(text: str) -> str:
    from scripts.mlx.mlx_chat_utils import strip_thinking as _strip

    return _strip(text)


def extract_json_array(text: str) -> str:
    text = strip_thinking(text)
    start = text.find("[")
    if start == -1:
        raise ValueError(f"no JSON array found in model output: {text[:300]!r}")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text[start:])
    return text[start : start + end]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    scenarios = load_scenarios(args.data)
    if args.limit is not None:
        scenarios = scenarios[args.offset : args.offset + args.limit]

    print(f"Loading model: {args.model}", file=sys.stderr)
    if args.adapter_path is not None:
        print(f"Loading adapter: {args.adapter_path}", file=sys.stderr)
    model, tokenizer = load(
        str(args.model),
        adapter_path=str(args.adapter_path) if args.adapter_path is not None else None,
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=args.temp)

    def generate_fn(prompt: str) -> str:
        from scripts.mlx.mlx_chat_utils import apply_chat_template_no_think, normalize_generation_text

        messages = [{"role": "user", "content": prompt}]
        tokens, gen_meta = apply_chat_template_no_think(tokenizer, messages)
        raw = generate(
            model,
            tokenizer,
            tokens,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        raw = normalize_generation_text(
            raw,
            json_brace_prefill=bool(gen_meta.get("json_brace_prefill")),
        )
        try:
            return extract_json_array(raw)
        except ValueError:
            return raw

    proposer = LearnedTypedRevisionProposer(generate_fn)
    predictions: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        prediction = run_revision_module_on_scenario(scenario, proposer=proposer)
        predictions.append(prediction)
        parse_ok = bool((prediction.get("response") or {}).get("decision"))
        print(
            f"[{index}/{len(scenarios)}] {scenario['scenario_id']}: "
            f"{'ok' if parse_ok else 'weak'}",
            file=sys.stderr,
        )
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
        "path": "A",
        "model": getattr(args, "model_tag", None),
        "variant": getattr(args, "variant_tag", "base"),
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
    parser.add_argument("--data", type=Path, required=True, help="Scenario JSONL slice.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, default=None)
    parser.add_argument("--out-metrics", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
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
