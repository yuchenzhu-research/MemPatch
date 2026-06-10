#!/usr/bin/env python3
"""Run one external memory baseline on HF CUDA and score with benchmark.api."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__)

from benchmark.api import evaluate_predictions, load_scenarios  # noqa: E402
from scripts.data.prepare_mempatch_v13_smoke import SYSTEM_PROMPT_V13_SMOKE  # noqa: E402
from scripts.linux.hf_inference import generate_from_messages, load_hf_model  # noqa: E402
from scripts.linux.run_hf_test_eval import prediction_from_output, write_jsonl  # noqa: E402
from scripts.memory.context_builders import BASELINE_IDS, build_baseline_prompt  # noqa: E402


def read_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["scenario_id"])
    return done


def run_baseline(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenarios = load_scenarios(args.eval_data)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    done_ids = read_done_ids(args.out_predictions) if args.resume else set()
    pending = [s for s in scenarios if s.get("scenario_id") not in done_ids]
    if not args.resume:
        args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.out_predictions.write_text("", encoding="utf-8")

    print(
        f"Baseline={args.baseline} model={args.model_id} pending={len(pending)}/{len(scenarios)}",
        file=sys.stderr,
    )
    model, tokenizer = load_hf_model(args.model_id, args.adapter_path)

    predictions: list[dict[str, Any]] = []
    for index, scenario in enumerate(pending, start=1):
        sid = str(scenario["scenario_id"])
        user_content = build_baseline_prompt(scenario, args.baseline, rag_top_k=args.rag_top_k)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V13_SMOKE},
            {"role": "user", "content": user_content},
        ]
        text, gen_meta = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )
        pred = prediction_from_output(
            sid,
            text,
            json_brace_prefill=bool(gen_meta.get("json_brace_prefill")),
        )
        pred["raw_output"] = text
        pred["gen_meta"] = gen_meta
        pred["baseline"] = args.baseline
        predictions.append(pred)
        with args.out_predictions.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + "\n")
        status = "ok" if pred.get("response") else "parse_error"
        print(f"[{index}/{len(pending)}] {sid}: {status}", file=sys.stderr)

    if args.resume and args.out_predictions.is_file():
        return [
            json.loads(line)
            for line in args.out_predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return predictions


def write_metrics(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if args.eval_data is None:
        return None
    scenario_ids = {p.get("scenario_id") for p in predictions}
    scenarios = [s for s in load_scenarios(args.eval_data) if s.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)

    run_tag = args.split_tag or f"baseline_{args.baseline}"
    result_dir = args.out_metrics.parent if args.out_metrics else args.out_predictions.parent
    from scripts.linux.eval_artifacts import write_eval_bundle

    write_eval_bundle(
        result_dir=result_dir,
        run_tag=run_tag,
        eval_result=result,
        predictions=predictions,
        run_meta={
            "backend": "hf",
            "baseline": args.baseline,
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "model_tag": args.model_tag,
            "variant_tag": args.baseline,
            "split_tag": run_tag,
            "max_tokens": args.max_tokens,
            "temp": args.temp,
            "eval_data": str(args.eval_data),
            "rag_top_k": args.rag_top_k,
        },
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINE_IDS)
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=root / "local/data/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rag-top-k", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--split-tag", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = run_baseline(args)
    if not args.resume:
        write_jsonl(args.out_predictions, predictions)
    result = write_metrics(args, predictions)
    if result is not None:
        print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
