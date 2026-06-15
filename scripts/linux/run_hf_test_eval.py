#!/usr/bin/env python3
"""HF Path B direct-response inference + benchmark.api scoring (no DPA)."""

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
from scripts.linux.hf_chat_utils import extract_json_object
from scripts.linux.response_schema_projection import project_response_schema


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_scenario_id(messages: list[dict[str, str]]) -> str:
    user = next(m for m in messages if m.get("role") == "user")
    payload = json.loads(user["content"])
    scenario_id = payload.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("SFT user payload has no scenario_id")
    return scenario_id


def prediction_from_output(
    scenario_id: str,
    output: str,
    *,
    json_brace_prefill: bool = False,
    scenario_public_view: dict[str, Any] | None = None,
    project_schema: bool = True,
) -> dict[str, Any]:
    try:
        response = extract_json_object(output, json_brace_prefill=json_brace_prefill)
        prediction = {"scenario_id": scenario_id, "response": response}
    except ValueError as exc:
        prediction = {
            "scenario_id": scenario_id,
            "response": {},
            "raw_output": output,
            "parse_error": str(exc),
        }
    if scenario_public_view is not None and project_schema:
        prediction["raw_response"] = prediction["response"]
        prediction["response"], prediction["schema_repairs"] = project_response_schema(
            prediction["response"], scenario_public_view
        )
    return prediction


def run_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    from scripts.linux.hf_inference import generate_from_messages, load_hf_model

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loading model: {args.model_id}", file=sys.stderr)
    if args.adapter_path is not None:
        print(f"Loading adapter: {args.adapter_path}", file=sys.stderr)
    else:
        print("Loading without adapter", file=sys.stderr)
    model, tokenizer = load_hf_model(args.model_id, args.adapter_path)

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        scenario_id = extract_scenario_id(messages)
        user = next(m for m in messages if m.get("role") == "user")
        scenario_public_view = json.loads(user["content"])
        text, gen_meta = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )
        pred = prediction_from_output(
            scenario_id,
            text,
            json_brace_prefill=bool(gen_meta.get("json_brace_prefill")),
            scenario_public_view=scenario_public_view,
            project_schema=not args.no_schema_projection,
        )
        pred["raw_output"] = text
        pred["gen_meta"] = gen_meta
        predictions.append(pred)
        status = "ok" if pred.get("response") else "parse_error"
        print(f"[{index}/{len(rows)}] {scenario_id}: {status}", file=sys.stderr)

    return predictions


def write_metrics(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if args.eval_data is None:
        return None
    scenario_ids = {p.get("scenario_id") for p in predictions}
    scenarios = [s for s in load_scenarios(args.eval_data) if s.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)
    raw_predictions = [
        {
            "scenario_id": row.get("scenario_id"),
            "response": row.get("raw_response", row.get("response", {})),
        }
        for row in predictions
    ]
    raw_result = evaluate_predictions(
        scenarios,
        raw_predictions,
        strict=False,
        allow_missing=True,
    )

    run_tag = args.split_tag or "eval"
    result_dir = args.out_metrics.parent if args.out_metrics is not None else args.out_predictions.parent
    from scripts.linux.eval_artifacts import write_eval_bundle

    paths = write_eval_bundle(
        result_dir=result_dir,
        run_tag=run_tag,
        eval_result=result,
        predictions=predictions,
        run_meta={
            "backend": "hf",
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "model_tag": args.model_tag,
            "variant_tag": args.variant_tag,
            "split_tag": args.split_tag,
            "max_tokens": args.max_tokens,
            "temp": args.temp,
            "eval_data": str(args.eval_data),
            "sft_data": str(args.data),
            "schema_projection": (
                "disabled" if args.no_schema_projection else "public_only_v1"
            ),
            "raw_response_schema_compliance_rate": raw_result["headline_metrics"].get(
                "response_schema_compliance_rate"
            ),
            "raw_validation_error_count": len(raw_result.get("errors") or []),
            "projected_repair_row_count": sum(
                bool(row.get("schema_repairs")) for row in predictions
            ),
        },
    )
    for name, path in paths.items():
        print(f"Wrote {name} -> {path}", file=sys.stderr)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=root / "local/data/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument("--out-metrics", type=Path, default=None)
    parser.add_argument(
        "--no-schema-projection",
        action="store_true",
        help="Score parsed model output as-is; do not fill or repair schema fields.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--variant-tag", default=None)
    parser.add_argument("--split-tag", default=None)
    args = parser.parse_args(argv)
    if args.no_adapter:
        args.adapter_path = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = run_predictions(args)
    write_jsonl(args.out_predictions, predictions)
    print(f"Wrote predictions -> {args.out_predictions}", file=sys.stderr)
    result = write_metrics(args, predictions)
    if result is not None:
        print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
