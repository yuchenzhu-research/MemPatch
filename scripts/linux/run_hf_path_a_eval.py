#!/usr/bin/env python3
"""HF Path A inference: direct response + typed actions + DPA projection."""

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
from benchmark.public_view import public_scenario_view  # noqa: E402
from mempatch.revision.runtime.ablation_projection import (  # noqa: E402
    project_actions_without_dpa,
)
from mempatch.revision.runtime.dpa_runtime import parse_actions  # noqa: E402
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt  # noqa: E402
from mempatch.revision.runtime.revision_module import (  # noqa: E402
    run_revision_module_on_scenario,
)
from mempatch.revision.runtime.scenario_revision import (  # noqa: E402
    build_scenario_revision_view,
)
from scripts.linux.run_hf_test_eval import (  # noqa: E402
    extract_scenario_id,
    prediction_from_output,
    read_jsonl,
    write_jsonl,
)

ACTION_SYSTEM_PROMPT = (
    "You are the typed-action proposer for MemPatch Path A. "
    "Return only one JSON array. Copy all IDs exactly from the supplied view."
)


def _restore_array_prefill(text: str, gen_meta: dict[str, Any]) -> str:
    if gen_meta.get("json_prefill") == "[" and not text.lstrip().startswith("["):
        return "[" + text
    return text


def run_predictions(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.linux.hf_inference import generate_from_messages, load_hf_model

    sft_rows = read_jsonl(args.data)
    scenarios = load_scenarios(args.eval_data)
    scenario_by_id = {str(row["scenario_id"]): row for row in scenarios}
    if args.limit is not None:
        sft_rows = sft_rows[: args.limit]

    print(f"Loading model: {args.model_id}", file=sys.stderr)
    print(
        f"Loading adapter: {args.adapter_path}" if args.adapter_path else "Loading without adapter",
        file=sys.stderr,
    )
    model, tokenizer = load_hf_model(args.model_id, args.adapter_path)

    path_a_predictions: list[dict[str, Any]] = []
    no_dpa_predictions: list[dict[str, Any]] = []
    path_b_predictions: list[dict[str, Any]] = []
    for index, row in enumerate(sft_rows, start=1):
        messages = row["messages"]
        scenario_id = extract_scenario_id(messages)
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"{scenario_id}: missing from --eval-data")

        user = next(m for m in messages if m.get("role") == "user")
        public_view = json.loads(user["content"])
        raw_text, response_meta = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_tokens=args.max_response_tokens,
            temp=args.temp,
        )
        path_b = prediction_from_output(
            scenario_id,
            raw_text,
            json_brace_prefill=bool(response_meta.get("json_brace_prefill")),
            scenario_public_view=public_view,
        )
        path_b["raw_output"] = raw_text
        path_b["gen_meta"] = response_meta
        path_b_predictions.append(path_b)

        revision_view = build_scenario_revision_view(scenario)
        action_prompt = build_proposer_prompt(revision_view)
        action_text, action_meta = generate_from_messages(
            model,
            tokenizer,
            [
                {"role": "system", "content": ACTION_SYSTEM_PROMPT},
                {"role": "user", "content": action_prompt},
            ],
            max_tokens=args.max_action_tokens,
            temp=args.temp,
            json_prefill="[",
        )
        action_text = _restore_array_prefill(action_text, action_meta)
        parse_result = parse_actions(action_text)
        no_dpa_predictions.append(
            {
                "scenario_id": scenario_id,
                "response": project_actions_without_dpa(
                    view=revision_view,
                    parse_result=parse_result,
                    raw_response=path_b.get("response") or {},
                    scenario_public_view=public_scenario_view(scenario),
                ),
                "raw_actions_output": action_text,
                "parse_result": parse_result.to_dict(),
            }
        )

        prediction = run_revision_module_on_scenario(
            scenario,
            actions_text=action_text,
            raw_response=path_b.get("response") or {},
            include_audit=True,
        )
        prediction.update(
            {
                "path_b_response": path_b.get("response") or {},
                "raw_response_output": raw_text,
                "raw_actions_output": action_text,
                "response_gen_meta": response_meta,
                "action_gen_meta": action_meta,
            }
        )
        path_a_predictions.append(prediction)

        audit = prediction["dpa_audit"]
        parse_ok = audit["parse_result"]["schema_valid"]
        rejected = len(audit["rejected_actions"])
        print(
            f"[{index}/{len(sft_rows)}] {scenario_id}: "
            f"actions_parse={'ok' if parse_ok else 'error'} rejected={rejected}",
            file=sys.stderr,
        )

    return path_a_predictions, no_dpa_predictions, path_b_predictions


def write_metrics(
    args: argparse.Namespace,
    predictions: list[dict[str, Any]],
    no_dpa_predictions: list[dict[str, Any]],
    path_b_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario_ids = {p["scenario_id"] for p in predictions}
    scenarios = [s for s in load_scenarios(args.eval_data) if s.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)
    no_dpa_result = evaluate_predictions(
        scenarios,
        no_dpa_predictions,
        strict=False,
        allow_missing=True,
    )
    path_b_result = evaluate_predictions(
        scenarios,
        path_b_predictions,
        strict=False,
        allow_missing=True,
    )
    audits = [row["dpa_audit"] for row in predictions]
    parse_valid = sum(bool(a["parse_result"]["schema_valid"]) for a in audits)
    rejected_actions = sum(len(a["rejected_actions"]) for a in audits)
    admitted_actions = sum(len(a["admitted_actions"]) for a in audits)

    from scripts.linux.eval_artifacts import write_eval_bundle

    run_tag = args.split_tag or "path_a"
    paths = write_eval_bundle(
        result_dir=args.out_predictions.parent,
        run_tag=run_tag,
        eval_result=result,
        predictions=predictions,
        run_meta={
            "backend": "hf",
            "method_path": "path_a_typed_actions_dpa",
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "model_tag": args.model_tag,
            "variant_tag": args.variant_tag,
            "split_tag": args.split_tag,
            "temp": args.temp,
            "max_response_tokens": args.max_response_tokens,
            "max_action_tokens": args.max_action_tokens,
            "eval_data": str(args.eval_data),
            "sft_data": str(args.data),
            "typed_proposer_training": "multitask_sft",
            "action_parse_valid_rate": parse_valid / len(audits) if audits else 0.0,
            "admitted_action_count": admitted_actions,
            "rejected_action_count": rejected_actions,
            "paired_path_b_headline_metrics": path_b_result.get("headline_metrics"),
        },
    )
    for name, path in paths.items():
        print(f"Wrote {name} -> {path}", file=sys.stderr)
    no_dpa_paths = write_eval_bundle(
        result_dir=args.out_predictions.parent,
        run_tag=f"{run_tag}_no_dpa",
        eval_result=no_dpa_result,
        predictions=no_dpa_predictions,
        run_meta={
            "backend": "hf",
            "method_path": "path_a_typed_actions_no_dpa",
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "paired_dpa_run_tag": run_tag,
            "projection": "direct_action_mapping_without_gate_or_dpa",
        },
    )
    for name, path in no_dpa_paths.items():
        print(f"Wrote no-DPA {name} -> {path}", file=sys.stderr)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--max-action-tokens", type=int, default=512)
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
    predictions, no_dpa_predictions, path_b_predictions = run_predictions(args)
    write_jsonl(args.out_predictions, predictions)
    result = write_metrics(args, predictions, no_dpa_predictions, path_b_predictions)
    print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
