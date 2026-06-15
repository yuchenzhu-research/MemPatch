#!/usr/bin/env python3
"""Run frozen MemPatch Zero-Shot with a local MLX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.api import evaluate_predictions, load_scenarios
from benchmark.model_runner import build_prompt
from benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.dpa_runtime import parse_actions
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view
from scripts.data.prepare_mempatch_v13_smoke import SYSTEM_PROMPT_V13_SMOKE
from scripts.linux.eval_artifacts import write_eval_bundle
from scripts.linux.run_hf_path_a_eval import ACTION_SYSTEM_PROMPT
from scripts.linux.run_hf_test_eval import prediction_from_output
from scripts.apple_mlx.mlx_inference import generate_from_messages, load_mlx_model


def restore_prefill(text: str, meta: dict, prefix: str) -> str:
    return prefix + text if meta.get("json_prefill") == prefix and not text.lstrip().startswith(prefix) else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    scenarios = load_scenarios(args.eval_data)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    model, tokenizer = load_mlx_model(args.model)
    predictions = []
    no_dpa_predictions = []
    for index, scenario in enumerate(scenarios, start=1):
        sid = str(scenario["scenario_id"])
        raw_response, response_meta = generate_from_messages(
            model,
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT_V13_SMOKE},
                {"role": "user", "content": build_prompt(public_scenario_view(scenario))},
            ],
            max_tokens=args.max_response_tokens,
        )
        direct = prediction_from_output(
            sid,
            raw_response,
            json_brace_prefill=bool(response_meta.get("json_brace_prefill")),
            scenario_public_view=public_scenario_view(scenario),
            project_schema=False,
        )
        view = build_scenario_revision_view(scenario)
        raw_actions, action_meta = generate_from_messages(
            model,
            tokenizer,
            [
                {"role": "system", "content": ACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_proposer_prompt(view)},
            ],
            max_tokens=args.max_action_tokens,
            json_prefill="[",
        )
        raw_actions = restore_prefill(raw_actions, action_meta, "[")
        parse_result = parse_actions(raw_actions)
        no_dpa_predictions.append(
            {
                "scenario_id": sid,
                "response": project_actions_without_dpa(
                    view=view,
                    parse_result=parse_result,
                    raw_response=direct.get("response") or {},
                    scenario_public_view=public_scenario_view(scenario),
                ),
                "raw_actions_output": raw_actions,
                "parse_result": parse_result.to_dict(),
            }
        )
        prediction = run_revision_module_on_scenario(
            scenario,
            actions_text=raw_actions,
            raw_response=direct.get("response") or {},
            include_audit=True,
        )
        prediction.update(
            path_b_response=direct.get("response") or {},
            raw_response_output=raw_response,
            raw_actions_output=raw_actions,
            response_gen_meta=response_meta,
            action_gen_meta=action_meta,
        )
        predictions.append(prediction)
        valid = prediction["dpa_audit"]["parse_result"]["schema_valid"]
        print(f"[{index}/{len(scenarios)}] {sid}: actions_parse={'ok' if valid else 'error'}", file=sys.stderr, flush=True)

    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)
    no_dpa_result = evaluate_predictions(scenarios, no_dpa_predictions, strict=False, allow_missing=True)
    audits = [row["dpa_audit"] for row in predictions]
    tag = "test500_mempatch_zero_shot_base"
    write_eval_bundle(
        result_dir=args.results_dir,
        run_tag=tag,
        eval_result=result,
        predictions=predictions,
        run_meta={
            "backend": "mlx",
            "method_path": "path_a_typed_actions_dpa",
            "model_id": str(args.model),
            "adapter_path": None,
            "model_tag": args.model_tag,
            "variant_tag": "base",
            "split_tag": tag,
            "eval_data": str(args.eval_data),
            "action_parse_valid_rate": sum(bool(a["parse_result"]["schema_valid"]) for a in audits) / len(audits),
            "admitted_action_count": sum(len(a["admitted_actions"]) for a in audits),
            "rejected_action_count": sum(len(a["rejected_actions"]) for a in audits),
        },
    )
    write_eval_bundle(
        result_dir=args.results_dir,
        run_tag=f"{tag}_no_dpa",
        eval_result=no_dpa_result,
        predictions=no_dpa_predictions,
        run_meta={
            "backend": "mlx",
            "method_path": "path_a_typed_actions_no_dpa",
            "model_id": str(args.model),
            "adapter_path": None,
            "model_tag": args.model_tag,
            "variant_tag": "base",
            "paired_dpa_run_tag": tag,
        },
    )
    print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
