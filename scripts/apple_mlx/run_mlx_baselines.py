#!/usr/bin/env python3
"""Run one frozen five-field baseline with a local MLX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.api import evaluate_predictions, load_scenarios
from scripts.data.prepare_mempatch_v13_smoke import SYSTEM_PROMPT_V13_SMOKE
from scripts.linux.eval_artifacts import write_eval_bundle
from scripts.linux.run_hf_test_eval import prediction_from_output
from scripts.memory.context_builders import BASELINE_DISPLAY_NAMES, BASELINE_IDS, build_baseline_prompt
from scripts.apple_mlx.mlx_inference import generate_from_messages, load_mlx_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINE_IDS)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rag-top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    scenarios = load_scenarios(args.eval_data)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    model, tokenizer = load_mlx_model(args.model)
    predictions = []
    for index, scenario in enumerate(scenarios, start=1):
        sid = str(scenario["scenario_id"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V13_SMOKE},
            {
                "role": "user",
                "content": build_baseline_prompt(
                    scenario, args.baseline, rag_top_k=args.rag_top_k
                ),
            },
        ]
        text, meta = generate_from_messages(
            model, tokenizer, messages, max_tokens=args.max_tokens
        )
        prediction = prediction_from_output(
            sid,
            text,
            json_brace_prefill=bool(meta.get("json_brace_prefill")),
        )
        prediction.update(raw_output=text, gen_meta=meta, baseline=args.baseline)
        predictions.append(prediction)
        print(f"[{index}/{len(scenarios)}] {sid}", file=sys.stderr, flush=True)

    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)
    tag = f"baseline_{args.baseline}"
    write_eval_bundle(
        result_dir=args.results_dir,
        run_tag=tag,
        eval_result=result,
        predictions=predictions,
        run_meta={
            "backend": "mlx",
            "baseline": args.baseline,
            "baseline_display_name": BASELINE_DISPLAY_NAMES[args.baseline],
            "model_id": str(args.model),
            "adapter_path": None,
            "model_tag": args.model_tag,
            "variant_tag": args.baseline,
            "split_tag": tag,
            "max_tokens": args.max_tokens,
            "temp": 0.0,
            "eval_data": str(args.eval_data),
            "rag_top_k": args.rag_top_k,
        },
    )
    print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
