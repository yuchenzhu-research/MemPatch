#!/usr/bin/env python3
"""Run MemPatch-Bench with shared-LLM external memory baselines.

Backends (all use the same MLX answer model and ``build_prompt`` schema):

- ``base`` — task-only, no historical memory or evidence
- ``full`` — all visible memory + event trace (strong context baseline)
- ``rag`` — lexical top-k event retrieval over the trace
- ``mem0`` — Mem0 OSS ingest + search (local HF/Ollama embedder + Chroma; no OpenAI)

Example::

  PYTHONPATH=.:src .venv/bin/python scripts/run_mempatch_memory_baselines.py \\
    --data hf_release/mempatch/test/scenarios.jsonl \\
    --backend rag --limit 50 \\
    --model local/models/Meta-Llama-3.1-8B-Instruct-4bit \\
    --out-predictions local/results/memory_baselines/llama8b_rag_test50.jsonl \\
    --out-metrics local/results/memory_baselines/llama8b_rag_test50_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.api import evaluate_predictions, load_scenarios  # noqa: E402
from benchmark.model_runner import (  # noqa: E402
    build_prompt,
    canonical_response,
    extract_json_object,
)
from mempatch_memory_context import BACKENDS, build_baseline_view  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prediction_from_output(scenario_id: str, output: str, *, json_brace_prefill: bool = False) -> dict[str, Any]:
    from mlx_chat_utils import extract_json_object as mlx_extract_json_object

    try:
        response = mlx_extract_json_object(output, json_brace_prefill=json_brace_prefill)
        return {"scenario_id": scenario_id, "response": canonical_response(response)}
    except ValueError as exc:
        try:
            response = extract_json_object(output)
            return {"scenario_id": scenario_id, "response": canonical_response(response)}
        except (ValueError, json.JSONDecodeError):
            return {
                "scenario_id": scenario_id,
                "response": {},
                "raw_output": output,
                "parse_error": str(exc),
            }


def run_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    from mlx_chat_utils import apply_chat_template_no_think, normalize_generation_text

    scenarios = load_scenarios(args.data)
    if args.limit is not None:
        scenarios = scenarios[args.offset : args.offset + args.limit]

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load(
        str(args.model),
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=args.temp)

    predictions: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        view = build_baseline_view(
            scenario,
            backend=args.backend,
            rag_top_k=args.rag_top_k,
            mem0_top_k=args.mem0_top_k,
            mem0_infer=args.mem0_infer,
            mem0_embedder=args.mem0_embedder,
            mem0_embed_model=args.mem0_embed_model,
            mem0_ollama_base_url=args.mem0_ollama_base_url,
            mem0_llm_model=args.mem0_llm_model,
        )
        prompt = build_prompt(view)
        if args.print_prompt_stats and index == 1:
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "prompt_chars": len(prompt),
                        "event_count": len(view.get("public_input", {}).get("event_trace") or []),
                        "memory_count": len(view.get("public_input", {}).get("initial_memory") or []),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return []

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
        raw = normalize_generation_text(raw, json_brace_prefill=bool(gen_meta.get("json_brace_prefill")))
        pred = prediction_from_output(
            scenario["scenario_id"],
            raw,
            json_brace_prefill=bool(gen_meta.get("json_brace_prefill")),
        )
        if args.backend != "full":
            pred["memory_baseline"] = view.get("memory_baseline")
        predictions.append(pred)
        status = "ok" if (pred.get("response") or {}).get("decision") else "weak"
        print(f"[{index}/{len(scenarios)}] {scenario['scenario_id']}: {status}", file=sys.stderr)

    return predictions


def write_metrics(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> None:
    if args.out_metrics is None:
        return
    scenario_ids = {p.get("scenario_id") for p in predictions}
    scenarios = [s for s in load_scenarios(args.data) if s.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(
        scenarios,
        predictions,
        strict=False,
        allow_missing=True,
    )
    payload = {
        "runner": "memory_baselines",
        "backend": args.backend,
        "model": str(args.model),
        "data": str(args.data),
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
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "hf_release/mempatch/test/scenarios.jsonl",
        help="scenarios.jsonl file or directory containing scenarios.jsonl",
    )
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "local/models/Meta-Llama-3.1-8B-Instruct-4bit",
        help="Shared MLX answer model for all baselines",
    )
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--mem0-top-k", type=int, default=5)
    parser.add_argument(
        "--mem0-infer",
        action="store_true",
        help="Let Mem0 infer structured memories (uses local Ollama LLM, not OpenAI).",
    )
    parser.add_argument(
        "--mem0-embedder",
        choices=("huggingface", "ollama"),
        default="huggingface",
        help="Local Mem0 embedder provider (default: huggingface, no API key).",
    )
    parser.add_argument(
        "--mem0-embed-model",
        default=None,
        help="Override embedder model (HF sentence-transformers or Ollama embed model).",
    )
    parser.add_argument(
        "--mem0-ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL when --mem0-embedder=ollama or --mem0-infer.",
    )
    parser.add_argument(
        "--mem0-llm-model",
        default=None,
        help="Ollama chat model for Mem0 infer=True (answer model stays --model MLX).",
    )
    parser.add_argument(
        "--print-prompt-stats",
        action="store_true",
        help="Print first-case prompt stats and exit without model inference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = run_predictions(args)
    if args.print_prompt_stats:
        return 0
    write_jsonl(args.out_predictions, predictions)
    print(f"Wrote predictions -> {args.out_predictions}", file=sys.stderr)
    write_metrics(args, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
