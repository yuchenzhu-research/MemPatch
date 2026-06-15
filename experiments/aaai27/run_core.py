"""Unified, resumable server runner for the AAAI-27 MemPatch experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.api import load_scenarios
from benchmark.model_runner import build_prompt, canonical_response, extract_json_object
from benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.dpa_runtime import parse_actions
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view

try:
    from .methods import build_method_view
except ImportError:
    from methods import build_method_view

BASELINE_METHODS = (
    "frozen_direct",
    "full_context",
    "lexical_rag",
    "time_aware_rag",
    "summary_memory",
)
ALL_METHODS = BASELINE_METHODS + ("mempatch", "mempatch_no_guard")


@dataclass(frozen=True)
class GenerationRecord:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class LocalHFBackend:
    def __init__(self, model_id: str, dtype: str, trust_remote_code: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        dtype_value = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_value,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def _format_chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Return only the requested JSON. Do not use Markdown."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationRecord:
        rendered = self._format_chat(prompt)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        input_tokens = int(encoded["input_ids"].shape[-1])
        generated = output[0, input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationRecord(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            latency_seconds=latency,
        )


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_response(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        return canonical_response(extract_json_object(text)), None
    except Exception as exc:
        return {
            "answer": "",
            "decision": None,
            "memory_state": {},
            "evidence_event_ids": [],
            "failure_diagnosis": None,
        }, f"{type(exc).__name__}: {exc}"


def _restore_action_array(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```", "").strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_predictions(raw_path: Path, output_dir: Path) -> None:
    rows = _jsonl_rows(raw_path)
    for method in ALL_METHODS:
        target = output_dir / f"{method}.predictions.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                prediction = row.get("predictions", {}).get(method)
                if prediction is not None:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def run_case(
    scenario: dict[str, Any],
    backend: LocalHFBackend,
    retrieval_k: int,
    response_tokens: int,
    action_tokens: int,
) -> dict[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    public_view = public_scenario_view(scenario)
    predictions: dict[str, dict[str, Any]] = {}
    generations: dict[str, Any] = {}

    frozen_response: dict[str, Any] | None = None
    for method in BASELINE_METHODS:
        method_view = build_method_view(method, public_view, retrieval_k)
        generation = backend.generate(build_prompt(method_view), response_tokens)
        response, parse_error = _safe_response(generation.text)
        predictions[method] = {"scenario_id": scenario_id, "response": response}
        generations[method] = {
            **asdict(generation),
            "parse_error": parse_error,
        }
        if method == "frozen_direct":
            frozen_response = response

    assert frozen_response is not None
    revision_view = build_scenario_revision_view(scenario)
    action_generation = backend.generate(build_proposer_prompt(revision_view), action_tokens)
    actions_text = _restore_action_array(action_generation.text)
    parse_result = parse_actions(actions_text)

    guarded_started = time.perf_counter()
    guarded = run_revision_module_on_scenario(
        scenario,
        actions_text=actions_text,
        raw_response=frozen_response,
        include_audit=True,
    )
    guarded_latency = time.perf_counter() - guarded_started
    no_guard_started = time.perf_counter()
    no_guard_response = project_actions_without_dpa(
        view=revision_view,
        parse_result=parse_result,
        raw_response=frozen_response,
        scenario_public_view=public_view,
    )
    no_guard_latency = time.perf_counter() - no_guard_started
    unguarded = {
        "scenario_id": scenario_id,
        "response": no_guard_response,
        "parse_result": parse_result.to_dict(),
    }
    predictions["mempatch"] = guarded
    predictions["mempatch_no_guard"] = unguarded
    generations["mempatch_shared_actions"] = {
        **asdict(action_generation),
        "actions_text": actions_text,
        "parse_result": parse_result.to_dict(),
    }
    generations["deterministic_projection"] = {
        "mempatch_latency_seconds": guarded_latency,
        "mempatch_no_guard_latency_seconds": no_guard_latency,
    }
    return {
        "scenario_id": scenario_id,
        "predictions": predictions,
        "generations": generations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", default="runs/aaai27_main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--response-tokens", type=int, default=1024)
    parser.add_argument("--action-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_root) / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_cases.jsonl"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; pass --resume or choose a new output root")

    scenarios = load_scenarios(args.data)
    completed = {str(row["scenario_id"]) for row in _jsonl_rows(raw_path)} if args.resume else set()
    pending = [scenario for scenario in scenarios if str(scenario["scenario_id"]) not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    manifest = {
        "campaign": "aaai27_main",
        "repository_sha": _git_sha(),
        "model_key": args.model_key,
        "model_id": args.model_id,
        "data": str(Path(args.data).resolve()),
        "seed": args.seed,
        "methods": list(ALL_METHODS),
        "pairing": {
            "frozen_direct_is_mempatch_raw_response": True,
            "mempatch_and_no_guard_share_actions": True,
        },
        "generation": {
            "temperature": 0.0,
            "retrieval_k": args.retrieval_k,
            "response_tokens": args.response_tokens,
            "action_tokens": args.action_tokens,
            "dtype": args.dtype,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset_size": len(scenarios),
        "already_completed": len(completed),
        "planned_now": len(pending),
        "started_at_unix": time.time(),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    backend = LocalHFBackend(args.model_id, args.dtype, args.trust_remote_code)
    for index, scenario in enumerate(pending, start=1):
        started = time.perf_counter()
        row = run_case(
            scenario,
            backend,
            args.retrieval_k,
            args.response_tokens,
            args.action_tokens,
        )
        _append_jsonl(raw_path, row)
        print(
            f"[{index}/{len(pending)}] {row['scenario_id']} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    _write_predictions(raw_path, output_dir)
    manifest["finished_at_unix"] = time.time()
    manifest["completed_total"] = len(_jsonl_rows(raw_path))
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
