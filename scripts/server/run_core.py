"""Unified, resumable server runner for MemPatch model campaigns."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.api import load_scenarios
from mempatch.benchmark.model_runner import canonical_response, extract_json_object
from mempatch.benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.dpa_runtime import parse_actions
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view

try:
    from .methods import build_method_view
except ImportError:
    from methods import build_method_view

from mempatch.benchmark.general_taxonomy import DECISIONS, FAILURE_MODES, MEMORY_OPERATIONS, MEMORY_STATUSES
from mempatch.benchmark.model_runner import _collect_memory_ids

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
        
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if "mistral" in model_id.lower():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    fix_mistral_regex=True,
                    **tokenizer_kwargs
                )
            except TypeError:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    **tokenizer_kwargs
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                **tokenizer_kwargs
            )
            
        dtype_value = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype_value,
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


def build_prompt(public_view: dict[str, Any]) -> str:
    memory_ids = _collect_memory_ids(public_view)
    decision_labels = ", ".join(DECISIONS)
    failure_mode_labels = ", ".join(FAILURE_MODES)
    memory_operation_labels = ", ".join(MEMORY_OPERATIONS)
    status_labels = ", ".join(MEMORY_STATUSES)
    payload = {
        "instruction": (
            "Answer as strict JSON only matching the schema exactly. Do not use Markdown fences. "
            "Use only the visible scenario content. Do not use external knowledge. "
            "Use exact enum strings. Do not invent memory IDs or event IDs. "
            "Cite only minimal supporting event IDs. "
            "Choose exactly one lifecycle memory_operation for the durable memory action. "
            "Decision order: refuse_due_to_policy, escalate, ask_clarification, "
            "mark_unresolved, use_current_memory (first applicable wins). "
            "CRITICAL: 'decision', 'memory_operation', and 'failure_diagnosis' must be scalar STRINGS, NOT lists or arrays. "
            "Provide exactly one valid enum string for 'decision', 'memory_operation', and 'failure_diagnosis' respectively. "
            "CRITICAL WARNING ON 'failure_diagnosis': Even if the memory state is correct, or your decision is use_current_memory, and there appears to be no issue, you MUST NOT output 'none', 'null', 'ok', or any other custom string. "
            f"You MUST select EXACTLY ONE failure mode from this list as the failure_diagnosis: {failure_mode_labels}. "
            "Select the failure mode that MOST CLOSELY represents the hypothetical or potential threat described in the scenario."
        ),
        "required_output_schema": {
            "answer": "short final answer/action text (string)",
            "decision": f"exactly one string from: {decision_labels} (string)",
            "memory_operation": f"exactly one string from: {memory_operation_labels} (string)",
            "memory_state": {mid: f"exactly one string from: {status_labels} (string)" for mid in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings from public_input.events or public_input.event_trace (list of strings)",
            "failure_diagnosis": f"exactly one string from: {failure_mode_labels} (string)",
            "followup_answer": "short answer to the visible followup_task after applying the memory operation (string)",
        },
        **public_view,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_response(parsed: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(parsed)
    for key in ("decision", "memory_operation", "failure_diagnosis"):
        val = cleaned.get(key)
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                cleaned[key] = val[0]
            elif len(val) == 0:
                cleaned[key] = ""
        elif val is None:
            cleaned[key] = ""
            
    mem = cleaned.get("memory_state")
    if isinstance(mem, dict):
        cleaned["memory_state"] = {
            k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
            for k, v in mem.items()
        }
    return cleaned


def _safe_response(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = extract_json_object(text)
        cleaned = clean_response(parsed)
        return canonical_response(cleaned), None
    except Exception as exc:
        return {
            "answer": "",
            "decision": None,
            "memory_operation": None,
            "memory_state": {},
            "evidence_event_ids": [],
            "failure_diagnosis": None,
            "followup_answer": "",
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
    parser.add_argument("--output-root", default="runs/eval_main")
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
    
    scenarios = load_scenarios(args.data)
    
    # Dataset Audit
    event_lens = [
        len((c.get("public_input", {}) or {}).get("event_trace") or (c.get("public_input", {}) or {}).get("events") or [])
        for c in scenarios
    ]
    min_len = min(event_lens) if event_lens else 0
    median_len = int(np.median(event_lens)) if event_lens else 0
    p90_len = int(np.percentile(event_lens, 90)) if event_lens else 0
    max_len = max(event_lens) if event_lens else 0
    
    print(f"[Dataset Audit] Scenarios count: {len(scenarios)}")
    print(f"[Dataset Audit] Event trace lengths - min: {min_len}, median: {median_len}, p90: {p90_len}, max: {max_len}")
    from collections import Counter
    for length, count in sorted(Counter(event_lens).items()):
        print(f"  Length {length}: {count} cases")
        
    final_k = args.retrieval_k
    if final_k >= median_len:
        if median_len > 3:
            final_k = 3
        else:
            final_k = max(1, median_len - 1)
        print(f"[Dataset Audit] Auto-adjusted retrieval_k to {final_k} (was {args.retrieval_k}) to avoid full-context degradation.")
        
    # 计算 sha256 校验
    import hashlib
    dataset_bytes = Path(args.data).read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    
    scenario_ids = [str(s["scenario_id"]) for s in scenarios]
    ids_str = ",".join(scenario_ids)
    scenario_ids_sha256 = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()
    
    manifest_path = output_dir / "run_manifest.json"
    
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; pass --resume or choose a new output root")
        
    if raw_path.exists() and args.resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = []
        if existing_manifest.get("dataset_sha256") != dataset_sha256:
            mismatches.append("dataset_sha256")
        if existing_manifest.get("scenario_ids_sha256") != scenario_ids_sha256:
            mismatches.append("scenario_ids_sha256")
        if existing_manifest.get("model_id") != args.model_id:
            mismatches.append("model_id")
        if existing_manifest.get("retrieval_k") != final_k:
            mismatches.append("retrieval_k")
        if existing_manifest.get("response_token_limit") != args.response_tokens:
            mismatches.append("response_token_limit")
        if existing_manifest.get("action_token_limit") != args.action_tokens:
            mismatches.append("action_token_limit")
            
        if mismatches:
            raise RuntimeError(
                f"Resuming aborted: current configuration does not match existing manifest "
                f"at {manifest_path}. Mismatched fields: {mismatches}."
            )

    completed = {str(row["scenario_id"]) for row in _jsonl_rows(raw_path)} if args.resume else set()
    pending = [scenario for scenario in scenarios if str(scenario["scenario_id"]) not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    backend = LocalHFBackend(args.model_id, args.dtype, args.trust_remote_code)
    
    import torch
    import transformers
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    manifest = {
        "campaign": "eval_main",
        "repository_sha": _git_sha(),
        "model_key": args.model_key,
        "model_id": args.model_id,
        "model_class": backend.model.__class__.__name__,
        "tokenizer_class": backend.tokenizer.__class__.__name__,
        "data_path": str(Path(args.data).resolve()),
        "dataset_sha256": dataset_sha256,
        "scenario_ids_sha256": scenario_ids_sha256,
        "seed": args.seed,
        "methods": list(ALL_METHODS),
        "pairing": {
            "frozen_direct_is_mempatch_raw_response": True,
            "mempatch_and_no_guard_share_actions": True,
        },
        "retrieval_k": final_k,
        "decoding_params": {
            "temperature": 0.0,
            "do_sample": False
        },
        "response_token_limit": args.response_tokens,
        "action_token_limit": args.action_tokens,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": cuda_version,
            "platform": platform.platform(),
        },
        "gpu_name": gpu_name,
        "dataset_size": len(scenarios),
        "already_completed": len(completed),
        "planned_now": len(pending),
        "started_at_unix": time.time(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for index, scenario in enumerate(pending, start=1):
        started = time.perf_counter()
        row = run_case(
            scenario,
            backend,
            final_k,
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
