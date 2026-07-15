"""Shared, paired case pipeline for MemPatch paper evaluations.

This module owns the case-level orchestration shared by server, local, and
provider-backed runners.  Model generation is the only external seam: callers
provide a ``Generator`` while prompt construction, parsing, typed revision,
conservative repair, and optional no-guard projection stay in-process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any, Protocol, Sequence

from mempatch.benchmark.contracts import RESPONSE_FIELDS, validate_prediction
from mempatch.benchmark.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    MEMORY_OPERATIONS,
    MEMORY_STATUSES,
)
from mempatch.benchmark.method_names import PAPER_METHODS
from mempatch.benchmark.public_view import public_scenario_view
from mempatch.dpa.action_parser import extract_json_object
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.benchmark_projection import project_to_benchmark_response
from mempatch.revision.runtime.dpa_runtime import parse_actions, run_from_text
from mempatch.revision.runtime.proposal_prompt import build_proposer_prompt
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view
from mempatch.benchmark.methods import build_method_view


# Preserve the generation order used by tools/evaluation/server/run_core.py.
_BASELINE_EXECUTION_ORDER = (
    "direct_json",
    "full_context_json",
    "bm25_rag_json",
    "dense_rag_json",
    "time_aware_rag_json",
    "summary_memory_json",
)


@dataclass(frozen=True)
class GenerationRecord:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float


class Generator(Protocol):
    """Generation adapter used by the shared case pipeline."""

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationRecord:
        ...


def _collect_memory_ids(public_view: dict[str, Any]) -> list[str]:
    public_input = public_view.get("public_input", {})
    memory_ids = [
        str(memory["memory_id"])
        for memory in public_input.get("initial_memory")
        or public_input.get("initial_memories")
        or []
        if isinstance(memory, dict) and memory.get("memory_id")
    ]
    for event in public_input.get("event_trace") or public_input.get("events") or []:
        if not isinstance(event, dict):
            continue
        for memory_id in event.get("related_memory_ids", []) or []:
            if memory_id not in memory_ids:
                memory_ids.append(str(memory_id))
    return memory_ids


def build_benchmark_prompt(public_view: dict[str, Any]) -> str:
    """Render the strict benchmark-response prompt used by the server runner."""
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
            "memory_state": {
                memory_id: f"exactly one string from: {status_labels} (string)"
                for memory_id in memory_ids
            },
            "evidence_event_ids": "minimal list of event_id strings from public_input.events or public_input.event_trace (list of strings)",
            "failure_diagnosis": f"exactly one string from: {failure_mode_labels} (string)",
            "followup_answer": "short answer to the visible followup_task after applying the memory operation (string)",
        },
        **public_view,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_benchmark_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalize common scalar-list mistakes before contract validation."""
    cleaned = dict(parsed)
    for key in ("decision", "memory_operation", "failure_diagnosis"):
        value = cleaned.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                cleaned[key] = value[0]
            elif len(value) == 0:
                cleaned[key] = ""
        elif value is None:
            cleaned[key] = ""

    memory_state = cleaned.get("memory_state")
    if isinstance(memory_state, dict):
        cleaned["memory_state"] = {
            key: value[0]
            if isinstance(value, (list, tuple)) and len(value) == 1
            else value
            for key, value in memory_state.items()
        }
    return cleaned


def _canonical_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep only benchmark fields and normalize missing containers."""
    response = {field: parsed.get(field) for field in RESPONSE_FIELDS}
    if not isinstance(response["memory_state"], dict):
        response["memory_state"] = {}
    if not isinstance(response["evidence_event_ids"], list):
        response["evidence_event_ids"] = []
    if response["followup_answer"] is None:
        response["followup_answer"] = ""
    return response


def parse_benchmark_response(text: str) -> tuple[dict[str, Any], str | None]:
    """Parse one generation, returning the fail-closed canonical response."""
    try:
        parsed = extract_json_object(text)
        return _canonical_response(clean_benchmark_response(parsed)), None
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


def restore_action_array(text: str) -> str:
    """Recover the outer JSON action array from common model wrappers."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```", "").strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def response_contract_valid(response: dict[str, Any] | None) -> bool:
    """Apply the conservative repair branch's existing contract check."""
    return response is not None and not validate_prediction({"parsed": response})


def repair_response(
    direct_response: dict[str, Any],
    typed_response: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Preserve a valid Direct response verbatim; otherwise use typed output."""
    if response_contract_valid(direct_response):
        return direct_response, "direct_json_valid"
    return typed_response, "typed_projection_fallback"


def _stable_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluate_case(
    scenario: dict[str, Any],
    generator: Generator,
    methods: Sequence[str] = PAPER_METHODS,
    retrieval_k: int = 3,
    response_tokens: int = 512,
    action_tokens: int = 384,
    include_no_guard: bool = False,
) -> dict[str, Any]:
    """Generate all requested predictions for one paired benchmark case.

    Direct JSON is generated at most once.  The MemPatch and optional no-guard
    branches both reuse that exact parsed response and apply the established
    conservative repair rule.
    """
    selected = tuple(dict.fromkeys(str(method) for method in methods))
    unknown = sorted(set(selected) - set(PAPER_METHODS))
    if unknown:
        raise ValueError(f"unknown paper methods: {unknown}")

    scenario_id = str(scenario["scenario_id"])
    public_input_before = _stable_sha256(scenario.get("public_input", {}))
    public_view = public_scenario_view(scenario)
    predictions: dict[str, dict[str, Any]] = {}
    generations: dict[str, Any] = {}

    needs_typed_revision = "mempatch" in selected or include_no_guard
    baseline_methods = set(selected) & set(_BASELINE_EXECUTION_ORDER)
    if needs_typed_revision:
        baseline_methods.add("direct_json")

    direct_response: dict[str, Any] | None = None
    for method in _BASELINE_EXECUTION_ORDER:
        if method not in baseline_methods:
            continue
        method_view = build_method_view(method, public_view, retrieval_k)
        generation = generator.generate(
            build_benchmark_prompt(method_view),
            response_tokens,
        )
        response, parse_error = parse_benchmark_response(generation.text)
        retrieval_metadata = method_view.get("retrieval_metadata") or {}
        if method in selected:
            predictions[method] = {
                "scenario_id": scenario_id,
                "method": method,
                "response": response,
                "retrieved_event_count": retrieval_metadata.get(
                    "retrieved_event_count"
                ),
            }
        generations[method] = {
            **asdict(generation),
            "parse_error": parse_error,
        }
        if method == "direct_json":
            direct_response = response

    if needs_typed_revision:
        if direct_response is None:
            raise AssertionError("typed revision requires one Direct response")

        # From this point onward the revision path receives only the sanitized
        # public view, never the original row that may contain hidden labels.
        revision_view = build_scenario_revision_view(public_view)
        action_generation = generator.generate(
            build_proposer_prompt(revision_view),
            action_tokens,
        )
        actions_text = restore_action_array(action_generation.text)
        parse_result = parse_actions(actions_text)

        guarded_started = time.perf_counter()
        runtime_result = run_from_text(revision_view, actions_text)
        guarded = {
            "scenario_id": scenario_id,
            "response": project_to_benchmark_response(
                runtime_result=runtime_result,
                raw_response=direct_response,
                scenario_public_view=public_view,
            ),
            "dpa_audit": runtime_result.to_dict(),
        }
        guarded_latency = time.perf_counter() - guarded_started
        guarded_response, guarded_source = repair_response(
            direct_response,
            guarded["response"],
        )
        if "mempatch" in selected:
            predictions["mempatch"] = {
                **guarded,
                "response": guarded_response,
                "repair_source": guarded_source,
                "method": "mempatch",
            }

        projection_timings = {"mempatch_latency_seconds": guarded_latency}
        if include_no_guard:
            no_guard_started = time.perf_counter()
            no_guard_projected = project_actions_without_dpa(
                view=revision_view,
                parse_result=parse_result,
                raw_response=direct_response,
                scenario_public_view=public_view,
            )
            no_guard_latency = time.perf_counter() - no_guard_started
            no_guard_response, no_guard_source = repair_response(
                direct_response,
                no_guard_projected,
            )
            predictions["mempatch_noguard"] = {
                "scenario_id": scenario_id,
                "method": "mempatch_noguard",
                "response": no_guard_response,
                "repair_source": no_guard_source,
                "parse_result": parse_result.to_dict(),
            }
            projection_timings.update(
                {
                    "mempatch_noguard_latency_seconds": no_guard_latency,
                    "mempatch_no_guard_latency_seconds": no_guard_latency,
                }
            )

        generations["mempatch_shared_actions"] = {
            **asdict(action_generation),
            "actions_text": actions_text,
            "parse_result": parse_result.to_dict(),
        }
        generations["deterministic_projection"] = projection_timings
        direct_hash = _stable_sha256(direct_response)
        generations["pairing"] = {
            "direct_response_sha256": direct_hash,
            "mempatch_base_response_sha256": direct_hash,
            "public_view_sha256": _stable_sha256(public_view),
        }

    if _stable_sha256(scenario.get("public_input", {})) != public_input_before:
        raise RuntimeError(f"{scenario_id}: evaluation mutated public_input")

    return {
        "scenario_id": scenario_id,
        "predictions": predictions,
        "generations": generations,
    }


__all__ = [
    "GenerationRecord",
    "Generator",
    "evaluate_case",
]
