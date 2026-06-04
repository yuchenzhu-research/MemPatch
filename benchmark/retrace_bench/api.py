"""Official public scoring API for ReTrace-Bench.

This module is the stable entrypoint external benchmark users should import. It
wraps the existing scorer (:mod:`benchmark.retrace_bench.scorers_general`) and
adds prediction loading, normalization, validation, and aggregate evaluation.

Typical usage::

    from benchmark.retrace_bench.api import (
        load_scenarios, load_predictions, evaluate_predictions,
    )

    scenarios = load_scenarios("data/retrace_bench/main_3000_en")
    predictions = load_predictions("my_model.predictions.jsonl")
    result = evaluate_predictions(scenarios, predictions, strict=True)
    print(result["headline_metrics"])

The scorer itself is unchanged; this module only orchestrates it and exposes a
documented, backward-compatible surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    MEMORY_STATUSES,
)
from benchmark.retrace_bench.scorers_general import (
    AUXILIARY_METRICS,
    HEADLINE_METRICS,
    aggregate_metrics,
    normalize_failure_mode,
    score_prediction,
)

__all__ = [
    "load_scenarios",
    "load_predictions",
    "normalize_prediction",
    "evaluate_predictions",
    "HEADLINE_METRICS",
    "AUXILIARY_METRICS",
    "DECISIONS",
    "MEMORY_STATUSES",
    "FAILURE_MODES",
]

SCENARIO_JSONL_NAME = "scenarios.jsonl"

# Fields that make up a canonical prediction ``response`` object. ``decision``
# is the only hard-required field; the rest are scored when present.
RESPONSE_FIELDS = (
    "answer",
    "decision",
    "memory_state",
    "evidence_event_ids",
    "failure_diagnosis",
)


def _normalize_response_object(response: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(response)
    for key in ("decision", "failure_diagnosis"):
        value = normalized.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            normalized[key] = value[0]
    
    memory_state = normalized.get("memory_state")
    if isinstance(memory_state, dict):
        new_mem = {}
        for k, v in memory_state.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                new_mem[k] = v[0]
            else:
                new_mem[k] = v
        normalized["memory_state"] = new_mem

    evidence = normalized.get("evidence_event_ids")
    if isinstance(evidence, str):
        normalized["evidence_event_ids"] = [evidence]
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_scenarios(path_or_dir: str | Path) -> list[dict[str, Any]]:
    """Load benchmark scenarios from a ``scenarios.jsonl`` file or a directory.

    If ``path_or_dir`` is a directory it must contain ``scenarios.jsonl``.
    """
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / SCENARIO_JSONL_NAME
    if not path.exists():
        raise FileNotFoundError(f"scenarios file not found: {path}")
    return _read_jsonl(path)


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL predictions (canonical or flat response format)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"predictions file not found: {p}")
    return _read_jsonl(p)


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single prediction row to the canonical nested form.

    Accepts either the canonical format with a nested ``response`` object or the
    flat format where the response fields live at the top level. Returns
    ``{"scenario_id": ..., "response": {...}}``.
    """
    scenario_id = row.get("scenario_id")
    raw_response = row.get("response")
    if isinstance(raw_response, dict):
        response = _normalize_response_object(raw_response)
    else:
        response = _normalize_response_object(
            {key: row[key] for key in RESPONSE_FIELDS if key in row}
        )
    return {"scenario_id": scenario_id, "response": response}


def _scenario_event_ids(scenario: dict[str, Any]) -> set[str]:
    events = scenario.get("public_input", {}).get("event_trace", [])
    return {e.get("event_id") for e in events if e.get("event_id") is not None}


def _scenario_memory_ids(scenario: dict[str, Any]) -> set[str]:
    """Memory IDs visible to the model (initial_memory only; never gold)."""
    memories = scenario.get("public_input", {}).get("initial_memory", [])
    return {m.get("memory_id") for m in memories if m.get("memory_id") is not None}


def _validate_response(
    scenario_id: Any,
    response: dict[str, Any],
    event_ids: set[str],
    memory_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate one normalized response, appending errors/warnings in place."""
    if not isinstance(response, dict) or not response:
        errors.append(f"{scenario_id}: missing or empty response")
        return

    def is_hashable(v: Any) -> bool:
        try:
            hash(v)
            return True
        except TypeError:
            return False

    # decision: required + must be a known label.
    decision = response.get("decision")
    if decision is None:
        errors.append(f"{scenario_id}: missing response field 'decision'")
    elif not is_hashable(decision) or decision not in DECISIONS:
        errors.append(
            f"{scenario_id}: invalid decision label {decision!r} "
            f"(expected one of {sorted(DECISIONS)})"
        )

    # memory_state: optional, but if present must be a dict of valid statuses.
    if "memory_state" not in response:
        warnings.append(f"{scenario_id}: missing response field 'memory_state'")
    else:
        memory_state = response.get("memory_state")
        if not isinstance(memory_state, dict):
            errors.append(f"{scenario_id}: 'memory_state' must be an object (memory_id -> status)")
        else:
            bad = []
            for s in memory_state.values():
                if not is_hashable(s) or s not in MEMORY_STATUSES:
                    bad.append(repr(s) if not is_hashable(s) else s)
            if bad:
                errors.append(
                    f"{scenario_id}: invalid memory_state labels {sorted(bad)} "
                    f"(expected one of {sorted(MEMORY_STATUSES)})"
                )
            # Completeness: a status is expected for every visible memory ID.
            # This is a warning (not error) in both modes so partial coverage is
            # scored; only initial_memory IDs are used, so no gold is exposed.
            omitted = sorted(mid for mid in memory_ids if mid not in memory_state)
            if omitted:
                warnings.append(
                    f"{scenario_id}: memory_state omits visible memory IDs {omitted}"
                )

    # evidence_event_ids: optional, but if present must reference real events.
    if "evidence_event_ids" not in response:
        warnings.append(f"{scenario_id}: missing response field 'evidence_event_ids'")
    else:
        evidence = response.get("evidence_event_ids")
        if not isinstance(evidence, list):
            errors.append(f"{scenario_id}: 'evidence_event_ids' must be a list of event IDs")
        else:
            unknown = []
            for eid in evidence:
                if not is_hashable(eid) or eid not in event_ids:
                    unknown.append(eid)
            if unknown:
                errors.append(
                    f"{scenario_id}: evidence_event_ids reference IDs not in event_trace: {unknown}"
                )

    # failure_diagnosis: required to be one of FAILURE_MODES (or an accepted
    # normalized alias) when present. Missing is a warning; an unrecognized
    # label is an error so strict mode rejects it. The scorer still normalizes
    # free-text aliases, so documented aliases pass.
    if "failure_diagnosis" not in response:
        warnings.append(f"{scenario_id}: missing response field 'failure_diagnosis'")
    else:
        diagnosis = response.get("failure_diagnosis")
        if normalize_failure_mode(diagnosis) not in FAILURE_MODES:
            errors.append(
                f"{scenario_id}: invalid failure_diagnosis label {diagnosis!r} "
                f"(expected one of {sorted(FAILURE_MODES)} or a documented alias)"
            )

    # answer: free text. Missing is a warning only.
    if "answer" not in response:
        warnings.append(f"{scenario_id}: missing response field 'answer'")


def evaluate_predictions(
    scenarios: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    strict: bool = True,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Score predictions against scenarios, matching by ``scenario_id``.

    Detects missing/duplicate/extra predictions and invalid response fields,
    decision labels, memory-state labels, and evidence IDs. In ``strict`` mode a
    :class:`ValueError` is raised if any errors are found; otherwise problems are
    returned as ``warnings``/``errors`` and everything scorable is still scored.
    Set ``allow_missing=True`` for partial local runs; missing predictions are
    summarized instead of reported as per-scenario errors.

    Returns a dict with ``count``, ``headline_metrics``, ``auxiliary_metrics``,
    ``all_metrics``, ``warnings``, ``errors``, and ``scored_predictions``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    missing_predictions: list[Any] = []

    scenario_by_id: dict[Any, dict[str, Any]] = {}
    for scenario in scenarios:
        sid = scenario.get("scenario_id")
        if sid in scenario_by_id:
            warnings.append(f"{sid}: duplicate scenario_id in scenarios (using first)")
            continue
        scenario_by_id[sid] = scenario

    # Index predictions, detecting duplicates and extras.
    normalized_by_id: dict[Any, dict[str, Any]] = {}
    for raw in predictions:
        norm = normalize_prediction(raw)
        sid = norm["scenario_id"]
        if sid is None:
            errors.append("prediction is missing 'scenario_id'")
            continue
        if sid in normalized_by_id:
            errors.append(f"{sid}: duplicate prediction for scenario_id (using first)")
            continue
        if sid not in scenario_by_id:
            errors.append(f"{sid}: extra prediction with no matching scenario")
            continue
        normalized_by_id[sid] = norm

    # Missing predictions.
    for sid in scenario_by_id:
        if sid not in normalized_by_id:
            missing_predictions.append(sid)
    if missing_predictions:
        if allow_missing:
            warnings.append(
                f"{len(missing_predictions)} missing prediction(s) ignored because allow_missing=True"
            )
        else:
            errors.extend(f"{sid}: missing prediction" for sid in missing_predictions)

    # Validate + build scorable rows.
    scored_rows: list[dict[str, Any]] = []
    for sid, scenario in scenario_by_id.items():
        norm = normalized_by_id.get(sid)
        if norm is None:
            continue
        response = norm["response"]
        _validate_response(
            sid,
            response,
            _scenario_event_ids(scenario),
            _scenario_memory_ids(scenario),
            errors,
            warnings,
        )
        if not isinstance(response, dict) or not response:
            continue
        gold = scenario.get("hidden_gold", {}) or {}
        rubric = gold.get("rubric", {}) or {}
        row = {
            "scenario_id": sid,
            "response": response,
            "domain": scenario.get("domain"),
            "primary_failure_mode": scenario.get("primary_failure_mode"),
            "expected_decision": gold.get("expected_decision"),
            "decision_aliases": (
                gold.get("decision_aliases")
                or rubric.get("decision_aliases")
                or scenario.get("decision_aliases")
            ),
        }
        row["metrics"] = score_prediction(scenario, row)
        scored_rows.append(row)

    if strict and errors:
        raise ValueError(
            "evaluate_predictions found "
            f"{len(errors)} error(s) in strict mode:\n  - " + "\n  - ".join(errors)
        )

    aggregate = aggregate_metrics(scored_rows)
    return {
        "count": aggregate["count"],
        "headline_metrics": aggregate["headline_metrics"],
        "auxiliary_metrics": aggregate["auxiliary_metrics"],
        "all_metrics": aggregate["all_metrics"],
        "warnings": warnings,
        "errors": errors,
        "missing_prediction_count": len(missing_predictions),
        "scored_predictions": scored_rows,
    }
