#!/usr/bin/env python3
"""Diagnose paper result bundles without changing predictions or scoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mempatch.dpa.action_parser import StructuredParseError, extract_json_array

METRIC_NAMES = {
    "Joint": "joint_revision_success",
    "MemState": "memory_state_accuracy",
    "EvidenceF1": "evidence_f1",
    "DecisionF1": "decision_macro_f1",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _run_id(adapter_path: Any) -> str | None:
    return Path(str(adapter_path)).parent.name if adapter_path else None


def _metric_summary(manifest: dict[str, Any]) -> str:
    headline = manifest.get("headline_metrics") or {}
    return " ".join(
        f"{label}={100 * float(headline.get(key, 0.0)):.1f}"
        for label, key in METRIC_NAMES.items()
    )


def _action_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    valid = admitted = rejected = total = 0
    for row in rows:
        total += 1
        audit = row.get("dpa_audit") or {}
        parse = audit.get("parse_result") or {}
        valid += bool(parse.get("schema_valid"))
        if parse.get("error_message"):
            errors[str(parse["error_message"])] += 1
        admitted += len(audit.get("admitted_actions") or [])
        rejected += len(audit.get("rejected_actions") or [])
        raw = str(row.get("raw_actions_output") or "")
        try:
            raw_actions = extract_json_array(raw)
        except StructuredParseError:
            raw_actions = []
        for action in raw_actions:
            if isinstance(action, dict):
                action_types[str(action.get("action_type") or "<missing>")] += 1
    return {
        "total": total,
        "valid": valid,
        "errors": errors,
        "action_types": action_types,
        "admitted": admitted,
        "rejected": rejected,
    }


def _intersection(scored: Iterable[dict[str, Any]]) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in scored:
        metrics = row.get("metrics") or {}
        response = row.get("response") or {}
        checks = {
            "schema_ok": metrics.get("response_schema_compliance_rate") == 1.0,
            "decision_ok": metrics.get("black_box_decision_accuracy") == 1.0,
            "memory_all_correct": metrics.get("memory_state_accuracy") == 1.0,
            "evidence_exact": metrics.get("minimal_evidence_exact_match") == 1.0,
            "answer_key_fact_ok": metrics.get("answer_key_fact_accuracy") == 1.0,
            "stale_reuse": metrics.get("stale_reuse_rate") == 1.0,
            "joint_success": metrics.get("joint_revision_success") == 1.0,
        }
        counts.update(key for key, value in checks.items() if value)
        answer = str(response.get("answer") or "")
        if checks["decision_ok"] and not checks["joint_success"] and len(examples) < 3:
            examples.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "memory": checks["memory_all_correct"],
                    "evidence": checks["evidence_exact"],
                    "answer": checks["answer_key_fact_ok"],
                    "schema": checks["schema_ok"],
                    "answer_text": answer[:120],
                    "failure_diagnosis": response.get("failure_diagnosis"),
                }
            )
    return counts, examples


def diagnose_phi4(result_dir: Path, run_id_filter: str | None) -> None:
    print(f"\n== Phi-4 Detailed Diagnostics: {result_dir.name} ==")

    selection_path = result_dir / "checkpoint_selection.json"
    if selection_path.is_file():
        sel = _load_json(selection_path)
        print(f"  Selected Checkpoint: {sel.get('checkpoint_dir')}")
    else:
        print(f"  Warning: checkpoint_selection.json missing under {result_dir}")

    manifests = sorted(result_dir.glob("*_manifest.json"))
    if not manifests:
        print("  Warning: no result manifests found under results dir")
        return

    for manifest_path in manifests:
        tag = manifest_path.name.removesuffix("_manifest.json")
        manifest = _load_json(manifest_path)
        meta = manifest.get("run_meta") or {}
        manifest_run_id = meta.get("run_id") or _run_id(meta.get("adapter_path"))
        if run_id_filter is not None and manifest_run_id != run_id_filter:
            continue

        print(f"\n  Manifest Tag: {tag}")
        print(f"    Slug: {result_dir.name}")
        print(f"    HF Model ID: {meta.get('model_id') or '<missing>'}")
        print(f"    RUN_ID: {manifest_run_id or '<missing>'}")
        print(f"    Adapter Path: {meta.get('adapter_path') or '<none/base>'}")

        predictions_path = result_dir / f"{tag}_predictions.jsonl"
        predictions = _rows(predictions_path)
        print(f"    Prediction counts: {len(predictions)}")
        if not predictions_path.is_file():
            print(f"    Warning: predictions file missing: {predictions_path.name}")

        headline = manifest.get("headline_metrics") or {}

        compliance = headline.get("response_schema_compliance_rate")
        if compliance is None:
            paired_path_b = meta.get("paired_path_b_headline_metrics") or {}
            compliance = paired_path_b.get("response_schema_compliance_rate")
        if compliance is not None:
            print(f"    Path B schema compliance: {100 * float(compliance):.2f}%")
        else:
            print(f"    Path B schema compliance: <missing>")

        if meta.get("method_path") in ("path_a_typed_actions_dpa", "path_a_typed_actions_no_dpa") or "path_a" in tag:
            diag = _action_diagnostics(predictions)
            valid_rate = diag["valid"] / diag["total"] if diag["total"] else 0.0
            print(f"    Path A typed-action parse validity: {100 * valid_rate:.2f}%")
            print(f"    Admitted action count: {diag['admitted']}")
            print(f"    Rejected action count: {diag['rejected']}")
            print(f"    Action type counts: {dict(diag['action_types'])}")
            print(f"    Top parse error messages:")
            for message, count in diag["errors"].most_common(3):
                print(f"      [{count}]: {message}")

        print(f"    Headline Metrics:")
        print(f"      decision_macro_f1: {headline.get('decision_macro_f1')}")
        print(f"      memory_state_accuracy: {headline.get('memory_state_accuracy')}")
        print(f"      evidence_f1: {headline.get('evidence_f1')}")
        print(f"      joint_revision_success: {headline.get('joint_revision_success')}")
        print(f"      stale_reuse_rate: {headline.get('stale_reuse_rate')}")


def diagnose_slug(result_dir: Path, *, examples: int, run_id_filter: str | None = None) -> None:
    if result_dir.name == "phi4_14b":
        diagnose_phi4(result_dir, run_id_filter)
        return

    print(f"\n== {result_dir.name} ==")
    manifests = sorted(result_dir.glob("*_manifest.json"))
    if not manifests:
        print("missing result manifests")
        return

    for manifest_path in manifests:
        tag = manifest_path.name.removesuffix("_manifest.json")
        manifest = _load_json(manifest_path)
        meta = manifest.get("run_meta") or {}
        manifest_run_id = meta.get("run_id") or _run_id(meta.get("adapter_path"))
        if run_id_filter is not None and manifest_run_id != run_id_filter:
            continue

        predictions = _rows(result_dir / f"{tag}_predictions.jsonl")
        print(
            f"{tag}: n={len(predictions)} adapter={meta.get('adapter_path')} "
            f"run_id={manifest_run_id} "
            f"{_metric_summary(manifest)}"
        )

        if meta.get("method_path") == "path_a_typed_actions_dpa":
            diag = _action_diagnostics(predictions)
            rate = diag["valid"] / diag["total"] if diag["total"] else 0.0
            print(
                f"  actions: parse_valid={100 * rate:.1f}% admitted={diag['admitted']} "
                f"rejected={diag['rejected']} types={dict(diag['action_types'].most_common())}"
            )
            for message, count in diag["errors"].most_common(5):
                print(f"  parse_error[{count}]: {message}")

        if tag.startswith("baseline_"):
            scored = _rows(result_dir / f"{tag}_scored.jsonl")
            counts, failed = _intersection(scored)
            generic = sum(
                str((row.get("response") or {}).get("answer") or "").strip().lower()
                in {"use current memory", "use current memory."}
                for row in scored
            )
            truncation_shape = sum(
                bool(row.get("raw_output"))
                and not str(row.get("raw_output")).rstrip().endswith("}")
                for row in predictions
            )
            labels = (
                "schema_ok", "decision_ok", "memory_all_correct", "evidence_exact",
                "answer_key_fact_ok", "stale_reuse", "joint_success",
            )
            print("  intersection: " + " ".join(f"{key}={counts[key]}" for key in labels))
            print(f"  generic_use_current={generic} truncation_shape={truncation_shape}")
            for row in failed[:examples]:
                print("  failed_joint: " + json.dumps(row, ensure_ascii=False, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("local/results"))
    parser.add_argument("--slugs", nargs="*", default=[])
    parser.add_argument("--slug", dest="slugs", action="append", help="Specify a single slug (can be repeated)")
    parser.add_argument("--run-id", type=str, default=None, help="Optionally filter manifests by run-id")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    discovered = (
        [path.name for path in sorted(args.results_root.iterdir()) if path.is_dir()]
        if args.results_root.is_dir()
        else []
    )
    # clean None values if `--slug` was used and `--slugs` had defaults
    slugs = [s for s in (args.slugs or discovered) if s]
    if not slugs:
        print(f"No result bundles found under {args.results_root}")
        return 0
    for slug in slugs:
        diagnose_slug(args.results_root / slug, examples=max(0, args.examples), run_id_filter=args.run_id)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
