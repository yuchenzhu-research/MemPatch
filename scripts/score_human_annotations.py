#!/usr/bin/env python3
"""Score human annotations for ReTrace-Bench (internal "v1.1").

Consumes annotation files from **one or more human annotators** (CSV or JSONL,
in the schema produced by ``export_human_annotation_packet.py``) plus the gold
packet, and reports:

* dataset-quality rates: solvable / topic-domain-consistent / ambiguity /
  filler-heavy;
* human-vs-gold metrics via the official scorer
  (``benchmark.retrace_bench.scorers_general.score_prediction``): decision
  accuracy, memory_state_accuracy, evidence_f1, minimal_evidence_exact_match,
  failure_diagnosis_accuracy, joint_revision_success, plus the per-annotator and
  pooled "human upper bound";
* inter-annotator agreement (IAA): pairwise Cohen's kappa for categorical labels,
  pairwise evidence F1 / Jaccard, per-memory memory-state agreement, and a
  nominal Krippendorff's alpha when 3+ annotators are present;
* per-decision-label, per-pattern, and per-difficulty breakdowns.

**Integrity:** LLMs/AI assistants may NOT be represented as human annotators. If
no annotation files are supplied, the script writes an *empty* results template
and reports status "protocol prepared; human annotation pending". It never
fabricates agreement numbers.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = REPO / "annotation_packets" / "retrace_bench_v1_1" / "paper_validation_200_gold.jsonl"
DEFAULT_OUT_DIR = REPO / "outputs" / "retrace_bench_v1_1" / "human_validation"
DEFAULT_TEMPLATE = REPO / "docs" / "retrace_bench" / "human_validation_results_template.md"

CATEGORICAL_IAA_FIELDS = (
    "decision_label",
    "failure_diagnosis",
    "solvable_from_visible_evidence",
    "topic_domain_consistent",
    "ambiguous_or_multiple_valid_answers",
)


# --------------------------------------------------------------------------- IO

def _read_gold(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["scenario_id"]] = row
    return gold


def _parse_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text)]
        except json.JSONDecodeError:
            pass
    return [tok.strip() for tok in text.replace(";", ",").split(",") if tok.strip()]


def _parse_obj(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    return {}


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    # Normalize structured fields.
    for r in rows:
        r["evidence_event_ids"] = _parse_list(r.get("evidence_event_ids"))
        r["memory_state_json"] = _parse_obj(r.get("memory_state_json"))
    return rows


def _has_content(row: dict[str, Any]) -> bool:
    return bool((row.get("decision_label") or "").strip())


# ----------------------------------------------------------------- IAA helpers

def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    paired = [(a, b) for a, b in zip(labels_a, labels_b) if a != "" and b != ""]
    n = len(paired)
    if n == 0:
        return None
    categories = sorted({c for pair in paired for c in pair})
    po = sum(1 for a, b in paired if a == b) / n
    pa = Counter(a for a, _ in paired)
    pb = Counter(b for _, b in paired)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in categories)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _evidence_f1(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    tp = len(sa & sb)
    prec = tp / len(sa) if sa else 0.0
    rec = tp / len(sb) if sb else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _krippendorff_alpha_nominal(by_unit: dict[str, list[str]]) -> float | None:
    """Nominal Krippendorff's alpha over units with >= 2 ratings each."""
    units = {u: [v for v in vals if v != ""] for u, vals in by_unit.items()}
    units = {u: vals for u, vals in units.items() if len(vals) >= 2}
    if not units:
        return None
    # Observed disagreement.
    do_num = 0.0
    n_pairs = 0
    value_counts: Counter[str] = Counter()
    for vals in units.values():
        m = len(vals)
        for v in vals:
            value_counts[v] += 1
        for x, y in itertools.permutations(vals, 2):
            do_num += 1 if x != y else 0
        n_pairs += m * (m - 1)
    if n_pairs == 0:
        return None
    do = do_num / n_pairs
    total = sum(value_counts.values())
    de = 1 - sum((c / total) ** 2 for c in value_counts.values())
    if de == 0:
        return 1.0
    return 1 - do / de


# --------------------------------------------------------- per-annotator vs gold

def _annotation_to_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row.get("scenario_id"),
        "response": {
            "answer": row.get("answer_short_free_text", ""),
            "decision": (row.get("decision_label") or "").strip(),
            "memory_state": row.get("memory_state_json") or {},
            "evidence_event_ids": row.get("evidence_event_ids") or [],
            "failure_diagnosis": (row.get("failure_diagnosis") or "").strip(),
        },
    }


def _score_annotator_vs_gold(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    per_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row.get("scenario_id")
        if sid not in gold or not _has_content(row):
            continue
        g = gold[sid]
        scenario = {"scenario_id": sid, "hidden_gold": g["hidden_gold"]}
        pred = _annotation_to_prediction(row)
        metrics = score_prediction(scenario, pred)
        record = {
            "scenario_id": sid,
            "metrics": metrics,
            "expected_decision": g["hidden_gold"].get("expected_decision"),
            "response": pred["response"],
        }
        scored.append(record)
        per_pattern[g.get("pattern", "unknown")].append(record)
        per_difficulty[g.get("difficulty", "unknown")].append(record)

    def _joint(rows: list[dict[str, Any]]) -> float:
        return aggregate_metrics(rows).get("all_metrics", {}).get("joint_revision_success", 0.0) if rows else 0.0

    summary = aggregate_metrics(scored).get("all_metrics", {}) if scored else {}
    return {
        "n_scored": len(scored),
        "metrics": summary,
        "per_pattern": {k: _joint(v) for k, v in per_pattern.items()},
        "per_difficulty": {k: _joint(v) for k, v in per_difficulty.items()},
        "_scored": scored,
    }


def _quality_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    content = [r for r in rows if _has_content(r)]
    n = len(content) or 1

    def rate(field: str, positive: str) -> float:
        return sum(1 for r in content if (r.get(field) or "").strip() == positive) / n

    return {
        "solvable_rate": rate("solvable_from_visible_evidence", "yes"),
        "topic_domain_consistency_rate": rate("topic_domain_consistent", "yes"),
        "ambiguity_rate": rate("ambiguous_or_multiple_valid_answers", "yes"),
        "filler_heavy_rate": rate("filler_heavy", "yes"),
        "n": len(content),
    }


# ---------------------------------------------------------------------- driver

def score(annotation_paths: list[Path], gold_path: Path) -> dict[str, Any]:
    gold = _read_gold(gold_path)
    annotators: dict[str, list[dict[str, Any]]] = {}
    for path in annotation_paths:
        rows = _read_annotations(path)
        # Group by annotator_id; default to filename stem.
        for r in rows:
            aid = (r.get("annotator_id") or "").strip() or path.stem
            annotators.setdefault(aid, []).append(r)

    annotators = {aid: rows for aid, rows in annotators.items() if any(_has_content(r) for r in rows)}

    result: dict[str, Any] = {
        "gold_scenarios": len(gold),
        "annotators": sorted(annotators),
        "n_annotators": len(annotators),
        "per_annotator": {},
        "quality_rates": {},
        "iaa": {},
        "human_upper_bound": {},
        "status": "protocol prepared; human annotation pending",
    }
    if not annotators:
        return result

    # Per-annotator vs gold + quality rates.
    for aid, rows in annotators.items():
        vg = _score_annotator_vs_gold(rows, gold)
        result["per_annotator"][aid] = {
            "n_scored": vg["n_scored"],
            "metrics": vg["metrics"],
            "per_pattern_joint": vg["per_pattern"],
            "per_difficulty_joint": vg["per_difficulty"],
            "quality_rates": _quality_rates(rows),
        }

    # Pooled quality rates.
    all_rows = [r for rows in annotators.values() for r in rows]
    result["quality_rates"] = _quality_rates(all_rows)

    # Human upper bound: best per-scenario joint across annotators, pooled.
    best_by_scenario: dict[str, float] = {}
    decision_acc_pool: list[float] = []
    for aid in annotators:
        for rec in _score_annotator_vs_gold(annotators[aid], gold)["_scored"]:
            j = rec["metrics"].get("joint_revision_success", 0.0)
            sid = rec["scenario_id"]
            best_by_scenario[sid] = max(best_by_scenario.get(sid, 0.0), j)
            decision_acc_pool.append(rec["metrics"].get("black_box_decision_accuracy", 0.0))
    if best_by_scenario:
        result["human_upper_bound"] = {
            "scenarios": len(best_by_scenario),
            "joint_revision_success_any_annotator": sum(best_by_scenario.values()) / len(best_by_scenario),
            "pooled_decision_accuracy": (sum(decision_acc_pool) / len(decision_acc_pool)) if decision_acc_pool else 0.0,
        }

    # IAA (needs >= 2 annotators).
    if len(annotators) >= 2:
        # Align by scenario_id per annotator.
        aligned: dict[str, dict[str, dict[str, Any]]] = {
            aid: {r["scenario_id"]: r for r in rows if _has_content(r) and r.get("scenario_id")}
            for aid, rows in annotators.items()
        }
        common = set.intersection(*[set(d) for d in aligned.values()]) if aligned else set()
        iaa: dict[str, Any] = {"common_scenarios": len(common), "pairwise_cohens_kappa": {}, "pairwise_evidence": {}, "memory_state_agreement": {}}
        for field in CATEGORICAL_IAA_FIELDS:
            iaa["pairwise_cohens_kappa"][field] = {}
        for a, b in itertools.combinations(sorted(aligned), 2):
            pair = f"{a}|{b}"
            for field in CATEGORICAL_IAA_FIELDS:
                la = [(aligned[a][s].get(field) or "").strip() for s in sorted(common)]
                lb = [(aligned[b][s].get(field) or "").strip() for s in sorted(common)]
                iaa["pairwise_cohens_kappa"][field][pair] = _cohens_kappa(la, lb)
            ev_f1 = [_evidence_f1(aligned[a][s]["evidence_event_ids"], aligned[b][s]["evidence_event_ids"]) for s in common]
            ev_jac = [_jaccard(aligned[a][s]["evidence_event_ids"], aligned[b][s]["evidence_event_ids"]) for s in common]
            iaa["pairwise_evidence"][pair] = {
                "evidence_f1": (sum(ev_f1) / len(ev_f1)) if ev_f1 else None,
                "evidence_jaccard": (sum(ev_jac) / len(ev_jac)) if ev_jac else None,
            }
            # Per-memory agreement: fraction of shared memory_ids with same status.
            mem_agree: list[float] = []
            for s in common:
                ma, mb = aligned[a][s]["memory_state_json"], aligned[b][s]["memory_state_json"]
                keys = set(ma) & set(mb)
                if keys:
                    mem_agree.append(sum(1 for k in keys if ma[k] == mb[k]) / len(keys))
            iaa["memory_state_agreement"][pair] = (sum(mem_agree) / len(mem_agree)) if mem_agree else None
        # Krippendorff alpha (nominal) per categorical field across all annotators.
        iaa["krippendorff_alpha_nominal"] = {}
        for field in CATEGORICAL_IAA_FIELDS:
            by_unit: dict[str, list[str]] = defaultdict(list)
            for aid in aligned:
                for s in common:
                    by_unit[s].append((aligned[aid][s].get(field) or "").strip())
            iaa["krippendorff_alpha_nominal"][field] = _krippendorff_alpha_nominal(by_unit)
        result["iaa"] = iaa

    # Per-decision-label agreement vs gold (pooled across annotators).
    per_label: dict[str, list[int]] = defaultdict(list)
    for aid, rows in annotators.items():
        for r in rows:
            sid = r.get("scenario_id")
            if sid in gold and _has_content(r):
                exp = gold[sid]["hidden_gold"].get("expected_decision", "")
                per_label[exp].append(1 if (r.get("decision_label") or "").strip() == exp else 0)
    result["per_decision_label_agreement"] = {
        k: {"n": len(v), "agreement": sum(v) / len(v)} for k, v in per_label.items()
    }

    if len(annotators) >= 2:
        result["status"] = "paper-grade human validation completed"
    else:
        result["status"] = "quick author audit completed"
    return result


def _render_template(result: dict[str, Any]) -> str:
    pending = result["n_annotators"] == 0
    lines = ["# ReTrace-Bench — Human Validation Results", ""]
    lines.append(f"**Status:** {result['status']}")
    lines.append(f"**Annotators:** {result['n_annotators']} ({', '.join(result['annotators']) or 'none'})")
    lines.append(f"**Gold scenarios available:** {result['gold_scenarios']}")
    lines.append("")
    if pending:
        lines += [
            "> No human annotations have been supplied yet. This is an **empty template**.",
            "> Run `scripts/score_human_annotations.py --annotations <files...>` once at",
            "> least two real human annotators have completed the sheet. LLM/AI outputs",
            "> must NOT be entered as human annotations.",
            "",
            "## Dataset-quality rates (to be filled)",
            "| metric | value |",
            "|---|---|",
            "| solvable_rate | _pending_ |",
            "| topic_domain_consistency_rate | _pending_ |",
            "| ambiguity_rate | _pending_ |",
            "| filler_heavy_rate | _pending_ |",
            "",
            "## Human-vs-gold metrics (to be filled)",
            "| metric | value |",
            "|---|---|",
            "| decision accuracy | _pending_ |",
            "| memory_state_accuracy | _pending_ |",
            "| evidence_f1 | _pending_ |",
            "| minimal_evidence_exact_match | _pending_ |",
            "| failure_diagnosis_accuracy | _pending_ |",
            "| joint_revision_success | _pending_ |",
            "| human upper bound (joint, any annotator) | _pending_ |",
            "",
            "## Inter-annotator agreement (to be filled)",
            "- Cohen's kappa (decision_label, failure_diagnosis, solvable, topic, ambiguity): _pending_",
            "- Krippendorff's alpha (nominal, 3+ annotators): _pending_",
            "- Evidence pairwise F1 / Jaccard: _pending_",
            "- Memory-state per-memory agreement: _pending_",
        ]
        return "\n".join(lines) + "\n"

    q = result["quality_rates"]
    lines += [
        "## Dataset-quality rates (pooled)",
        f"- scenarios with annotations: {q.get('n', 0)}",
        "| metric | value |",
        "|---|---|",
        f"| solvable_rate | {q.get('solvable_rate', 0):.3f} |",
        f"| topic_domain_consistency_rate | {q.get('topic_domain_consistency_rate', 0):.3f} |",
        f"| ambiguity_rate | {q.get('ambiguity_rate', 0):.3f} |",
        f"| filler_heavy_rate | {q.get('filler_heavy_rate', 0):.3f} |",
        "",
        "## Human-vs-gold metrics (per annotator)",
        "| annotator | n | decision_acc | memory_state_acc | evidence_f1 | min_evidence_em | failure_diag_acc | joint |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for aid, pa in result["per_annotator"].items():
        m = pa["metrics"]
        lines.append(
            f"| {aid} | {pa['n_scored']} | {m.get('black_box_decision_accuracy', 0):.3f} | "
            f"{m.get('memory_state_accuracy', 0):.3f} | {m.get('evidence_f1', 0):.3f} | "
            f"{m.get('minimal_evidence_exact_match', 0):.3f} | {m.get('failure_diagnosis_accuracy', 0):.3f} | "
            f"{m.get('joint_revision_success', 0):.3f} |"
        )
    hub = result.get("human_upper_bound", {})
    if hub:
        lines += [
            "",
            "## Human upper bound (pooled)",
            f"- scenarios: {hub.get('scenarios', 0)}",
            f"- joint_revision_success (any annotator correct): {hub.get('joint_revision_success_any_annotator', 0):.3f}",
            f"- pooled decision accuracy: {hub.get('pooled_decision_accuracy', 0):.3f}",
        ]
    iaa = result.get("iaa")
    if iaa:
        lines += ["", "## Inter-annotator agreement", f"- common scenarios: {iaa['common_scenarios']}", "", "### Cohen's kappa (pairwise)"]
        for field, pairs in iaa["pairwise_cohens_kappa"].items():
            rendered = ", ".join(f"{p}={'NA' if v is None else f'{v:.3f}'}" for p, v in pairs.items())
            lines.append(f"- {field}: {rendered}")
        lines.append("")
        lines.append("### Krippendorff's alpha (nominal)")
        for field, v in iaa.get("krippendorff_alpha_nominal", {}).items():
            lines.append(f"- {field}: {'NA' if v is None else f'{v:.3f}'}")
        lines.append("")
        lines.append("### Evidence & memory-state agreement (pairwise)")
        for pair, ev in iaa["pairwise_evidence"].items():
            f1 = ev["evidence_f1"]
            jac = ev["evidence_jaccard"]
            mem = iaa["memory_state_agreement"].get(pair)
            lines.append(
                f"- {pair}: evidence_f1={'NA' if f1 is None else f'{f1:.3f}'}, "
                f"jaccard={'NA' if jac is None else f'{jac:.3f}'}, "
                f"memory_state={'NA' if mem is None else f'{mem:.3f}'}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, nargs="*", default=[],
                    help="Annotation CSV/JSONL files, one (or more) per annotator. Empty => emit template only.")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = ap.parse_args(argv)

    result = score(args.annotations, args.gold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "human_validation_results.json").write_text(
        json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rendered = _render_template(result)
    (args.out_dir / "human_validation_results.md").write_text(rendered, encoding="utf-8")
    args.template.write_text(rendered, encoding="utf-8")

    print(f"status: {result['status']}")
    print(f"annotators: {result['n_annotators']}")
    print(f"wrote {args.out_dir / 'human_validation_results.json'}")
    print(f"wrote {args.template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
