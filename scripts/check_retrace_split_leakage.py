#!/usr/bin/env python3
"""Exact-overlap leakage checker for the ReTrace train / dev / test splits.

Fails (non-zero exit) if any of the following overlap across splits:

* ``scenario_id``
* ``memory_id`` (initial memory + introduced/replacement memories)
* ``event_id``
* exact full public event text
* exact expected answer (``hidden_gold.expected_answer``)

It also reports prefix / seed-range disjointness and counts by domain, failure
mode, and expected decision. Semantic leakage detection is intentionally out of
scope; exact overlap is sufficient for today's package.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.analyze_retrace_template_signatures import event_text_templates, scenario_signature
from scripts.run_template_lookup_baseline import evaluate as evaluate_template_lookup


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario_ids(rows: list[dict]) -> set[str]:
    return {r["scenario_id"] for r in rows}


def memory_ids(rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        for m in r["public_input"]["initial_memory"]:
            out.add(m["memory_id"])
        for m in r["hidden_gold"].get("expected_memory_state", {}):
            out.add(m)
        rubric = r["hidden_gold"].get("rubric", {})
        for m in rubric.get("introduced_memories", {}):
            out.add(m)
    return out


def event_ids(rows: list[dict]) -> set[str]:
    return {e["event_id"] for r in rows for e in r["public_input"]["event_trace"]}


def event_texts(rows: list[dict]) -> set[str]:
    return {e["text"] for r in rows for e in r["public_input"]["event_trace"]}


def expected_answers(rows: list[dict]) -> set[str]:
    return {r["hidden_gold"]["expected_answer"] for r in rows}


def id_prefixes(rows: list[dict]) -> set[str]:
    return {r["scenario_id"].rsplit("-", 1)[0] + "-" for r in rows}


def seed_range(rows: list[dict]) -> tuple[int, int]:
    seeds = [r["metadata"]["seed"] for r in rows if "seed" in r.get("metadata", {})]
    return (min(seeds), max(seeds)) if seeds else (-1, -1)


def _pairwise_overlap(name: str, sets: dict[str, set]) -> list[str]:
    errors: list[str] = []
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = sets[a] & sets[b]
            if common:
                sample = sorted(list(common))[:5]
                errors.append(f"{name} overlap between {a} and {b}: {len(common)} shared, e.g. {sample}")
    return errors


def check(train: list[dict], dev: list[dict], test: list[dict]) -> tuple[list[str], dict]:
    splits = {"train": train, "dev": dev, "test": test}
    errors: list[str] = []
    errors += _pairwise_overlap("scenario_id", {k: scenario_ids(v) for k, v in splits.items()})
    errors += _pairwise_overlap("memory_id", {k: memory_ids(v) for k, v in splits.items()})
    errors += _pairwise_overlap("event_id", {k: event_ids(v) for k, v in splits.items()})
    errors += _pairwise_overlap("public_event_text", {k: event_texts(v) for k, v in splits.items()})
    errors += _pairwise_overlap("expected_answer", {k: expected_answers(v) for k, v in splits.items()})

    report = {}
    for k, v in splits.items():
        report[k] = {
            "count": len(v),
            "scenario_id_prefixes": sorted(id_prefixes(v)),
            "seed_range": list(seed_range(v)),
            "by_domain": dict(sorted(Counter(r["domain"] for r in v).items())),
            "by_failure_mode": dict(sorted(Counter(r["primary_failure_mode"] for r in v).items())),
            "by_expected_decision": dict(sorted(Counter(r["hidden_gold"]["expected_decision"] for r in v).items())),
        }
    return errors, report


def template_report(train: list[dict], dev: list[dict], test: list[dict], train_path: str, test_path: str) -> dict:
    train_sigs = {scenario_signature(r) for r in train}
    dev_sigs = {scenario_signature(r) for r in dev}
    test_sigs = {scenario_signature(r) for r in test}
    train_event_templates = {template for row in train for template in event_text_templates(row)}
    test_event_templates = {template for row in test for template in event_text_templates(row)}
    lookup = evaluate_template_lookup(Path(train_path), Path(test_path))
    return {
        "train_test_signature_overlap_count": len(train_sigs & test_sigs),
        "dev_test_signature_overlap_count": len(dev_sigs & test_sigs),
        "train_test_signature_overlap_rate": 0.0 if not test_sigs else len(train_sigs & test_sigs) / len(test_sigs),
        "train_test_event_template_overlap_count": len(train_event_templates & test_event_templates),
        "template_lookup_coverage_rate": lookup["coverage_rate"],
        "template_lookup_decision_accuracy": lookup["decision_accuracy"],
        "template_lookup_failure_mode_accuracy": lookup["failure_mode_accuracy"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--strict-template-heldout", action="store_true")
    args = parser.parse_args(argv)

    train, dev, test = load(args.train), load(args.dev), load(args.test)
    errors, report = check(train, dev, test)
    tmpl = template_report(train, dev, test, args.train, args.test)
    report["template_level"] = tmpl
    if args.strict_template_heldout:
        if tmpl["train_test_signature_overlap_rate"] > 0.10:
            errors.append(
                f"train->test signature overlap {tmpl['train_test_signature_overlap_rate']:.3f} > 0.10"
            )
        if tmpl["template_lookup_decision_accuracy"] > 0.55:
            errors.append(
                f"template lookup decision accuracy {tmpl['template_lookup_decision_accuracy']:.3f} > 0.55"
            )

    if args.json:
        print(json.dumps({"errors": errors, "report": report}, indent=2))
    else:
        for k, r in report.items():
            if k == "template_level":
                continue
            print(f"[{k}] count={r['count']} prefixes={r['scenario_id_prefixes']} seed_range={r['seed_range']}")
            print(f"  domains={len(r['by_domain'])}/8 modes={len(r['by_failure_mode'])}/11 decisions={r['by_expected_decision']}")
        if errors:
            print("\nLEAKAGE DETECTED:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\nOK: no scenario_id / memory_id / event_id / public-text / expected-answer overlap.")
        print(
            "Template diagnostics: "
            f"train_test_signature_overlap={tmpl['train_test_signature_overlap_count']} "
            f"({tmpl['train_test_signature_overlap_rate']:.3f}), "
            f"event_template_overlap={tmpl['train_test_event_template_overlap_count']}, "
            f"lookup_decision_acc={tmpl['template_lookup_decision_accuracy']:.3f}, "
            f"lookup_failure_acc={tmpl['template_lookup_failure_mode_accuracy']:.3f}"
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
