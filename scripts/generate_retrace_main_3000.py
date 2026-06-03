#!/usr/bin/env python3
"""Generate the clean, paper-facing ``main_3000_en`` ReTrace-Bench v1.0 split.

``main_3000_en`` is the primary controlled benchmark split: broad coverage
across the 8 domains, 11 failure modes, 5 decisions, and 4 difficulty tiers.
It is produced by the de-actionalized, leakage-audited builder in
``benchmark.retrace_bench.generation.deactionalized`` so no authoritative /
verified record contains a direct decision-action phrase (the gold decision
must be inferred from the described state).

This is **evaluation** data (``training_targets: false``); ``hidden_gold`` holds
evaluation gold only. SFT supervision lives separately under
``data/retrace_learn/``.

Determinism: output depends only on ``--count`` and ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.generation.deactionalized import SplitConfig, build_split
from benchmark.retrace_bench.generation.release_manifest import (
    BENCHMARK_VERSION,
    build_manifest,
)

MAIN_CONFIG = SplitConfig(
    split="main",
    sid_prefix="rt-main-en-",
    renderer="main_v1_deactionalized",
    seed=1010000,
    case_base=520000,
    person_base=810000,
    secret_base=930000,
    scope_tag="MN",
    project_tag="MN",
    extra_metadata={
        "benchmark_version": BENCHMARK_VERSION,
        "source_type": "controlled_synthetic",
        "annotation_status": "synthetic_gold",
    },
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(out_dir: Path, manifest: dict) -> None:
    audit = manifest["leakage_audit_summary"]
    text = f"""# ReTrace-Bench `main_3000_en` (v{BENCHMARK_VERSION})

Primary controlled benchmark split of ReTrace-Bench v1.0 (public split name:
**`main`**). It provides broad coverage across the 8 domains, 11 memory-revision
failure modes, 5 revision decisions, and 4 difficulty tiers, and is used for the
main benchmark results.

- **Scenarios:** {manifest['scenario_count']}
- **Source type:** `{manifest['source_type']}`
- **Annotation status:** `{manifest['annotation_status']}` (controlled synthetic gold)
- **Benchmark version:** `{manifest['version']}`
- **Schema:** `retrace_bench_general_1`
- **Training targets:** none (evaluation-only; `hidden_gold` is evaluation gold)

## Benchmark hygiene / leakage audit

Authoritative (verified/trusted) records are **de-actionalized**: each states a
fact or status and never begins with a final action verb (`Escalate…`,
`Refuse…`, `Ask for clarification…`, `Mark … unresolved`, `Use current
memory`). The gold decision must be recovered by reasoning over the described
state, not copied from a word.

Decision-word leakage audit over authoritative records:
`scenarios_with_decision_word_leak = {audit['scenarios_with_decision_word_leak']}`
(`clean = {str(audit['clean']).lower()}`).

## Scale

- avg / max events per scenario: {manifest['avg_event_count']} / {manifest['max_event_count']}
- avg / max memories per scenario: {manifest['avg_memory_count']} / {manifest['max_memory_count']}
- avg required evidence events per scenario: {manifest['avg_required_evidence_count']}

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_main_3000.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \\
  --data data/retrace_bench/main_3000_en/scenarios.jsonl
```
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--out", default="data/retrace_bench/main_3000_en/scenarios.jsonl")
    parser.add_argument("--seed", type=int, default=MAIN_CONFIG.seed)
    args = parser.parse_args(argv)

    config = MAIN_CONFIG
    if args.seed != MAIN_CONFIG.seed:
        config = SplitConfig(**{**MAIN_CONFIG.__dict__, "seed": args.seed})
    rows = build_split(args.count, config)

    out = Path(args.out)
    write_jsonl(out, rows)
    manifest = build_manifest(
        rows,
        split="main",
        source_type="controlled_synthetic",
        annotation_status="synthetic_gold",
        role="Primary controlled benchmark split (main benchmark results).",
    )
    out.parent.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_readme(out.parent, manifest)
    print(f"wrote {len(rows)} scenarios to {out}")
    print(f"leakage_audit clean={manifest['leakage_audit_summary']['clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
