#!/usr/bin/env python3
"""Generate the ``calibration_80_en`` ReTrace-Bench v1.0 smoke-test split.

``calibration_80_en`` (public split name: **`calibration`**) is a small,
clean, de-actionalized quickstart / smoke-test split. It is **not** for
headline benchmark claims, checkpoint selection, tuning, or model selection.

It is regenerated clean (the legacy ``sample_80_hard_en`` split embedded action
verbs in its verified events and is intentionally not reused). It uses a
disjoint id / entity-number namespace from ``main_3000_en`` so the two splits
share no scenario / memory / event ids and no exact public text.

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

CALIBRATION_CONFIG = SplitConfig(
    split="calibration",
    sid_prefix="rb-cal-en-",
    renderer="calibration_v1_deactionalized",
    seed=4040000,
    case_base=620000,
    person_base=910000,
    secret_base=940000,
    scope_tag="CB",
    project_tag="CB",
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
    text = f"""# ReTrace-Bench `calibration_80_en` (v{BENCHMARK_VERSION})

Small, clean, de-actionalized **smoke-test / quickstart** split of
ReTrace-Bench v1.0 (public split name: **`calibration`**).

> **Not** for headline benchmark claims, checkpoint selection, tuning, or model
> selection. Use `main_3000_en` for main results and `hard_300_en` for the
> stress evaluation. This split exists only to let you confirm an evaluation
> pipeline runs end-to-end in seconds.

- **Scenarios:** {manifest['scenario_count']}
- **Source type:** `{manifest['source_type']}`
- **Annotation status:** `{manifest['annotation_status']}`
- **Benchmark version:** `{manifest['version']}`

Decision-word leakage audit:
`scenarios_with_decision_word_leak = {audit['scenarios_with_decision_word_leak']}`
(`clean = {str(audit['clean']).lower()}`).

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_calibration_80.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \\
  --data data/retrace_bench/calibration_80_en/scenarios.jsonl
```
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--out", default="data/retrace_bench/calibration_80_en/scenarios.jsonl")
    parser.add_argument("--seed", type=int, default=CALIBRATION_CONFIG.seed)
    args = parser.parse_args(argv)

    config = CALIBRATION_CONFIG
    if args.seed != CALIBRATION_CONFIG.seed:
        config = SplitConfig(**{**CALIBRATION_CONFIG.__dict__, "seed": args.seed})
    rows = build_split(args.count, config)

    out = Path(args.out)
    write_jsonl(out, rows)
    manifest = build_manifest(
        rows,
        split="calibration",
        source_type="controlled_synthetic",
        annotation_status="synthetic_gold",
        role="Quickstart / smoke-test split only (not for model selection).",
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
