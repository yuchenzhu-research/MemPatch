#!/usr/bin/env python3
"""Build the ReTrace-Bench (internal "v1.1") Hugging Face release bundle.

Copies the public paper-facing splits from ``data/retrace_bench_v1_1/`` into
``hf_release/retrace_bench_v1_1/`` and emits the dataset card, license, manifest,
checksums, ``dataset_info.json`` and ``VERSION``.

Policy:
- Public bundle = main / hard / realistic / calibration only (3780 cases).
- ``private_hidden_200_en`` is a private evaluation split. It is bundled ONLY
  when ``--include-private`` is passed, into a clearly separated
  ``private/`` subtree, and must never be published to the public HF dataset.
- The full ``scenarios.jsonl`` files are written locally for the user to upload;
  they are git-ignored. Only manifest/checksums/card/license/VERSION are tracked.
- This script does NOT upload anything to Hugging Face.

The bundle ships the canonical nested scenario rows (the same format the official
evaluator/validator consume), including ``hidden_gold`` — ReTrace-Bench is an
evaluation benchmark and must distribute gold so others can score. Do NOT train
on this data (see the dataset card and DATASET_LICENSE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "data" / "retrace_bench_v1_1"
OUT_ROOT = REPO_ROOT / "hf_release" / "retrace_bench_v1_1"

GITHUB_URL = "https://github.com/yuchenzhu-research/ReTrace"
HF_URL = "https://huggingface.co/datasets/Sylvan-Vale-Moon/ReTrace-Bench"
RELEASE_VERSION = "1.1.0"
GENERATION_SEED = 2027

# (on-disk split dir, public split name, expected count)
PUBLIC_SPLITS = (
    ("main_3000_en", "main", 3000),
    ("hard_500_en", "hard", 500),
    ("realistic_200_en", "realistic", 200),
    ("calibration_80_en", "calibration", 80),
)
PRIVATE_SPLIT = ("private_hidden_200_en", "private_hidden", 200)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def copy_split(src_dir: str, rel_out_dir: str) -> dict:
    src = SRC_ROOT / src_dir / "scenarios.jsonl"
    if not src.exists():
        raise SystemExit(f"missing source split: {src}")
    out_dir = OUT_ROOT / rel_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "scenarios.jsonl"
    shutil.copyfile(src, dst)
    # carry the existing per-split manifest if present
    src_manifest = SRC_ROOT / src_dir / "manifest.json"
    if src_manifest.exists():
        shutil.copyfile(src_manifest, out_dir / "manifest.json")
    n = count_rows(dst)
    return {
        "dir": rel_out_dir,
        "scenarios": str(dst.relative_to(OUT_ROOT)),
        "rows": n,
        "sha256": sha256_of(dst),
    }


def dataset_card(public_counts: dict[str, int]) -> str:
    total = sum(public_counts.values())
    return f"""---
license: cc-by-4.0
language:
- en
pretty_name: ReTrace-Bench
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- benchmark
- memory-revision
- evaluation
configs:
- config_name: main
  data_files: main/scenarios.jsonl
- config_name: hard
  data_files: hard/scenarios.jsonl
- config_name: realistic
  data_files: realistic/scenarios.jsonl
- config_name: calibration
  data_files: calibration/scenarios.jsonl
---

# ReTrace-Bench

ReTrace-Bench evaluates **agent memory-revision reliability**: whether a system
processes new evidence to update, block, release, reaffirm, or reject memory
states without introducing stale, out-of-scope, or policy-invalid beliefs. It
scores not only the final decision but also memory-state tracking, minimal
evidence grounding, and failure diagnosis.

> **Evaluation-only.** ReTrace-Bench is a benchmark for *evaluating* systems.
> Do **not** train models on it, and in particular do **not** train the separate
> *ReTrace-Learn* method on this data. Doing so invalidates results.

## Splits (public release)

| split | cases | purpose |
|---|---|---|
| `main` | {public_counts['main']} | broad coverage across domains, difficulties, failure modes |
| `hard` | {public_counts['hard']} | L3/L4 adversarial; minimal-evidence, no latest-event shortcut |
| `realistic` | {public_counts['realistic']} | realistic-style stress split; **`synthetic_gold_unreviewed`** |
| `calibration` | {public_counts['calibration']} | smoke / quickstart only |

**Public total: {total} cases.**

- `realistic` is `synthetic_gold_unreviewed`: its gold has **not** been human
  reviewed yet. Treat it as a secondary/stress split with a limitation note, not
  a headline split, until human validation is recorded.
- `calibration` is **smoke / quickstart only** — it is not a model-selection /
  checkpoint-selection validation set and must not be used to tune or select
  systems, nor for headline claims.
- A `private_hidden` split (200 cases) exists for private evaluation and is **not
  part of this public release**.

## Format

Each line is a JSON scenario object (the native format consumed by the official
evaluator). It includes a gold-free public input plus the `hidden_gold` block
used for scoring. The public-facing model input must be taken through the
official public view; do not feed `hidden_gold` or internal fields to a model.

## Scoring

Use the official evaluator from the GitHub repository
([{GITHUB_URL}]({GITHUB_URL})):

```bash
python scripts/evaluate_retrace_bench_predictions.py \\
  --data <split>/scenarios.jsonl --predictions <your_predictions>.jsonl
```

Core metrics: `decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`,
`minimal_evidence_exact_match`, `failure_diagnosis_accuracy`,
`joint_revision_success`, `format_failure_rate`.

## Licensing

- **Dataset:** CC BY 4.0 (see `DATASET_LICENSE.md`).
- **Code** (evaluator/validators/scripts on GitHub): MIT. The code and dataset
  licenses are separate.

## Provenance

- Deterministically generated with seed `{GENERATION_SEED}`.
- Release version `{RELEASE_VERSION}`.
- Code, schema, validators, and reproducible baselines: [{GITHUB_URL}]({GITHUB_URL}).

## Citation

```bibtex
@misc{{retracebench,
  title  = {{ReTrace-Bench: Evaluating Agent Memory-Revision Reliability}},
  author = {{ReTrace authors}},
  year   = {{2026}},
  note   = {{Evaluation-only benchmark. \\url{{{GITHUB_URL}}}}}
}}
```
"""


DATASET_LICENSE = """# ReTrace-Bench Dataset License

The ReTrace-Bench **dataset** (the scenario JSONL files in this bundle) is
licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt the dataset for any purpose, provided you give
appropriate credit.

## Note on intended use

ReTrace-Bench is an **evaluation-only** benchmark. While CC BY 4.0 permits
adaptation, the authors request that you do **not** use it as training data for
memory-revision systems (including ReTrace-Learn), since doing so contaminates
evaluation.

## Separate code license

The accompanying code (evaluator, validators, generation/scoring scripts) hosted
at https://github.com/yuchenzhu-research/ReTrace is licensed separately under the
MIT License. This dataset license (CC BY 4.0) applies only to the data files.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ReTrace-Bench v1.1 HF release bundle (no upload).")
    parser.add_argument("--include-private", action="store_true", help="also bundle the private_hidden split under private/ (local only)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    split_records = []
    public_counts: dict[str, int] = {}
    for src_dir, public_name, expected in PUBLIC_SPLITS:
        rec = copy_split(src_dir, public_name)
        if rec["rows"] != expected:
            raise SystemExit(f"{public_name}: expected {expected} rows, got {rec['rows']}")
        rec["split"] = public_name
        rec["visibility"] = "public"
        split_records.append(rec)
        public_counts[public_name] = rec["rows"]

    if args.include_private:
        src_dir, public_name, expected = PRIVATE_SPLIT
        rec = copy_split(src_dir, f"private/{public_name}")
        rec["split"] = public_name
        rec["visibility"] = "private"
        split_records.append(rec)

    public_total = sum(public_counts.values())

    # checksums.json
    checksums = {r["scenarios"]: r["sha256"] for r in split_records}
    (OUT_ROOT / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # manifest.json
    manifest = {
        "dataset": "ReTrace-Bench",
        "release_version": RELEASE_VERSION,
        "generation_seed": GENERATION_SEED,
        "public_total": public_total,
        "public_splits": [r for r in split_records if r["visibility"] == "public"],
        "private_splits": [r for r in split_records if r["visibility"] == "private"],
        "dataset_license": "CC-BY-4.0",
        "code_license": "MIT",
        "evaluation_only": True,
        "github": GITHUB_URL,
        "huggingface": HF_URL,
        "notes": {
            "realistic": "synthetic_gold_unreviewed until human validation is recorded",
            "calibration": "smoke/quickstart only; not for model selection or headline claims",
            "private_hidden": "private evaluation only; not part of the public release",
        },
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # dataset_info.json (lightweight HF-style descriptor)
    dataset_info = {
        "dataset_name": "ReTrace-Bench",
        "version": RELEASE_VERSION,
        "license": "cc-by-4.0",
        "splits": {
            r["split"]: {"num_examples": r["rows"], "data_file": r["scenarios"]}
            for r in split_records
            if r["visibility"] == "public"
        },
        "total_public_examples": public_total,
    }
    (OUT_ROOT / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (OUT_ROOT / "README.md").write_text(dataset_card(public_counts), encoding="utf-8")
    (OUT_ROOT / "DATASET_LICENSE.md").write_text(DATASET_LICENSE, encoding="utf-8")
    (OUT_ROOT / "VERSION").write_text(RELEASE_VERSION + "\n", encoding="utf-8")

    print(f"built HF bundle at {OUT_ROOT}")
    print(f"public total: {public_total}")
    for r in split_records:
        print(f"  [{r['visibility']}] {r['split']}: {r['rows']} rows  sha256={r['sha256'][:12]}…")


if __name__ == "__main__":
    main()
