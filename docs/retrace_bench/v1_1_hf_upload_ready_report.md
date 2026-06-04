# ReTrace-Bench — Hugging Face Upload-Ready Report

Preflight status for publishing the canonical ReTrace-Bench (internally "v1.1")
to Hugging Face. **This report does not publish anything.** Publishing is a
separate, explicitly-instructed manual step (see `v1_1_hf_release_plan.md`).

Generated from local bundle `hf_release/retrace_bench_v1_1/`, rebuilt via
`python scripts/build_hf_release_v1_1.py`.

## Target

- **HF dataset repo:** `Sylvan-Vale-Moon/ReTrace-Bench` (`--repo-type dataset`)
- **License:** CC BY 4.0 (data) — see `DATASET_LICENSE.md`.

## Exact files to upload

Upload the **entire** `hf_release/retrace_bench_v1_1/` tree:

```
DATASET_LICENSE.md
README.md                      # dataset card
VERSION
dataset_info.json
manifest.json                  # top-level manifest
checksums.json                 # sha256 over the 4 split JSONL files
main/scenarios.jsonl           # + main/manifest.json
hard/scenarios.jsonl           # + hard/manifest.json
realistic/scenarios.jsonl      # + realistic/manifest.json
calibration/scenarios.jsonl    # + calibration/manifest.json
```

(The split `scenarios.jsonl` files are git-ignored on GitHub and rebuilt locally
before upload; the metadata/manifests/checksums are committed.)

## Row counts (verified this pass)

| Split | Rows | Notes |
|---|---:|---|
| `main` | 3000 | canonical evaluation |
| `hard` | 500 | difficulty-balanced canonical |
| `realistic` | 200 | **`synthetic_gold_unreviewed`** (not yet human-validated) |
| `calibration` | 80 | **smoke-only** (wiring/format checks, not headline) |
| **Public total** | **3780** | |

## Private hidden — EXCLUDED

- `private_hidden` (200) is a held-out contamination probe. It is **not** in the
  bundle, **not** committed to GitHub, and **must not** be uploaded to Hugging
  Face. Verified: no `private_hidden` path appears in
  `hf_release/retrace_bench_v1_1/`.

## Preflight results

- **Schema/leakage validation** (`validate_retrace_bench_dataset.py`):
  - `main`: 0 errors, 0 warnings
  - `hard`: 0 errors, 0 warnings
  - `realistic`: 0 errors, **200 expected** `synthetic_gold_unreviewed` warnings
  - `calibration`: 0 errors, 0 warnings
- **Gold-oracle replay** (`check_retrace_bench_gold_oracle.py`, `hard`):
  `pass = true`; decision_accuracy = memory_state_accuracy =
  minimal_evidence_exact_match = evidence_f1 = failure_diagnosis_accuracy =
  joint_revision_success = **1.0**; `format_failure_rate = 0.0`;
  `stale_reuse_rate = 0.0`.
- **Checksums** (`checksums.json`, sha256): 4/4 split JSONL files match, 0
  problems → **checksums OK**.

## Status flags to carry into the dataset card / paper

- `realistic` = **`synthetic_gold_unreviewed`** until independent human
  validation completes (see `human_validation_status.md`).
- `calibration` = **smoke-only**; do not use as a headline evaluation split.
- Human validation: **Level 1 (automatic) complete; Levels 2–4 pending.** No IAA
  or human upper bound may be cited yet.

## Before overwriting the HF dataset

- The existing `Sylvan-Vale-Moon/ReTrace-Bench` repo currently hosts the
  **v1.0-era** release. **Archive/tag v1.0 first** (e.g. a `v1.0` git-style tag
  or a `legacy/` snapshot in the dataset repo) so it is not silently overwritten.
  See `v1_0_deprecation_notice.md` and `v1_1_hf_release_plan.md`.

## Manual publish command (run only when explicitly instructed)

```bash
export HF_TOKEN="<token provided out-of-band; never committed/printed>"
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli upload Sylvan-Vale-Moon/ReTrace-Bench \
  hf_release/retrace_bench_v1_1 . --repo-type dataset
unset HF_TOKEN
```

Do **not** upload `private_hidden`, `.env`, raw API responses, or local caches.
After upload, immediately `unset HF_TOKEN`.
