# Data

## v1.1 (current HF: `Sylvan-Vale-Moon/MemPatch`)

Public scenarios: `main` 3000 + `hard` 500. Download:

```bash
python scripts/download_mempatch_dataset.py --out-dir local/MemPatch
```

Audit:

```bash
python scripts/report_split_decision_distribution.py \
  --split main local/MemPatch/main \
  --split hard local/MemPatch/hard
```

## v1.2 (HF release 1.2.0)

Target: five-decision `train` (2700) + `main` (800) + `hard` (500). See `docs/mempatch_v1_2_dataset_redesign_plan.md`.

Generate locally (JSONL gitignored) with `scripts/generate_mempatch_v12.py`, then package:

```bash
python scripts/package_mempatch_release.py \
  --input-dir local/mempatch_v12_export \
  --out-dir hf_release/mempatch_v1_2 \
  --base-manifest hf_release/mempatch_v1_2/manifest.json \
  --validate --report
```

Method-side training subsets: `data/retrace_learn/` (internal path).
