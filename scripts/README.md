# Scripts

Paper reproduction runs on **Linux CUDA** (`scripts/linux/`). Shared Python utilities live under `scripts/data/`, `scripts/memory/`, and `scripts/mlx_support/` (chat/JSON helpers used by Linux HF eval).

## Public (no GPU)

| Script | Role |
|--------|------|
| `workflows/evaluate_mempatch_predictions.py` | Score any `predictions.jsonl` |
| `workflows/audit_decision_boundary.py` | Dataset gate before training |
| `data/generate_mempatch.py` | Regenerate scenario JSONL |
| `data/package_mempatch_release.py` | Manifest + checksums |

## Linux paper pipeline

| Script | Role |
|--------|------|
| `linux/run_paper_campaign.sh` | Three backbones end-to-end |
| `linux/run_model.sh` | One backbone (prefetch → train → pick → eval → baselines) |
| `linux/run_eval_subset.sh` | Fast 8 baseline proxies + Path A LoRA/DPA on N test cases (default 25) |
| `linux/06_eval_test.sh` | Path B test500 base + LoRA |
| `linux/07_eval_path_a.sh` | Path A typed actions: paired DPA + no-DPA test500 |
| `linux/run_resume_8plus1_smoke.sh` | Offline three-model resume + one-case 8+1 compatibility smoke |

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export RUN_ID=full512
SLUG=mistral_nemo_12b bash scripts/linux/run_model.sh
```

Dataset on disk: `$LOCAL_ROOT/data/mempatch/{train,test}/scenarios.jsonl` (3500 / 500).

Baseline proxy definitions and audit requirements: `memory/BASELINE_ADAPTER_AUDIT.md`.
