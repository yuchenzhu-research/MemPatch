# Linux CUDA paper pipeline

CUDA + Hugging Face QLoRA + transformers inference for MemPatch-Bench test500.
Mac MLX scripts under `scripts/workflows/` stay unchanged; use this directory on AutoDL / Linux GPU nodes.

## End-to-end flow

```text
(0) setup          venv, pip [cuda], huggingface-cli login, audit gate
        │
(1) SFT bundles    prepare_mempatch_v13_smoke.py × 5 folds  → train.jsonl + valid.jsonl
        │
(2) QLoRA train    train_qlora.py × fold 0..4  → checkpoints @ 64/128/192/256
        │
(3) select fold    pick_best_kfold_fold.py  → lowest valid loss across folds
        │
(4) select ckpt    pick_best_checkpoint.py  → best iter on winning fold
        │
(5) test500 eval   run_hf_test_eval.py  base (NO_ADAPTER) + lora (best ckpt)
        │
(6) baselines      (next) 8 external + mempatch_lora row — same predictions contract
        │
(7) score          evaluate_mempatch_predictions.py  → headline table
```

**Selection protocol (same as Mac):**

| Step | Rule |
|------|------|
| K-fold | Stratified 5-fold on train3500 → 2800 train + 700 valid per fold |
| Best fold | Lowest **valid loss** at end of training |
| Best checkpoint | Among saves at **64 / 128 / 192 / 256**, pick lowest valid loss |
| Final eval | Held-out **test500** only (never used for selection) |

**Artifacts (all under `local/`, gitignored):**

```text
local/train_data/kfold/{slug}_fold{N}/     train.jsonl, valid.jsonl
local/adapters/{slug}_pathB_lora/fold{N}/full256/
local/logs/kfold/{slug}_fold{N}/           trainer_metrics.json
local/results/{slug}/                      predictions + metrics JSON
```

## Cursor Cloud / AutoDL smoke (run before full paper)

**Fix:** `06_eval_test.sh` reads `checkpoint_selection.json` (not subprocess `export`).

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local   # or Cursor cloud data path
export HF_HOME=$LOCAL_ROOT/hf_cache
mkdir -p "$LOCAL_ROOT" "$HF_HOME"

cd MemPatch
git pull
bash scripts/linux/00_setup.sh
huggingface-cli login

# One command: 10-step train, resume probe, 5-fold, pick-best, with/without eval (20 cases)
SLUG=llama3_1_8b bash scripts/linux/run_smoke_test.sh
```

Smoke defaults: `TRAIN_ITERS=10`, `SAVE_EVERY=2`, `RUN_ID=smoke10`, `KFOLDS=5`, `EVAL_LIMIT=20`.
Override example: `TRAIN_ITERS=10 SAVE_EVERY=2 SLUG=gemma3_12b bash scripts/linux/run_smoke_test.sh`

8+1 baselines: `run_baseline_matrix.sh` is not in repo yet; smoke skips that step with a notice.

## Quick start (one model)

```bash
# On AutoDL: clone repo, expand data disk ≥200GB, pick PyTorch 2.3+ / CUDA 12 image
cd MemPatch
bash scripts/linux/00_setup.sh

export HF_TOKEN=hf_...   # or: huggingface-cli login
SLUG=gemma3_12b bash scripts/linux/run_paper_model.sh
```

## Step-by-step commands

### 0 — Setup

```bash
bash scripts/linux/00_setup.sh
huggingface-cli login
```

### 1 — Audit (must pass before training)

```bash
bash scripts/linux/01_audit.sh
```

### 2 — Prepare k-fold SFT (one fold)

```bash
KFOLD_FOLD=0 SLUG=gemma3_12b bash scripts/linux/02_prepare_kfold.sh
```

### 3 — Train one fold

```bash
KFOLD_FOLD=0 SLUG=gemma3_12b bash scripts/linux/03_train_fold.sh
```

### 4 — Train all 5 folds

```bash
SLUG=gemma3_12b bash scripts/linux/04_train_all_folds.sh
```

### 5 — Pick best fold + checkpoint

```bash
SLUG=gemma3_12b bash scripts/linux/05_pick_best.sh
# exports BEST_FOLD, BEST_CHECKPOINT for eval
```

### 6 — Eval test500 (base + LoRA)

```bash
SLUG=gemma3_12b bash scripts/linux/06_eval_test.sh
```

### 7 — Score only (existing predictions)

```bash
PYTHONPATH=.:src python scripts/workflows/evaluate_mempatch_predictions.py \
  --data hf_release/mempatch/test/scenarios.jsonl \
  --predictions local/results/gemma3_12b/test500_lora_best_predictions.jsonl \
  --out-scored local/results/gemma3_12b/test500_lora_best_scored.jsonl \
  --no-strict --print-table
```

Each HF eval run also writes a **rich artifact bundle** under `local/results/{slug}/`:

```text
{run_tag}_predictions.jsonl      # raw_output + parse_error preserved
{run_tag}_scored.jsonl           # per-case metrics + validation_errors
{run_tag}_validation_errors.jsonl
{run_tag}_metrics.json
{run_tag}_manifest.json
```

Headline metric `response_schema_compliance_rate` counts invalid JSON schema outputs
(missing fields, hallucinated event IDs, invalid enums) — not only parse failures.

## Four paper models

| SLUG | Default HF hub ID | VRAM note (QLoRA 4-bit) |
|------|-------------------|-------------------------|
| `qwen3_14b` | `OpenPipe/Qwen3-14B-Instruct` | ~20–24 GB |
| `gemma3_12b` | `google/gemma-3-12b-it` | ~18–22 GB |
| `mistral_nemo_12b` | `mistralai/Mistral-Nemo-Instruct-2407` | ~18–22 GB |
| `llama3_1_8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ~12–16 GB |

Override any row: `export HF_MODEL_GEMMA3_12B=...` before running.

## Mac vs Linux

| | Mac (MLX) | Linux (this dir) |
|--|-----------|------------------|
| Train | `scripts/workflows/run_kfold_train.sh` | `scripts/linux/03_train_fold.sh` |
| Eval | `scripts/eval/run_lora_test_eval.py` | `scripts/linux/run_hf_test_eval.py` |
| Adapters | MLX safetensors | HF PEFT adapter — **not interchangeable** |
| Scoring | `benchmark.api` | same |

Mac-only scratch workflows belong in `scripts/_mac/` (gitignored).

## Inference contract (unchanged)

- Temperature **0.0**, max tokens **256**
- Output: one JSON object with five `response.*` fields
- Scorer: `evaluate_predictions(..., strict=False)` for paper runs
