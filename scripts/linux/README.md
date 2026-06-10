# Linux CUDA paper pipeline

QLoRA train → pick best (5 folds × 4 checkpoints) → test500 with/without LoRA → 11+1 baselines.

**Three backbones (no Llama):** `mistral_nemo_12b`, `gemma3_12b`, `qwen3_14b`

## Why earlier runs failed (not your fault)

| Problem | Cause | Fix in this refactor |
|---------|--------|----------------------|
| `Fetching 5 files: 0%` forever | HF xet CDN from AutoDL | `HF_HUB_DISABLE_XET=1` + **prefetch to `LOCAL_MODEL_ROOT`** |
| `HF_HUB_OFFLINE` errors | Offline on by default with incomplete cache | **Removed** default offline; load from local dir after prefetch |
| screen died | Python process crashed on HF error | `run_model.sh` phases + `pipeline.log` |
| Restart from scratch | No phase detection | `PHASES=auto` skips completed steps |

## One command (start with Mistral)

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export HF_HOME=$LOCAL_ROOT/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_DOWNLOAD_WORKERS=1
export HF_TOKEN=hf_...

cd /root/autodl-tmp/MemPatch
git pull
bash scripts/linux/00_setup.sh

screen -dmS mempatch bash -lc '
  export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
  export HF_HOME=$LOCAL_ROOT/hf_cache
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_DOWNLOAD_WORKERS=1
  export HF_TOKEN=hf_...
  cd /root/autodl-tmp/MemPatch
  SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b) bash scripts/linux/run_paper_three.sh
'

screen -ls
tail -f /root/autodl-tmp/mempatch_local/logs/pipeline.log
```

## Pipeline per model (`run_model.sh`)

```text
prefetch   snapshot_download -> LOCAL_MODEL_ROOT/{hub-id-as-dir}/
train      5-fold QLoRA, saves @ 64/128/192/256 -> trainer_metrics.json
pick       best fold (lowest val loss) + best checkpoint on that fold
eval       test500 without adapter + with best LoRA
baselines  11 baselines + mempatch_lora_best (RESUME=1)
```

**Selection:** 5 folds × 4 checkpoints = **20 candidates**; pick lowest valid loss fold, then lowest valid loss step on that fold.

## Status / resume

```bash
bash scripts/linux/status_models.sh
SLUG=mistral_nemo_12b bash scripts/linux/status_models.sh
```

## Download triage

Start with one model and probe access before downloading weights:

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export HF_HOME=$LOCAL_ROOT/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_DOWNLOAD_WORKERS=1
export HF_TOKEN=hf_...

cd /root/autodl-tmp/MemPatch
SLUG=mistral_nemo_12b HF_PREFETCH_PROBE_ONLY=1 bash scripts/linux/prefetch_model.sh
SLUG=mistral_nemo_12b bash scripts/linux/prefetch_model.sh
```

If `gemma3_12b` fails at the probe, accept the Gemma terms on Hugging Face with the same account used by `HF_TOKEN`. If `hf-mirror.com` still hangs at `Fetching files: 0%`, retry the same command with `HF_ENDPOINT=` to use the official endpoint.

Mistral training already done on server → only runs prefetch (if needed), eval, baselines:

```bash
SLUG=mistral_nemo_12b PHASES=eval,baselines bash scripts/linux/run_model.sh
```

Or full auto (skips finished train/pick):

```bash
SLUG=mistral_nemo_12b bash scripts/linux/run_model.sh
```

## Remove Llama local artifacts

```bash
bash scripts/linux/clean_llama_local.sh
```

## Artifacts

```text
LOCAL_MODEL_ROOT/              full HF weights (use these, not hub cache)
local/adapters/{slug}_pathB_lora/fold{N}/full256/checkpoint-{64,128,192,256}
local/logs/kfold/{slug}_fold{N}/trainer_metrics.json
local/results/{slug}/          predictions + metrics + selection JSON
local/logs/pipeline.log        unified log
```

## Models

| SLUG | HF hub ID |
|------|-----------|
| `mistral_nemo_12b` | `mistralai/Mistral-Nemo-Instruct-2407` |
| `gemma3_12b` | `google/gemma-3-12b-it` |
| `qwen3_14b` | `OpenPipe/Qwen3-14B-Instruct` |

Override: `export HF_MODEL_GEMMA3_12B=/path/to/local/dir`

## Step scripts (still valid)

| Script | Role |
|--------|------|
| `prefetch_model.sh` | Download one model to `LOCAL_MODEL_ROOT` |
| `04_train_all_folds.sh` | Train all folds (used internally by `run_model.sh`) |
| `05_pick_best.sh` | Pick fold + checkpoint |
| `06_eval_test.sh` | test500 base + lora |
| `run_baseline_matrix.sh` | 11+1 baselines |
