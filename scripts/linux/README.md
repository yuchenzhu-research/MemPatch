# Linux CUDA paper pipeline

One multitask QLoRA train per backbone on **train3500 (L3)** → pick checkpoint on **fixed L3 val partition** → eval Path A/Path B on **test500 (L4)**.

The full comparison produces Path A+DPA, paired Path A no-DPA, Path B base/LoRA, and seven public-input baseline proxy rows. Subset evaluation adds BM25 and skips the Path B base row by default.

**Backbones:** `mistral_nemo_12b`, `gemma3_12b`, `qwen3_14b`

## Data layout

```text
$LOCAL_ROOT/data/mempatch/train/scenarios.jsonl   # 3500, L3
$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl    # 500, L4 (never for checkpoint pick)
```

No runtime HF dataset download when these files exist. HF is only for **base model weight prefetch**.

## Full paper

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export HF_HOME=$LOCAL_ROOT/hf_cache
export RUN_ID=full512
cd /root/autodl-tmp/MemPatch && git pull

bash scripts/linux/run_paper_campaign.sh
```

## Per-model phases (`run_model.sh`)

```text
prefetch → multitask train (512 steps) → pick checkpoint → Path A/Path B test500 → baselines
```

```bash
SLUG=mistral_nemo_12b PHASES=train,pick,eval,baselines bash scripts/linux/run_model.sh
```

## Fast subset eval (8+1, skip base)

```bash
SLUG=mistral_nemo_12b EVAL_LIMIT=15 bash scripts/linux/run_eval_subset.sh
# or EVAL_LIMIT=25
```

Output: `$LOCAL_ROOT/results_eval_test{N}/{slug}/`

## Offline resume + 8+1 smoke

This smoke requires all three model directories and both dataset JSONL files to already exist locally. It sets the Hugging Face model, Transformers, and Datasets offline flags, trains each backbone to step 1, resumes to step 2, then runs eight baseline proxies plus Path A LoRA/DPA on one case each. The paired no-DPA artifact is also required; action quality is not enforced after only two training steps.

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
bash scripts/linux/run_resume_8plus1_smoke.sh
```

Artifacts, package versions, checkpoint audits, warnings, and one-case metrics are kept under `$LOCAL_ROOT/smoke/resume_8plus1/<timestamp>/`. Set `SMOKE_FAIL_ON_WARNINGS=1` to fail on detected deprecation/future-warning text.

## Training partition (not k-fold CV)

Within train3500, a **fixed** stratified 80/20 scenario split (`SPLIT_PARTS=5`, `SPLIT_INDEX=0`): ~2800 SFT-train scenarios and ~700 held-out val scenarios. Each scenario yields two independent SFT rows: one direct five-field response and one typed-action array, producing 5600 train rows and 1400 val rows. Checkpoint selection uses their mixed val loss.

**L3 vs L4:** val and SFT are L3; test500 is L4. Test scores measure generalization to harder cases, not seen during training.

## Artifacts

```text
$LOCAL_ROOT/train_data/splits/{slug}_split0/
$LOCAL_ROOT/adapters/{slug}_multitask_lora/split0/full512/checkpoint-*
$LOCAL_ROOT/logs/splits/{slug}_split0/full512/trainer_metrics.json
$LOCAL_ROOT/results/{slug}/
```

## Step scripts

| Script | Role |
|--------|------|
| `02_prepare_split.sh` | Build train/valid JSONL |
| `03_train.sh` | QLoRA train |
| `05_pick_best.sh` | Pick checkpoint |
| `06_eval_test.sh` | Path B test500 |
| `07_eval_path_a.sh` | Path A DPA + same-action no-DPA test500 |
| `run_baseline_matrix.sh` | Baselines |
| `run_eval_subset.sh` | Subset 8+1 |
