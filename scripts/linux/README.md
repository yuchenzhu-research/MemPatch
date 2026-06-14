# Linux CUDA paper pipeline

One multitask QLoRA train per backbone on **train3500 (L3)** → pick checkpoint on **fixed L3 val partition** → eval Path A/Path B on **test500 (L4)**.

The full comparison produces Path A+DPA, paired Path A no-DPA, Path B base/LoRA, and seven public-input baseline proxy rows. Subset evaluation adds BM25 and skips the Path B base row by default.

**Backbones:** `mistral_nemo_12b`, `gemma3_12b`, `qwen3_14b`

## Data layout

```text
$LOCAL_ROOT/data/mempatch/train/scenarios.jsonl   # 3500, L3
$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl    # 500, L4 (never for checkpoint pick)
```

No runtime HF dataset download when these files exist. With local base weights under `$LOCAL_ROOT/models/`, no Hugging Face login is required.

## Full paper

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export HF_HOME=$LOCAL_ROOT/hf_cache
export RUN_ID=full512
cd /root/autodl-tmp/MemPatch && git pull

bash scripts/linux/run_paper_campaign.sh
```

To train all three backbones first, then run the seven main public baselines
plus MemPatch Path A on all 500 held-out cases for each backbone:

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export RUN_ID=full512
export CONFIRM_FRESH=1
bash scripts/linux/run_train_all_then_7plus1.sh
```

This is destructive only for the exact `RUN_ID` adapter/log directories and
the three model result directories. It preserves base model weights and the
train/test dataset. The 7+1 methods are Structured Direct, Full Context,
Vanilla RAG, Time-Aware RAG, Summary Memory, Mem0-style, A-MEM-style, and
MemPatch Path A LoRA+DPA. Path A also writes its paired no-DPA audit artifact.
The baseline generation cap remains the paper default of 256 tokens; set
`BASELINE_MAX_TOKENS=512 EVAL_LIMIT=20` only for an explicit truncation subset
diagnostic, not silently for the main table.

The campaign performs a CUDA/package preflight before deleting any artifacts.
Install the Linux QLoRA dependencies first with `bash scripts/linux/00_setup.sh`.

Existing prediction files can be rescored without Torch, a GPU, training, or
new inference. The output must be a separate directory so raw results remain
unchanged:

```bash
python scripts/linux/rescore_result_bundle.py \
  --data "$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl" \
  --source-results local/results \
  --out-results local/results_rescored
```

## Per-model phases (`run_model.sh`)

```text
prefetch → multitask train (512 steps) → pick checkpoint → Path A/Path B test500 → baselines
```

All three backbones default to `max_seq_length=2048`. Gemma/Qwen cap in-train eval to 512 val rows on 32GB GPUs. `MemorySafeSFTTrainer` overrides `prediction_step` so TRL does not materialize full-vocab logits at step 128. Override via `TRAIN_MAX_SEQ_LENGTH_<SLUG>`, `TRAIN_EVAL_MAX_SAMPLES_<SLUG>`.

```bash
SLUG=mistral_nemo_12b PHASES=train,pick,eval,baselines bash scripts/linux/run_model.sh
```

## Safe explicit rerun

Do not use `PHASES=auto` when replacing paper results. Remove artifacts for the
exact `RUN_ID`, then run every phase explicitly:

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export RUN_ID=full512
CONFIRM_RERUN=1 bash scripts/linux/rerun_qwen_mistral.sh
```

The script deletes only Qwen/Mistral adapters and logs under the exact
`split${SPLIT_INDEX}/${RUN_ID}` path, clears their result directories, runs
`PHASES=train,pick,eval,baselines`, and prints the diagnostic report. Existing
artifacts from names such as `full512_2048` are not treated as `full512`.

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
| `run_train_all_then_7plus1.sh` | Train all three, then full test500 7+1 |
| `rescore_result_bundle.py` | Rescore preserved predictions without inference |
