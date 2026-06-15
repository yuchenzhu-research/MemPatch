# Linux CUDA paper pipeline

One multitask QLoRA train per backbone on **train3500 (L3)** → pick checkpoint on **fixed L3 val partition** → eval Path A/Path B on **test500 (L4)**.

The formal comparison produces MemPatch, its paired no-guard diagnostic,
Final-State Control, and five frozen external baselines. Mem0/A-MEM outputs are
supplement-only when present.

**Campaign order:** `qwen3_14b`, `gemma3_12b`, `phi4`, `mistral_nemo_12b`

The five formal public-input baselines use the frozen base backbone and receive no
MemPatch training. They measure prompting, retrieval, recency, summarization,
and memory organization without benchmark-specific gradient updates. The
strict like-for-like mechanism comparison is Final-State Control versus
adapted Path A MemPatch: both use the same multitask LoRA checkpoint and
training examples. Report these as two separate comparison groups rather than
claiming that the frozen 7-baseline table isolates the scaffold alone.

## Data layout

```text
$LOCAL_ROOT/data/mempatch/train/scenarios.jsonl   # 3500, L3
$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl    # 500, L4 (never for checkpoint pick)
```

No runtime HF dataset download when these files exist. With local base weights under `$LOCAL_ROOT/models/`, no Hugging Face login is required.

## Smoke Test

The smoke pipeline uses exactly 30 deterministic test500 cases and no LoRA. It
runs Qwen3, Gemma-3, and Phi-4 in that order; Mistral is excluded.

```bash
bash scripts/linux/run_experiment.sh smoke
```

Dry-run without model execution:

```bash
DRY_RUN=1 bash scripts/linux/run_experiment.sh smoke
```

Systems are Frozen Direct Prompting, Full Context, and MemPatch Zero-Shot.

## Formal Test

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export HF_HOME=$LOCAL_ROOT/hf_cache
cd /root/autodl-tmp/MemPatch && git pull

bash scripts/linux/run_experiment.sh formal
```

The component commands `run_formal_frozen.sh`, `run_formal_adapted.sh`, and
`build_paper_results.sh` remain available for partial reruns.

The adapted pipeline uses 1,024 optimization
steps, validation/checkpointing every 128 steps, and retention of all eight
checkpoint candidates. It selects the lowest mixed-task L3 validation loss
before test500 inference. Final-State Control and MemPatch use the same selected
checkpoint. The fixed order is Qwen3, Gemma-3, Phi-4, then Mistral-Nemo.

Install the Linux QLoRA dependencies first with `bash scripts/linux/00_setup.sh`.

## Frozen 5+1 completion run

Qwen is already complete. Run the two remaining server models without training
or LoRA (Gemma first, then Phi-4):

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
bash scripts/linux/run_server_frozen_gemma_phi.sh
```

Each model writes five frozen baselines plus MemPatch Zero-Shot under
`$LOCAL_ROOT/results/final/{slug}`. The local Apple-silicon Mistral counterpart
is `scripts/apple_mlx/run_local_mistral_frozen.sh`.

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
prefetch → multitask train (1024 steps) → pick among checkpoints 128..1024 → Path A/Path B test500 → frozen baselines
```

All four campaign backbones default to `max_seq_length=2048`. Gemma/Qwen cap in-train eval to 512 val rows on 32GB GPUs. `MemorySafeSFTTrainer` overrides `prediction_step` so TRL does not materialize full-vocab logits at step 128. Override via `TRAIN_MAX_SEQ_LENGTH_<SLUG>`, `TRAIN_EVAL_MAX_SAMPLES_<SLUG>`.

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
$LOCAL_ROOT/adapters/{slug}_multitask_lora/split0/full1024/checkpoint-*
$LOCAL_ROOT/logs/splits/{slug}_split0/full1024/trainer_metrics.json
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
| `rescore_result_bundle.py` | Rescore preserved predictions without inference |
