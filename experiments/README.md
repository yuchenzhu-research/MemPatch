# Unified Experiments

This directory is the only supported entrypoint for the evaluation campaign.
It does not call the historical `scripts/linux` or Apple launchers.

## Design

- `frozen_direct`, `full_context`, `lexical_rag`, `time_aware_rag`, and
  `summary_memory` use the same model and strict five-field response contract.
- The exact `frozen_direct` response is reused as the raw semantic response for
  both MemPatch variants.
- `mempatch` and `mempatch_no_guard` use the exact same generated typed actions.
  Their only difference is deterministic Revision Guard projection.
- Decoding is greedy (`temperature=0`, `do_sample=False`).
- Every completed scenario is flushed to `raw_cases.jsonl`; `--resume` skips it.
- Paper predictions are materialized with one canonical JSONL file per method.

## Server setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r experiments/requirements.txt
```

Before a full run, use a separate three-case smoke directory:

```bash
LIMIT=3 OUTPUT_ROOT=runs/eval_smoke bash experiments/run_all.sh qwen3_14b
python experiments/validate_run.py \
  --data local/data/mempatch/test/scenarios.jsonl \
  --run-dir runs/eval_smoke/qwen3_14b \
  --expected-cases 3
```

Run one model per GPU allocation:

```bash
bash experiments/run_all.sh qwen3_14b
bash experiments/run_all.sh phi4_14b
bash experiments/run_all.sh mistral_nemo_12b
```

Local model directories can replace Hub IDs:

```bash
QWEN3_MODEL_ID=/models/qwen3-14b bash experiments/run_all.sh qwen3_14b
```

After all three runs:

```bash
bash experiments/run_all.sh guard
bash experiments/run_all.sh analyze
```

The main paper inputs are:

- `paper_results/main_results.csv`
- `paper_results/paired_cluster_bootstrap.csv`
- `paper_results/efficiency.csv`
- `paper_results/interface_funnel.csv`
- each model's `guard_stress.json`
- each model's `run_manifest.json`

Use `metadata.decision_variant` as the paired bootstrap cluster. The generator
contains eight broad pattern families but eighteen decision variants; the paper
must not call these eighteen items "pattern families."
