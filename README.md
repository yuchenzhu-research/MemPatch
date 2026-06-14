# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI is the ability of an LLM agent to revise which memories remain usable (`current`, `blocked`, `unresolved`, `out_of_scope`, or `should_not_store`) when new evidence arrives, rather than blindly appending text.

| Component | Location |
|-----------|----------|
| **MemPatch-Bench** (dataset view + scoring) | `benchmark/` |
| **Scenario generator** (v1.3) | `benchmark/generation/` |
| **Revision module** (Path A) | `mempatch/revision/` (`import mempatch.revision`) |
| **DPA kernel** (deterministic verifier) | `mempatch/dpa/` (`import mempatch.dpa`) |
| **Regression tests** | `mempatch/tests/` |
| **Reproducibility CLIs** | `scripts/` |

The scenario JSONL bundle is **not** checked into this repository. Regenerate it locally (see [Dataset](#dataset-v13)) or load from a public dataset host after release.

## Architecture

```text
benchmark/                    dataset format, public view, scoring ONLY
  api.py                      evaluate_predictions() — the single public scorer
  public_view.py              model-visible scenario fields (no hidden_gold)
  model_runner.py             optional JSON prompt adapter for baselines (NOT revision)

mempatch/revision/           Path A method (view → proposer → DPA → projection)
mempatch/dpa/                DPA authorization kernel (authorize)
scripts/                      thin CLIs that wire runners to the evaluator
```

**No overlap:** `benchmark/` scores any compliant `predictions.jsonl`. `mempatch/` implements the typed Path A revision stack. Linux evaluates full Path A, a paired no-DPA action projection, and direct-response Path B. External baselines use `benchmark.model_runner.build_prompt` as a JSON adapter, not the revision stack.

## Data flow

```text
{split}/scenarios.jsonl          # local or downloaded bundle
        │
        ▼
scripts/linux/run_hf_path_a_eval.py / run_hf_test_eval.py
        │
        ▼
predictions.jsonl              { "scenario_id", "response": { five fields } }
        │
        ▼
benchmark.api.evaluate_predictions()
        │
        ▼
metrics JSON / headline table
```

Paper metric hierarchy:

| Tier | Metrics | Interpretation |
|------|---------|----------------|
| Primary effects | `decision_macro_f1`, `memory_state_accuracy` | Balanced decision quality and memory revision correctness |
| Primary validity gate | `response_schema_compliance_rate` | Operational validity; report both raw and projected rates |
| Secondary | `evidence_f1`, `failure_diagnosis_accuracy` | Evidence selection and diagnostic quality |
| Confirmatory strict composite | `joint_revision_success` | Exact all-or-nothing success; expected to be sparse on L4 |
| Safety / diagnostics | `stale_reuse_rate`, answer and per-mode metrics | Failure analysis rather than the main efficacy claim |

The strict joint metric is not relaxed after evaluation. Sparse or zero values
are reported alongside component metrics instead of changing the scorer.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional LLM baselines: `pip install -e ".[dev,llm]"` (export provider API keys yourself).

## Dataset (v1.3)

| Split | Rows | Use |
|-------|-----:|-----|
| `train` | 3500 scenarios | L3 fixed 80/20 split; each train/val scenario yields Path A and Path B SFT rows |
| `test` | 500 | Held-out L4 final eval (never used for training or checkpoint selection) |

v1.3 uses 7 primary failure modes, 8 pattern families, and 6 domains. **Train is L3; test is L4** — test measures generalization to a harder distribution, not memorization of training cases.

The fixed partition is stratified by decision. SFT preparation additionally fails if either the SFT-train or held-out val partition omits any required decision, failure-diagnosis, or memory-status label.

Regenerate locally (default output directory is gitignored):

```bash
python scripts/data/generate_mempatch.py --full --out-dir local/data/mempatch
python scripts/data/package_mempatch_release.py \
  --input-dir local/data/mempatch --out-dir local/data/mempatch --validate
```

Audit gate (must pass before training):

```bash
python scripts/workflows/audit_decision_boundary.py \
  --data local/data/mempatch/train \
  --data local/data/mempatch/test
```

## Evaluate predictions

```bash
python scripts/workflows/evaluate_mempatch_predictions.py \
  --data local/data/mempatch/test/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.api import load_scenarios, load_predictions, evaluate_predictions

result = evaluate_predictions(
    load_scenarios("local/data/mempatch/test/scenarios.jsonl"),
    load_predictions("predictions.jsonl"),
    strict=True,
)
print(result["headline_metrics"])
```

Smoke (no dataset download; uses committed fixtures):

```bash
python scripts/workflows/evaluate_mempatch_predictions.py \
  --data mempatch/tests/fixtures/smoke_scenarios.jsonl \
  --predictions mempatch/tests/fixtures/smoke_predictions.jsonl
```

## Paper reproduction (Linux CUDA)

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
export RUN_ID=full1024
bash scripts/linux/run_paper_campaign.sh
```

Place dataset at `$LOCAL_ROOT/data/mempatch/{train,test}/scenarios.jsonl`. The
campaign runs Qwen3, Gemma-3, Phi-4, then Mistral-Nemo for 1,024 steps, retains
checkpoints every 128 steps, and selects only by the fixed L3 validation loss
before test500 evaluation. See `scripts/linux/README.md`.

Protocol per backbone: one 1,024-step multitask QLoRA run over `FINAL_STATE`
and `PATCH_ACTION`, checkpoints every
128 steps, selection by the lowest mixed-task L3 val loss, then held-out L4
test500 evaluation. Each scenario in the fixed 80/20 train partition yields
one Path B response target and one Path A typed-action target; the original
scenario JSONL is unchanged. This is one fixed partition, not k-fold
cross-validation.

## Local workspace (`local/`, gitignored)

```text
local/
  data/mempatch/    generated scenario JSONL (train + test)
  models/           prefetched HF weights (Linux) or MLX weights (dev)
  adapters/         LoRA checkpoints
  runs/             eval outputs
  logs/             training logs
```

## Repository layout

```text
benchmark/              MemPatch-Bench scorer + generator
mempatch/
  revision/             Path A revision module
  dpa/                  Defeat-Path Authorization kernel
  tests/                pytest + smoke fixtures
scripts/                reproducibility CLIs (eval, workflows, data)
data/mempatch/          tracked boundary-audit artifact (v1.3)
```

Script index: `scripts/README.md`.

## Reproduction models

| Slug | Model | Params |
|------|-------|-------:|
| `qwen3_14b` | Qwen3-14B | 14B |
| `gemma3_12b` | Gemma 3 12B Instruct | 12B |
| `phi4` | Phi-4 | 14B |
| `mistral_nemo_12b` | Mistral Nemo 12B Instruct | 12B |

Paper reproduction uses 4-bit NF4 QLoRA on Linux CUDA with temperature 0 evaluation. MLX remains a development backend and is not the final paper protocol.

## Naming (paper ↔ code)

| Paper term | Code import / path |
|------------|-------------------|
| **MemPatch** | Project name; PyPI package `mempatch` |
| **MemPatch-Bench** | `benchmark/` — dataset view + `evaluate_predictions` |
| **MemPatch revision module** | `mempatch.revision` — Path A view → proposer → DPA → projection |
| **DPA** (Defeat-Path Authorization) | `mempatch.dpa.authorize` |
| **Path A (full MemPatch)** | Revision view → typed action proposer → DPA → benchmark projection |
| **Path A no-DPA** | Same generated typed actions → direct action/status projection without gate or DPA |
| **Path B (ablation)** | Direct five-field response JSON without the typed revision/DPA stack |

The Linux runner writes full Path A predictions and audit traces, a paired no-DPA result from the exact same typed actions, and Path B direct-response results. The current v1.3 training scenarios supervise `BLOCKS`, `UNCERTAIN`, `REAFFIRMS`, and `NO_REVISION`; they do not contain gold `SUPERSEDES` or `RELEASES` transitions, which remains an explicit action-coverage limitation.

## Paper result policy

Final tables must use the unified 1,024-step protocol on all four campaign
backbones and the complete held-out test500 split. Report frozen external
baselines separately from the Final-State Control versus MemPatch comparison, and
report raw schema compliance, projected schema compliance, and repair counts
separately.

## Citation

If you use MemPatch-Bench or this code, please cite the accompanying paper (bibtex to be added upon publication).
