# Baseline results — `sample_80_hard_en`

> **Legacy (pre-v1.0) document.** Describes a legacy pre-v1.0 split/pilot, recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`. Retained for provenance only; it does **not** describe the ReTrace-Bench v1.0 splits (`main`/`hard`/`realistic`/`calibration`).


Current baseline results on the headline hard split (80 scenarios, all 8 domains
and all 11 failure modes). Deterministic baselines are reproducible offline; the
API baseline (`llm_json_answerer`) is provider-dependent and not reported here.

## How to reproduce

```bash
PYTHONPATH=. python scripts/run_retrace_bench_baseline.py \
  --data data/retrace_bench/sample_80_hard_en \
  --baseline <baseline> \
  --out outputs/retrace_bench/<baseline>.jsonl
```

`<baseline>` ∈ {`latest_only`, `retrieve_all`, `rag_lexical`, `crud_memory`,
`mem0_style`, `retrace_oracle_engine`}. Metrics are written next to `--out` as
`<baseline>.metrics.json`. (`outputs/` is gitignored and must not be committed.)

## Headline metrics

Higher is better for every column **except `stale_reuse`** (lower is better).

| baseline | decision_acc | decision_macro_f1 | non_answer_acc | mem_state_acc | evidence_f1 | diagnosis_acc | stale_reuse↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| latest_only | 0.650 | 0.158 | 0.000 | 0.583 | 0.000 | 0.100 | 0.000 |
| retrieve_all | 0.650 | 0.158 | 0.000 | 0.583 | 0.273 | 0.100 | 1.000 |
| rag_lexical | 0.650 | 0.158 | 0.000 | 0.598 | 0.400 | 0.100 | 0.900 |
| crud_memory | 0.738 | 0.495 | 0.500 | 0.722 | 0.675 | 0.450 | 0.000 |
| mem0_style | 0.650 | 0.286 | 0.250 | 0.682 | 0.675 | 0.275 | 0.000 |
| **retrace_oracle_engine** *(oracle, upper bound)* | **1.000** | **1.000** | **1.000** | **0.886** | **1.000** | **1.000** | **0.000** |

## Reading the results

- **The benchmark is hard and discriminative.** Deployable baselines top out at
  `crud_memory` (0.738 decision accuracy, 0.495 macro-F1, 0.722 memory-state),
  far below the oracle. The headline `decision_accuracy` is misleading on its own
  because `use_current_memory` is the majority class (52/80); **`decision_macro_f1`**
  and **`non_answer_decision_accuracy`** expose how weak the non-trivial decisions
  are (e.g. `latest_only`/`rag_lexical` score 0.0 on non-answer cases).
- **Retrieval alone reuses stale memory.** `retrieve_all` and `rag_lexical` have
  high `stale_reuse` (1.00 / 0.90): with no mutation/temporal semantics they
  surface the outdated-but-plausible answer. `crud_memory`/`mem0_style` avoid this
  by applying last-write-wins, but still miss conflict/scope/policy reasoning.
- **Diagnosis is the hardest view.** Even `crud_memory` reaches only 0.45
  `failure_diagnosis_accuracy`; the sanity/retrieval baselines are at 0.10
  (≈ chance over 11 modes).
- **The oracle is a true upper bound, not perfect gold replay.** It scores 1.0 on
  decision/evidence/diagnosis but only **0.886** memory-state, because the engine's
  deterministic status mapping does not reproduce every gold `restored`/`deleted`
  label (see per-mode table). This is expected: the oracle bounds what
  engine-routable typed revisions can achieve, and is reported for context only —
  **never as a deployable competitor**.

## Per-failure-mode (memory_state_accuracy)

Where the oracle upper bound itself is below 1.0, indicating gold states that the
deterministic engine mapping does not fully reconstruct:

| failure mode | n | oracle mem_state | crud_memory mem_state |
| --- | --- | --- | --- |
| failure_to_release_or_restore | 7 | 0.50 | 0.50 |
| failure_to_forget | 7 | 0.71 | 0.71 |
| over_update | 8 | 0.93 | 0.75 |
| stale_memory_reuse | 8 | 0.93 | 0.71 |
| conflict_collapse | 7 | 0.94 | 0.97 |
| policy_violation | 7 | 1.00 | 0.69 |

`restore`/`forget` are the weakest modes for every method, making them the most
informative targets for future systems.

## Provenance

Numbers regenerated from the committed `sample_80_hard_en/scenarios.jsonl` using
the deterministic baselines above. The dataset passes
`scripts/validate_retrace_bench_dataset.py` (0 errors) and a gold-equal
prediction scores perfectly on every scenario
(`tests/retrace_bench/test_sample_80_hard_validation.py`).
