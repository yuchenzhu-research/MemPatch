# Offline baseline results — `test_800_en` (held-out)

Held-out internal evaluation split: `data/retrace_bench/test_800_en/scenarios.jsonl`
(800 scenarios, all 8 domains + 11 failure modes).

Command (offline, no API):

```bash
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
  --data data/retrace_bench/test_800_en/scenarios.jsonl \
  --out-dir outputs/retrace_bench/ablation_test_800_offline \
  --max-cases 800
```

## Held-out policy

`test_800_en` is the internal ReTrace-Bench held-out evaluation set. It is **not**
used for any training, prompt tuning, policy optimization, or checkpoint
selection. Method development uses the disjoint `train_3000_en` (supervision
pool) and `dev_400_en` (selection) splits only — these share no scenario, case,
entity, memory ID, event ID, exact text, hidden gold, or seed range with the
test split (see `docs/retrace_bench/split_leakage_report.md`).

## Baseline table

Higher is better for all columns except `stale_reuse` (lower is better).
The oracle row is an **upper bound**, not a deployable / comparable baseline.

| group | baseline | oracle | decision_acc | decision_macro_f1 | non_answer_acc | key_fact | memory_state | evidence_f1 | diagnosis | stale_reuse↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| sanity | latest_only | no | 0.635 | 0.155 | 0.000 | 0.000 | 0.581 | 0.000 | 0.091 | 0.000 |
| sanity | retrieve_all | no | 0.635 | 0.155 | 0.000 | 0.000 | 0.581 | 0.272 | 0.091 | 1.000 |
| memory_baseline | crud_memory | no | 0.728 | 0.494 | 0.500 | 0.681 | 0.720 | 0.681 | 0.454 | 0.000 |
| memory_baseline | heuristic_memory_state | no | 0.635 | 0.155 | 0.000 | 0.681 | 0.670 | 0.681 | 0.090 | 0.000 |
| memory_baseline | mem0_style | no | 0.636 | 0.284 | 0.250 | 0.681 | 0.680 | 0.681 | 0.273 | 0.000 |
| memory_baseline | rag_lexical | no | 0.635 | 0.155 | 0.000 | 0.206 | 0.595 | 0.401 | 0.091 | 0.909 |
| **oracle (upper bound)** | retrace_oracle_engine | **yes** | 1.000 | 1.000 | 1.000 | 1.000 | 0.881 | 1.000 | 1.000 | 0.000 |

## Reading the numbers

- **Retrieval-only sanity baselines** (`latest_only`, `retrieve_all`, `rag_lexical`)
  collapse on decision macro-F1 (0.155), never produce non-answer actions, and
  reuse stale memory at 0.909–1.000 — they cannot revise beliefs under evolving
  evidence, scope, trust, or policy.
- **`crud_memory`** is the strongest deployable baseline here (decision 0.728,
  macro-F1 0.494, non-answer 0.500, memory-state 0.720, evidence-F1 0.681,
  diagnosis 0.454) but still far below the oracle, leaving large headroom.
- **`retrace_oracle_engine`** is the oracle upper bound (perfect on decision /
  evidence / diagnosis; 0.881 memory-state — restore/forget statuses remain the
  hardest). It is **not** a comparable deployable method; it consumes the
  structured authorization view and is shown only to bound achievable scores.
- The spread between deployable baselines (~0.64–0.73 decision) and the oracle
  (1.0) confirms the held-out split is discriminative and not solvable by naive
  recency or retrieval.

## Reproducibility / hygiene

- `outputs/` is **not** committed (gitignored). Re-running the command above
  regenerates the metrics deterministically.
- No API baselines were run (no API key requested today). The offline matrix is
  fully deterministic. If a key is later provided, an optional DeepSeek run
  should be limited to a small sample of test cases, not the full 800.
