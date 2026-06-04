# ReTrace-Bench (internal "v1.1") — Offline Baseline Report

**Scope of this pass:** offline, **no paid/API model evaluations were run.** Every
number below comes from deterministic, local, rule-based baselines via
`scripts/run_retrace_bench_baseline.py`. These are **sanity checks**, not
model-quality results.

- API model results (e.g. Kimi / GLM / DeepSeek) must be added later from valid,
  committed prediction/metric artifacts, or recovered from a clean rerun.
- The previously **lost** hard500 API numbers (Kimi ≈ 0.054, GLM ≈ 0.140,
  DeepSeek ≈ 0.232 joint) are **internal lost-run notes only** and are **not**
  used as official table results anywhere in this repo or paper.

**Predictions + metrics:** `outputs/retrace_bench_v1_1/baselines/`
**Splits evaluated:** `main_3000_en` (3000), `hard_500_en` (500)

## Baselines run (all offline / API-free)

| Baseline | Group | Reads gold? | Intent |
|---|---|---|---|
| `latest_only` | recency | no | uses only the latest event/memory; tests recency bias |
| `retrieve_all` | over-retrieval | no | cites/uses everything; tests evidence gaming |
| `rag_lexical` | retrieval | no | keyword/lexical retrieval baseline |
| `crud_memory` | memory_baseline | no | naive create/read/update/delete memory |
| `mem0_style` | memory_baseline | no | approximate popular memory-framework heuristic |
| `heuristic_memory_state` | memory_baseline | no | rule-based memory-state tracker |
| `retrace_oracle_engine` | **oracle** | **yes** | upper-bound reference; **not deployable** |

`llm_json_answerer` exists but requires an API provider and was **deliberately
not run** in this pass.

## Results — `hard_500_en`

| Baseline | joint | decision_macro_f1 | memory_state_acc | evidence_f1 | min_evidence_em | failure_diag_acc | format_fail |
|---|---|---|---|---|---|---|---|
| latest_only | 0.000 | 0.133 | 0.447 | 0.000 | – | 0.172 | 0.0 |
| retrieve_all | 0.000 | 0.133 | 0.447 | 0.432 | – | 0.172 | 0.0 |
| rag_lexical | 0.000 | 0.133 | 0.447 | 0.131 | – | 0.172 | 0.0 |
| crud_memory | 0.000 | 0.133 | 0.447 | 0.391 | – | 0.172 | 0.0 |
| mem0_style | 0.000 | 0.133 | 0.447 | 0.391 | – | 0.172 | 0.0 |
| heuristic_memory_state | 0.000 | 0.133 | 0.781 | 0.391 | – | 0.120 | 0.0 |
| **retrace_oracle_engine** (oracle) | **0.570** | 1.000 | 0.847 | 1.000 | – | 1.000 | 0.0 |

## Results — `main_3000_en`

| Baseline | joint | decision_macro_f1 | memory_state_acc | evidence_f1 | min_evidence_em | failure_diag_acc | format_fail |
|---|---|---|---|---|---|---|---|
| latest_only | 0.000 | 0.282 | 0.578 | 0.000 | – | 0.067 | 0.0 |
| retrieve_all | 0.000 | 0.282 | 0.578 | 0.412 | – | 0.067 | 0.0 |
| rag_lexical | 0.000 | 0.282 | 0.578 | 0.260 | – | 0.067 | 0.0 |
| crud_memory | 0.000 | 0.282 | 0.578 | 0.592 | – | 0.067 | 0.0 |
| mem0_style | 0.000 | 0.282 | 0.578 | 0.592 | – | 0.067 | 0.0 |
| heuristic_memory_state | 0.000 | 0.282 | 0.828 | 0.592 | – | 0.133 | 0.0 |
| **retrace_oracle_engine** (oracle) | **0.767** | 1.000 | 0.903 | 1.000 | – | 1.000 | 0.0 |

## Reading of results

1. **All deployable (non-oracle) baselines score `joint_revision_success = 0.0`**
   on both splits. The joint metric requires decision + memory-state + minimal
   evidence + diagnosis to all be correct simultaneously; recency, over-retrieval,
   lexical retrieval, and simple CRUD/Mem0-style heuristics cannot clear this bar.
   This is the intended discrimination property of ReTrace-Bench.
2. **`latest_only` gets `evidence_f1 = 0.0`** — the latest-event shortcut never
   recovers the minimal supporting evidence set, confirming there is no
   latest-event shortcut to the answer.
3. **`retrieve_all` does not win on evidence** (hard `evidence_f1 = 0.432`):
   over-citing everything is penalized by the minimal-evidence requirement, so
   evidence cannot be gamed by dumping all events.
4. **`retrace_oracle_engine` is a partial upper bound, not 1.0.** It reads the
   oracle proposal but still commits through the deterministic engine, so it is
   distinct from the *gold-oracle self-consistency* check (which **is** 1.0; see
   `v1_1_validation_report.md` / `outputs/retrace_bench_v1_1/gold_oracle/`). Its
   sub-1.0 joint score reflects that the engine, not the gold, produces the final
   committed state — exactly what a deployable upper bound should show.
5. **`format_failure_rate = 0.0` everywhere** — all baselines emit
   scorer-parseable predictions.

## Not implemented / not run in this cleanup pass

- Any API/LLM model evaluation (`llm_json_answerer` and all paid providers).
- No new model predictions were generated. Headline model tables remain **TODO**
  pending valid artifacts.

## Reproduction

```bash
for split in main_3000 hard_500; do
  for b in latest_only retrieve_all rag_lexical crud_memory mem0_style heuristic_memory_state retrace_oracle_engine; do
    python scripts/run_retrace_bench_baseline.py \
      --data data/retrace_bench_v1_1/${split}_en/scenarios.jsonl \
      --baseline $b \
      --out outputs/retrace_bench_v1_1/baselines/${split}__${b}.jsonl
  done
done
```
