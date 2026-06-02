# Offline Baseline Results: test_800_templateheldout_en

`data/retrace_bench/test_800_templateheldout_en/` is the canonical paper-facing held-out benchmark split. `data/retrace_bench/test_800_en/` is retained as prototype/diagnostic only and must not be used for paper headline numbers.

`data/retrace_supervision/train_3000_en/` and `data/retrace_supervision/dev_400_en/` are synthetic supervision/selection pools, not benchmark test sets. The template lookup diagnostic is a shortcut-leakage probe, not a deployable memory baseline.

Command:

```bash
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl --out-dir outputs/retrace_bench/ablation_test_800_templateheldout_offline --max-cases 800
```

## Results

Rows are grouped so the gold-replay consistency reference is never mistaken for a deployable method:

- **sanity:** `latest_only`, `retrieve_all`
- **retrieval / memory baselines:** `rag_lexical`, `crud_memory`, `mem0_style`, `heuristic_memory_state`
- **oracle (gold-replay consistency reference, not a deployable method):** `retrace_oracle_engine`

| group | baseline | oracle? | decision acc. | decision macro-F1 | non-answer acc. | key fact | memory state | evidence F1 | diagnosis | stale reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sanity | latest_only | false | 0.291 | 0.090 | 0.000 | 0.107 | 0.323 | 0.107 | 0.091 | 0.000 |
| sanity | retrieve_all | false | 0.291 | 0.090 | 0.000 | 0.000 | 0.316 | 0.303 | 0.091 | 1.000 |
| memory_baseline | rag_lexical | false | 0.291 | 0.090 | 0.000 | 0.007 | 0.316 | 0.305 | 0.091 | 0.993 |
| memory_baseline | crud_memory | false | 0.496 | 0.357 | 0.321 | 0.694 | 0.502 | 0.694 | 0.229 | 0.000 |
| memory_baseline | mem0_style | false | 0.292 | 0.140 | 0.034 | 0.694 | 0.393 | 0.694 | 0.203 | 0.000 |
| memory_baseline | heuristic_memory_state | false | 0.291 | 0.090 | 0.000 | 0.694 | 0.603 | 0.694 | 0.091 | 0.000 |
| oracle | retrace_oracle_engine | true | 1.000 | 1.000 | 1.000 | 1.000 | 0.968 | 1.000 | 1.000 | 0.000 |

`retrace_oracle_engine` is a **gold-replay consistency reference / oracle consistency diagnostic**, not a competing deployable method. It is allowed to read `hidden_gold` (answer, decision, evidence, diagnosis) and replays the gold typed revision through the deterministic ReTrace-Engine. Its role is to (a) confirm the benchmark is internally consistent and solvable from the gold labels and (b) bound the achievable state/evidence/diagnosis scores. It must never be reported alongside the deployable baselines as if it were a competing system.

### Why memory_state is 0.968 and not 1.000 for the oracle

`memory_state_accuracy` for the oracle is `0.968`, not `1.000`, and this is unchanged by the decision fix in this update (it was `0.968` before as well). The reason is a property of the oracle proposer's typed-edge construction, which runs through the deterministic engine:

- The oracle proposer only emits `SUPERSEDES` edges (for targets whose gold state is `outdated`) and `UNCERTAIN` edges (for targets whose gold state is `unresolved`).
- It does **not** emit `BLOCKS`/`RELEASES` edges, deletion, or restore actions, and it only acts on the target belief, not on condition-type memories.
- Consequently, on the 90/800 scenarios whose gold memory state includes `deleted` (failure-to-forget), `restored`/`blocked` (failure-to-release/restore), or `unresolved` on a *condition* memory, those memories remain `current` after deterministic authorization, lowering the average.

This is a known, scoped limitation of the minimal oracle proposer's edge vocabulary — not a decision-mapping bug — so `memory_state_accuracy` for the oracle is an upper-bound diagnostic for the `SUPERSEDES`/`UNCERTAIN` paths the oracle exercises, rather than a global ceiling for all eight memory states. The decision, evidence, diagnosis, and non-answer paths are gold-replayed exactly and reach `1.000`.

## Change log

This document was regenerated after fixing `retrace_oracle_engine` to replay `hidden_gold.expected_decision` instead of reconstructing the decision from a hard-coded `failure_mode -> decision` mapping. Before the fix the oracle reported decision acc `0.294` / decision macro-F1 `0.232` / non-answer acc `0.132`, which was below `crud_memory` and unacceptable for a gold-replay reference. After the fix the oracle reaches decision acc `1.000` / macro-F1 `1.000` / non-answer acc `1.000`.
