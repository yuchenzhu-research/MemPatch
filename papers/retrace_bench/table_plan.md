# ReTrace-Bench — Table Plan

Planned paper tables. For each: source file/command, expected columns, whether
the data already exists, and missing items. ReTrace-Bench v1.0 uses four
paper-facing splits (`main` / `hard` / `realistic` / `calibration`); headline
baselines require a full model-suite rerun on the v1.0 splits and are not yet
committed. Any numbers below are illustrative and must be regenerated before
final submission.

---

## Table 1 — Benchmark task views

Per-view summary of what the model outputs and how it is scored.

- **Columns:** task view | model output | gold field | headline metric.
- **Rows:** black-box answer; memory-state; evidence-retrieval; diagnostic.
- **Source:** `general_taxonomy.py` (`TASK_TYPES`),
  `scorers_general.py` (`HEADLINE_METRICS`).
- **Exists:** yes (static). **Missing:** none.

## Table 2 — Taxonomy

Full label spaces.

- **Columns:** axis | labels | count.
- **Rows:** domains (8); failure modes (11); difficulties (L1–L4); memory
  statuses (8); decisions (5); trust levels (3).
- **Source:** `benchmark/retrace_bench/general_taxonomy.py`.
- **Exists:** yes (static). **Missing:** none.

## Table 3 — Split summary (ReTrace-Bench v1.0)

- **Columns:** split (public name) | count | role | model selection allowed? | in public release?
- **Rows:**
  - `main` (`main_3000_en`) | 3000 | controlled benchmark main split | no | yes
  - `hard` (`hard_300_en`) | 300 | long-context / multi-evidence stress | no | yes
  - `realistic` (`realistic_100_en`) | 100 | realistic-style, annotation pending | no | yes
  - `calibration` (`calibration_80_en`) | 80 | smoke / quickstart only | no | yes
- **Source:** `data/retrace_bench/*`,
  `release/huggingface/ReTrace-Bench/README.md`. ReTrace-Learn method-track data
  (`data/retrace_learn/`) reuses ReTrace-Bench-derived scenarios with declared
  split roles.
- **Exists:** yes. **Missing:** confirm counts before final submission via
  `validate_retrace_bench_dataset.py`.

## Table 4 — Main baseline results (`main` split)

- **Columns:** group | baseline | oracle? | decision macro-F1 | non-answer acc. |
  memory state | evidence F1 | diagnosis | stale reuse | (aux) decision acc. |
  key fact.
- **Rows:** `latest_only`, `retrieve_all` (sanity); `rag_lexical`, `crud_memory`,
  `mem0_style`, `heuristic_memory_state` (memory); `retrace_oracle_engine`
  (oracle, separated).
- **Source / command:**
  ```bash
  PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
    --data data/retrace_bench/main_3000_en/scenarios.jsonl \
    --out-dir outputs/retrace_bench/ablation_main_3000_offline
  ```
- **Exists:** no — v1.0 baselines must be regenerated on `main` (and `hard`).
  **Missing:** the full offline baseline + oracle run on the v1.0 splits, plus
  at least one real LLM baseline (`llm_json_answerer`, needs a provider/API key).

## Table 5 — Per-failure-mode breakdown

- **Columns:** failure mode | n | decision macro-F1 | memory state | evidence F1
  | diagnosis acc.
- **Source:** `aggregate_metrics(...)["by_failure_mode"]` (see
  `scorers_general.py`); emit from the ablation/baseline runner output.
- **Exists:** computable now from the offline run. **Missing:** rendered table in
  the baseline-results doc (currently only the overall table is rendered).

## Table 6 — Per-domain breakdown

- **Columns:** domain | n | decision macro-F1 | memory state | evidence F1 |
  diagnosis acc.
- **Source:** `aggregate_metrics(...)["by_domain"]`.
- **Exists:** computable now. **Missing:** rendered table.

## Table 7 — Ablations / diagnostics

- **Columns:** variant | headline metrics.
- **Rows:** e.g. evidence-only vs. full; difficulty L1→L4 trend; decision-word
  leakage probe (upper bound, clearly labeled non-deployable).
- **Source:** `scripts/run_retrace_bench_ablation.py`, the v1.0 leakage audit in
  `benchmark/retrace_bench/generation/release_manifest.py`.
- **Exists:** partially. **Missing:** consolidated ablation table on v1.0 splits.

## Appendix tables

- **A1 Validation gates:** gate | threshold | observed rate. Source:
  `validate_retrace_bench_dataset.py` output (e.g. non_answer 0.709,
  verified_over_trusted 0.666, events_ge_7 0.84, distractors 1.0, cross_scope
  1.0). Exists: yes.
- **A2 Leakage checks:** v1.0 decision-word leakage audit (no authoritative
  record contains a decision phrase); cross-split ID/text disjointness. Source:
  `benchmark/retrace_bench/generation/release_manifest.py`,
  per-split `manifest.json` (`leakage_audit_summary`). Legacy
  template-signature reports retained for provenance only. Exists: yes.
- **A3 Oracle consistency:** oracle headline metrics + the memory-state 0.968
  ceiling explanation. Source: baseline-results doc. Exists: yes.
