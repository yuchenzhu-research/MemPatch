# ReTrace-Bench — Table Plan

Planned paper tables. For each: source file/command, expected columns, whether
the data already exists, and missing items. Numbers below are illustrative of the
current offline run on the canonical split; regenerate before camera-ready.

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

## Table 3 — Split summary

- **Columns:** split | count | role | train/tune allowed? | in public release?
- **Rows:**
  - `test_800_templateheldout_en` | 800 | canonical held-out test | no | yes
  - `sample_80_hard_en` | 80 | calibration/quickstart (HF `validation`) | no | yes
  - `train_3000_en` | 3000 | supervision pool | yes | yes
  - `dev_400_en` | 400 | selection pool | yes | yes
  - `test_800_en` | 800 | prototype/diagnostic | no | no (excluded)
- **Source:** `data/retrace_bench/*`, `data/retrace_supervision/*`,
  `release/huggingface/ReTrace-Bench/README.md`.
- **Exists:** yes. **Missing:** confirm counts at camera-ready via
  `validate_retrace_bench_dataset.py`.

## Table 4 — Main baseline results (canonical test split)

- **Columns:** group | baseline | oracle? | decision macro-F1 | non-answer acc. |
  memory state | evidence F1 | diagnosis | stale reuse | (aux) decision acc. |
  key fact.
- **Rows:** `latest_only`, `retrieve_all` (sanity); `rag_lexical`, `crud_memory`,
  `mem0_style`, `heuristic_memory_state` (memory); `retrace_oracle_engine`
  (oracle, separated).
- **Source / command:**
  ```bash
  PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
    --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
    --out-dir outputs/retrace_bench/ablation_test_800_templateheldout_offline \
    --max-cases 800
  ```
  Pre-rendered in
  `docs/retrace_bench/baseline_results_test_800_templateheldout_en.md`.
- **Exists:** yes (offline baselines + oracle). **Missing:** at least one real
  LLM baseline (`llm_json_answerer`, needs a provider/API key); optional real
  Mem0/Graphiti-style baseline.

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
- **Rows:** e.g. evidence-only vs. full; difficulty L1→L4 trend; template-lookup
  shortcut probe (leakage upper bound, clearly labeled non-deployable).
- **Source:** `scripts/run_retrace_bench_ablation.py`,
  `template_lookup_test_800_templateheldout_en.md`.
- **Exists:** partially (template-lookup probe exists). **Missing:** consolidated
  ablation table.

## Appendix tables

- **A1 Validation gates:** gate | threshold | observed rate. Source:
  `validate_retrace_bench_dataset.py` output (e.g. non_answer 0.709,
  verified_over_trusted 0.666, events_ge_7 0.84, distractors 1.0, cross_scope
  1.0). Exists: yes.
- **A2 Leakage checks:** template-signature overlap; template-lookup probe
  scores. Source: `template_signature_report.md`,
  `template_lookup_test_800_templateheldout_en.md`,
  `split_leakage_report.md`. Exists: yes.
- **A3 Oracle consistency:** oracle headline metrics + the memory-state 0.968
  ceiling explanation. Source: baseline-results doc. Exists: yes.
