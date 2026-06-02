# ReTrace-Bench — Figure Plan

Planned figures. For each: purpose, input data, and whether it can be produced
now. Figures must be blind-review safe (no author/repo identifiers in captions).

---

## Figure 1 — Scenario structure

- **Purpose:** show the anatomy of one scenario: `public_input.event_trace` (with
  trust levels / visibility scopes / timestamps) and `initial_memory`, plus the
  hidden gold kept separate from model inputs.
- **Input:** a single representative scenario from `sample_80_hard_en`.
- **Producible now:** yes (hand-drawn schematic from one scenario JSON).

## Figure 2 — Four task views over one scenario

- **Purpose:** illustrate how black-box answer, memory-state, evidence-retrieval,
  and diagnostic views derive from the same underlying state.
- **Input:** the same scenario as Figure 1 plus its gold fields (for the
  figure only, not exposed to models).
- **Producible now:** yes (schematic).

## Figure 3 — Evaluation pipeline

- **Purpose:** prediction JSONL -> normalize -> strict validation -> per-scenario
  scoring -> aggregate -> headline metrics; show the no-model / no-API-key
  evaluator path.
- **Input:** the API/CLI flow in `benchmark/retrace_bench/api.py` and
  `scripts/evaluate_retrace_bench_predictions.py`.
- **Producible now:** yes (flow diagram).

## Figure 4 — Failure-mode examples

- **Purpose:** show 2–4 contrasting failure modes (e.g. `stale_memory_reuse` vs.
  `over_update` vs. `scope_leakage`) with the correct decision and a typical
  wrong answer.
- **Input:** selected scenarios spanning failure modes from `sample_80_hard_en`.
- **Producible now:** yes (curated examples).

## Figure 5 — Baseline performance (bar / radar)

- **Purpose:** visualize the headline-metric gap between deployable baselines and
  the oracle consistency reference; emphasize collapse on non-answer decisions
  and evidence F1.
- **Input:** Table 4 numbers
  (`docs/retrace_bench/baseline_results_test_800_templateheldout_en.md`).
- **Producible now:** yes from existing offline numbers; refresh after adding a
  real LLM baseline. Keep the oracle visually separated/annotated as
  non-deployable.
