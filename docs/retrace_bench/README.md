# ReTrace-Bench

**ReTrace-Bench** is a general, evaluation-only benchmark for **agent memory
revision reliability** in agentic workflows. It is method-neutral: it does not
depend on ReTrace-Learn training, and any memory-enabled agent (LLM-only, RAG,
CRUD store, Mem0-style, or a trained policy) can be scored on it.

## Benchmark paper assets

- **Benchmark-paper framing & gap analysis:** [`benchmark_paper.md`](benchmark_paper.md)
  — why existing long-context / long-term-dialogue / stale-memory / general-agent
  benchmarks do not directly measure shared memory revision reliability, and how
  ReTrace-Bench fills that gap.
- **Dataset design / card:** [`dataset_design.md`](dataset_design.md),
  [`dataset_card_hf.md`](dataset_card_hf.md).
- **Historical pilot baselines (legacy pre-v1.0 splits, kept for provenance):**
  [`baseline_results_sample_80_hard_en.md`](baseline_results_sample_80_hard_en.md).

ReTrace-Bench v1.0 ships four paper-facing splits under public names `main` /
`hard` / `realistic` / `calibration` (`data/retrace_bench/main_3000_en`,
`hard_300_en`, `realistic_100_en`, `calibration_80_en`). The legacy pre-v1.0
layout is recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`.

## Conceptual Framing & Boundary

ReTrace-Bench is strictly **evaluation-only** and independent of the method paper framework (ReTrace-Learn).
- **ReTrace-Learn**: The trainable method (SFT/RSFT/DPO) that trains a Graph Builder and Proposal Policy. Uses internal data under `data/retrace_learn/`.
- **ReTrace-Bench**: The independent benchmark paper evaluating different agent memory updating strategies. The evaluation data under `data/retrace_bench/` must **never** be used for training or validation tuning.
- **DPA Diagnostic Track**: Structured revision / Defeat-Path Authorization remains an optional diagnostic protocol. It is **not** mandatory for all benchmark participants. The main track is the model-agnostic **Black-box Task Protocol**.

---

## Benchmark Design Target (v2)

The benchmark is structured across:
- **8 industrial domains**:
  - `software_engineering_agent`
  - `enterprise_multi_tool_workflow`
  - `customer_support_crm`
  - `calendar_task_workflow`
  - `research_knowledge_work`
  - `personal_assistant_preference`
  - `ecommerce_recommendation`
  - `data_analysis_bi`
- **11 memory reliability failure modes**:
  - `stale_memory_reuse`, `under_update`, `over_update`, `conflict_collapse`, `scope_leakage`, `policy_violation`, `wrong_source_attribution`, `memory_hallucination`, `unnecessary_memory_write`, `failure_to_forget`, `failure_to_release_or_restore`.
- **4 evaluation protocols**:
  - `black_box_task`, `memory_state_task`, `structured_revision_task`, `oracle_diagnostic_task`.

---

## CLI Usage

### 1. Validate Data
```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl
```

### 2. Run Baselines
```bash
PYTHONPATH=. python scripts/run_retrace_bench_baseline.py \
  --data data/retrace_bench/calibration_80_en/scenarios.jsonl \
  --baseline latest_only \
  --out outputs/retrace_bench/latest_only_calibration.jsonl
```

---

## Backward Compatibility (v1)

For users of ReTrace-Bench v1:
- The legacy schemas (`schemas.py`) and taxonomy (`taxonomy.py`) remain fully functional.
- The v1 scripts (`validate_retrace_bench.py` and `run_retrace_bench_baseline.py`) are preserved unchanged.
- v2 is designed additively, and includes a deterministic mapping layer to convert v1 history lists into v2 event traces.
