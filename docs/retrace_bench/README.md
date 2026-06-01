# ReTrace-Bench

**ReTrace-Bench** is an evaluation-only benchmark that evaluates memory reliability in agentic workflows.

## Conceptual Framing & Boundary

ReTrace-Bench is strictly **evaluation-only** and independent of the method paper framework (ReTrace-Learn).
- **ReTrace-Learn**: The trainable method (SFT/RL/DPO) that trains Graph Extractors and Typed Revision Proposers. Uses internal data under `data/retrace_learn/`.
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

## CLI Usage (v2)

### 1. Validate Data
```bash
PYTHONPATH=. python scripts/validate_retrace_bench_v2.py --data data/retrace_bench/sample_20_v2
```

### 2. Run Baselines
```bash
PYTHONPATH=. python scripts/run_retrace_bench_v2_baseline.py \
  --data data/retrace_bench/sample_20_v2 \
  --baseline latest_only_v2 \
  --out outputs/retrace_bench_v2/latest_only_sample20.jsonl
```

---

## Backward Compatibility (v1)

For users of ReTrace-Bench v1:
- The legacy schemas (`schemas.py`) and taxonomy (`taxonomy.py`) remain fully functional.
- The v1 scripts (`validate_retrace_bench.py` and `run_retrace_bench_baseline.py`) are preserved unchanged.
- v2 is designed additively, and includes a deterministic mapping layer to convert v1 history lists into v2 event traces.

