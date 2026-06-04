# Hard150 Balanced — Validation & SiliconFlow Eval Report

**Dataset:** `data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl`  
**Outputs:** `outputs/retrace_bench_siliconflow_hard150_balanced/`  
**Date:** 2026-06-04

## Blockers addressed

### Blocker 1 — Public leakage sanitization

Added `benchmark/retrace_bench/public_view.py` with `sanitize_public_input()` and `public_scenario_view()`.

- Deep-copies `public_input` and strips internal fields: `is_distractor`, `hidden_gold`, `validation_notes`, `metadata`, `source_pointers`, `primary_failure_mode`, `pattern_trap_type`, `canonical_failure_mode`.
- Events retain: `event_id`, `timestamp_order`, `actor_role`, `trust_level`, `visibility_scope`, `event_type`, `text`, `related_memory_ids`, `timestamp`.
- Memory retains: `memory_id`, `text`, `scope`, `source_event_ids`.

Runners updated: `run_retrace_bench_siliconflow.py`, `run_retrace_bench_gemini_hard150.py`, `run_retrace_bench_api_models.py`, `run_retrace_bench_baseline.py` (`llm_json_answerer`).

### Blocker 2 — Expected decision distribution

Hard-split scheduler in `pattern_spec.py` (`build_hard_pattern_decision_plan`) assigns pattern/decision pairs deterministically from seed 2027.

## Expected decision distribution (actual vs targets)

| Decision | Target share | Actual count | Actual share |
| --- | --- | --- | --- |
| use_current_memory | 45–55% | 75 | **50.0%** |
| mark_unresolved | 15–25% | 30 | **20.0%** |
| ask_clarification | 10–15% | 18 | **12.0%** |
| refuse_due_to_policy | 8–12% | 15 | **10.0%** |
| escalate | 5–10% | 12 | **8.0%** |

**Pre-fix hard150 (skewed):** use_current_memory=115 (76.7%), mark_unresolved=25, refuse=10, ask=0, escalate=0.

## Gate results

| Gate | Result |
| --- | --- |
| Validator (`validate_retrace_bench_dataset.py`) | **PASS** (0 errors) |
| Gold oracle (`check_retrace_bench_gold_oracle.py`) | **PASS** (joint=1.0, all gold checks) |
| `format_failure_rate == 0` (all 3 models) | **PASS** |
| `joint_revision_success < 0.5` (all models) | **PASS** (max 0.227) |
| Decision distribution in target bands | **PASS** |

## Main metrics — hard150_balanced (sanitized prompts)

| model | joint_revision_success | memory_state_accuracy | non_answer_decision_accuracy | black_box_decision_accuracy | decision_macro_f1 | failure_diagnosis_accuracy | evidence_f1 | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro | 0.227 | 0.787 | 0.027 | 0.507 | 0.183 | 0.147 | 0.824 | 0.000 |
| GLM-5.1 | 0.193 | 0.791 | 0.373 | 0.533 | 0.342 | 0.233 | 0.837 | 0.000 |
| Kimi-K2.6 | 0.133 | 0.787 | 0.000 | 0.500 | 0.133 | 0.180 | 0.694 | 0.000 |

## Comparison vs pre-sanitize hard150 (`outputs/retrace_bench_siliconflow_hard150/`)

Same three SiliconFlow models; pre-sanitize run used unsanitized `public_input` (including `is_distractor`).

| model | memory_state_accuracy (old → new) | Δ | non_answer_decision_accuracy (old → new) | joint_revision_success (old → new) |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro | 0.851 → 0.787 | **−6.4 pp** | 0.000 → 0.027 | 0.273 → 0.227 |
| GLM-5.1 | 0.849 → 0.791 | **−5.8 pp** | 0.600 → 0.373 | 0.233 → 0.193 |
| Kimi-K2.6 | 0.853 → 0.787 | **−6.6 pp** | 0.000 → 0.000 | 0.133 → 0.133 |

### Interpretation

- **memory_state_accuracy dropped ~6 pp** across all models after removing `is_distractor` leakage — confirms the field was inflating memory-state scores (models could mark distractor memories `out_of_scope` without reading evidence).
- **non_answer_decision_accuracy is more meaningful** on balanced split: 75 non-answer gold cases (50%) vs 35 (23%) before. GLM reaches 0.373; Kimi/DeepSeek still near 0 because they default to `use_current_memory` on ask/escalate/refuse cases.
- **joint_revision_success remains below 0.5** for all models (best 0.227) — split is appropriately hard.
- **format_failure_rate == 0** for all models.

## Baselines (hard150_balanced)

| baseline | joint_revision_success | memory_state_accuracy |
| --- | --- | --- |
| latest_only | (see `latest_only.predictions.metrics.json`) | — |
| retrieve_all | (see `retrieve_all.predictions.metrics.json`) | — |

## hard500_candidate generation

All gates passed → generated **`data/retrace_bench_hard500_candidate/hard_500_en/scenarios.jsonl`**.

- Validator: **PASS**
- Gold oracle: **PASS** (500 scenarios; decision distribution scaled: use_current=250, mark_unresolved=100, ask=60, refuse=50, escalate=40)

## Files changed (implementation)

- `benchmark/retrace_bench/public_view.py` (new)
- `benchmark/retrace_bench/generation/pattern_spec.py`
- `benchmark/retrace_bench/generation/hard_plus_blueprints.py`
- `scripts/generate_retrace_bench_final.py`
- `scripts/run_retrace_bench_siliconflow.py`
- `scripts/run_retrace_bench_gemini_hard150.py`
- `scripts/run_retrace_bench_api_models.py`
- `scripts/run_retrace_bench_baseline.py`
- `tests/retrace_bench/test_public_view.py` (new)
- `tests/retrace_bench/test_hard_decision_schedule.py` (new)
- `tests/retrace_bench/test_pattern_spec.py` (updated)

## Tests run

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments benchmark scripts
.venv/bin/python -m pytest tests/retrace_bench/test_public_view.py tests/retrace_bench/test_hard_decision_schedule.py tests/retrace_bench/test_pattern_spec.py tests/retrace_bench/test_baseline_no_gold_leak.py::test_llm_json_answerer_accepts_fenced_json_response -q
```

All targeted tests pass (13/13).
