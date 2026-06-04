# Gemini Hard150 Benchmark Summary

**Run status:** PAUSED at 20/150 — Google AI Studio returned HTTP 429 quota errors after ~20 successes. Resume with `scripts/run_retrace_bench_gemini_hard150.py` when quota resets.

## A. API check

- Status: **PASS** (`gemini-3.5-flash`)

## B. Model

- `gemini-3.5-flash` (override via `GEMINI_MODEL`)

## C. Run counts

- Success: **20** / 150
- Errors: **32** (HTTP 429 quota)
- Format failures on successes: **0**

## D. Metrics (partial n=20)

| metric | Gemini |
| --- | --- |
| decision_macro_f1 | 0.286 |
| black_box_decision_accuracy | 0.750 |
| non_answer_decision_accuracy | 0.000 |
| memory_state_accuracy | 0.567 |
| evidence_f1 | 0.502 |
| minimal_evidence_exact_match | 0.250 |
| evidence_precision | 0.575 |
| overcitation_rate | 0.325 |
| counterevidence_recall | 0.400 |
| failure_diagnosis_accuracy | 0.000 |
| stale_reuse_rate | 0.050 |
| latest_event_shortcut_failure_rate | 0.000 |
| answer_state_consistency | 0.000 |
| joint_revision_success | 0.000 |
| format_failure_rate | 0.000 |

Full partial JSON: `gemini_metrics_partial_n20.json`

## E. Gemini vs DeepSeek-V4-Pro

- Partial n=20: Gemini `joint=0.000`, `failure_diagnosis=0.000`
- DeepSeek full hard150 reference: `joint=0.247`, `failure_diagnosis=0.207`
- Not comparable until Gemini completes 150/150

## F. Hard150 discriminability

- Successful rows are valid JSON (not empty); joint remains 0 on this partial slice
- Quota limit prevents full discriminability assessment

## G. Scale to hard_500?

- **Not yet**: finish Gemini hard150 after quota recovery first

## Progress

See `progress.json` (`status: paused_quota_429`)
