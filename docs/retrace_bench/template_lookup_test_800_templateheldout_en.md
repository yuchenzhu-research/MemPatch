# Template Lookup Diagnostic

> **Legacy (pre-v1.0) document.** Describes a legacy pre-v1.0 split/pilot, recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`. Retained for provenance only; it does **not** describe the ReTrace-Bench v1.0 splits (`main`/`hard`/`realistic`/`calibration`).


This is a diagnostic shortcut baseline, not a deployable memory baseline. It predicts from de-identified template signatures learned from the training split.

| metric | value |
| --- | ---: |
| train scenarios | 3000 |
| test scenarios | 800 |
| coverage rate | 0.000 |
| decision accuracy | 0.291 |
| decision macro-F1 | 0.090 |
| failure-mode accuracy | 0.091 |

Fallback for unseen signatures: decision `use_current_memory`, failure mode `stale_memory_reuse`.

## Example Predictions

| scenario | covered | expected decision | predicted decision | expected failure | predicted failure |
| --- | ---: | --- | --- | --- | --- |
| `rt-templateheldout-test-000001` | false | `use_current_memory` | `use_current_memory` | `stale_memory_reuse` | `stale_memory_reuse` |
| `rt-templateheldout-test-000002` | false | `use_current_memory` | `use_current_memory` | `under_update` | `stale_memory_reuse` |
| `rt-templateheldout-test-000003` | false | `use_current_memory` | `use_current_memory` | `over_update` | `stale_memory_reuse` |
| `rt-templateheldout-test-000004` | false | `mark_unresolved` | `use_current_memory` | `conflict_collapse` | `stale_memory_reuse` |
| `rt-templateheldout-test-000005` | false | `use_current_memory` | `use_current_memory` | `scope_leakage` | `stale_memory_reuse` |
| `rt-templateheldout-test-000006` | false | `refuse_due_to_policy` | `use_current_memory` | `policy_violation` | `stale_memory_reuse` |
| `rt-templateheldout-test-000007` | false | `use_current_memory` | `use_current_memory` | `wrong_source_attribution` | `stale_memory_reuse` |
| `rt-templateheldout-test-000008` | false | `ask_clarification` | `use_current_memory` | `memory_hallucination` | `stale_memory_reuse` |
| `rt-templateheldout-test-000009` | false | `use_current_memory` | `use_current_memory` | `unnecessary_memory_write` | `stale_memory_reuse` |
| `rt-templateheldout-test-000010` | false | `use_current_memory` | `use_current_memory` | `failure_to_forget` | `stale_memory_reuse` |
| `rt-templateheldout-test-000011` | false | `use_current_memory` | `use_current_memory` | `failure_to_release_or_restore` | `stale_memory_reuse` |
| `rt-templateheldout-test-000012` | false | `ask_clarification` | `use_current_memory` | `stale_memory_reuse` | `stale_memory_reuse` |
