# ReTrace-Bench Manual Validation Report

## Scope

- **Split:** `test_800_templateheldout_en`
- **Path:** `data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl`
- **Sample:** 88-cell stratified sample, covering 8 domains × 11 failure modes
- **Reviewed cells:** 88 / 88
- **Reviewer:** project author

This is a focused artifact validation pass for release readiness. It is not a
formal human-subject study, a large-scale annotation study, a multi-annotator
review, or an inter-annotator agreement measurement.

## Checks Performed

For each sampled cell, the manual pass checked:

1. **Visible-evidence solvability:** the public event trace, initial memory, and
   task text are sufficient to reach the intended decision without reading
   hidden gold as model input.
2. **Hidden-gold consistency:** `expected_decision`,
   `expected_memory_state`, `expected_evidence_event_ids`, and
   `expected_failure_diagnosis` are consistent with the revision logic implied
   by the trace.
3. **Public-text leakage:** visible scenario text does not disclose hidden labels
   or failure-mode names.
4. **Non-answer justification:** scenarios whose expected decision is
   `escalate`, `ask_clarification`, `refuse_due_to_policy`, or
   `mark_unresolved` contain visible evidence justifying that non-answer action.

## Outcome Summary

The completed pass found the 88-cell stratified sample suitable for artifact
release use under the checks above. No claim is made about exhaustive manual
inspection of all 800 scenarios beyond the automated validator and the
stratified manual sample.

The deterministic automated validator remains the release gate for the full
split.

## Known Limitations

- One project-author pass only.
- No independent annotator panel.
- No inter-annotator agreement statistic.
- The manual pass is a sample-based quality check, not a replacement for
  automated validation over the full split.

## Final Validator Command

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl
```
