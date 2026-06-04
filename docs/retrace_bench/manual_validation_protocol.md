# ReTrace-Bench Manual Validation Protocol

Status: **reusable protocol.** This document specifies the manual validation
checks. Legacy pre-v1 artifacts were removed from the active tree; this protocol
should be run on the ReTrace-Bench v1.0 `main` split — the v1.0 splits have
**not** yet been manually validated.

This is a focused artifact validation pass, not a formal human-subject study,
large-scale annotation effort, multi-annotator review, or inter-annotator
agreement measurement. The automated validator remains required for the final
released split.

## 1. Sampling

- Inspect a **stratified sample** covering all **8 domains × 11 failure modes**
  (88 cells). Enumerate one scenario per present cell from the canonical split.
- Optionally over-sample harder difficulty tiers (L3/L4) and the
  `failure_to_forget` / `failure_to_release_or_restore` / `policy_violation`
  modes, which exercise the less common memory states.

## 2. Per-scenario checks

For each sampled scenario the reviewer verifies:

1. **Solvable from visible evidence alone.** The public trace
   (`public_input.event_trace` + `public_input.initial_memory` + `tasks`) is
   sufficient to reach the intended decision/answer without access to
   `hidden_gold`.
2. **Hidden labels match the intended revision logic.** `hidden_gold`
   (`expected_decision`, `expected_memory_state`, `expected_evidence_event_ids`,
   `expected_failure_diagnosis`) is consistent with how the evidence evolves in
   the trace.
3. **No hidden-label leakage into public text.** The visible scenario text does
   not reveal gold labels, benchmark terminology, or the failure-mode name
   (see `PUBLIC_FORBIDDEN_TERMS` in `benchmark/retrace_bench/general_taxonomy.py`).
4. **Non-answer decisions are justified.** When `expected_decision` is a
   non-answer action (`escalate`, `ask_clarification`, `refuse_due_to_policy`,
   `mark_unresolved`), the trace contains a concrete reason that action is
   correct rather than answering from current memory.

## 3. Recording fixes

- Record any blueprint/renderer fix triggered by review (scenario id, cell, and
  the nature of the fix) in the reviewer notes or the completed validation
  report.
- Fixes must be applied at the blueprint/renderer level and the split
  regenerated; do not hand-edit `scenarios.jsonl`.

## 4. Automated gate (required)

The final released split must pass the automated validators regardless of the
manual pass:

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl
```

This enforces reference integrity, public-text hygiene, task coverage, and
distribution gates (distractors, cross-scope traps, verified contradictions of
stale trusted notes, false-premise rejection, and non-answer actions).

## 5. Honest-status rule

Describe the completed pass precisely: one project-author manual pass over the
8-domain × 11-failure-mode stratified sample, checking visible-evidence
solvability, hidden-gold consistency, public-text leakage, and non-answer
justification. Do **not** describe it as a formal human-subject study, a
large-scale annotation study, a multi-annotator review, or an inter-annotator
agreement result.
