# ReTrace-Bench Manual Validation Protocol

Status: **planned / recommended protocol.** This document specifies a credible,
honestly executable manual validation procedure for the canonical paper-facing
held-out split `data/retrace_bench/test_800_templateheldout_en/`. It does **not**
claim that a completed human annotation study has been performed. The
script-assisted stratified sample to be reviewed is enumerated in
[`manual_validation_sample_88.md`](manual_validation_sample_88.md); reviewer
checkboxes there are intentionally left unchecked until a human reviewer fills
them in.

This is a focused reviewer-facing protocol, not a large annotation study. It is
designed to be runnable by a small number of reviewers in a single pass.

## 1. Sampling

- Inspect a **stratified sample** covering all **8 domains × 11 failure modes**
  (88 cells). One scenario per present cell is enumerated in
  `manual_validation_sample_88.md`; all 88 cells are populated in the canonical
  split.
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
  the nature of the fix) in the reviewer-notes column of
  `manual_validation_sample_88.md`.
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

Do **not** describe this protocol as a completed human validation study unless a
filled-in `manual_validation_sample_88.md` (or an equivalent signed manifest)
accompanies it. Until then this is a recommended protocol plus an enumerated
sample template.
