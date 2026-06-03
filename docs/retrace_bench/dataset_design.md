# ReTrace-Bench General English Dataset Design

ReTrace-Bench is a general benchmark for persistent-memory reliability in
agentic workflows. It is method-neutral: scenarios are not specific to
ReTrace-Learn, DPA, RAG, CRUD stores, Mem0-style systems, or any single model
family.

## Splits (ReTrace-Bench v1.0)

Four paper-facing splits, public names `main` / `hard` / `realistic` /
`calibration` (never train / dev / validation / test):

- `data/retrace_bench/main_3000_en/` (`main`, 3000) - **controlled benchmark
  main split**. All headline numbers come from here.
- `data/retrace_bench/hard_300_en/` (`hard`, 300) - rule-defined long-context /
  multi-evidence / multi-memory **stress split** (20–100 events, ≥5 memories,
  ≥2 evidence events per case).
- `data/retrace_bench/realistic_100_en/` (`realistic`, 100) - realistic-style
  workflow split; `source_type = realistic_style_synthetic`,
  `annotation_status = pending` (gold not yet annotated; template in
  `annotations_template.jsonl`). No human validation or public-source
  provenance is claimed.
- `data/retrace_bench/calibration_80_en/` (`calibration`, 80) - **smoke /
  quickstart** split only; **not** for model selection, checkpoint selection,
  tuning, or headline claims.

Supervision / selection pools for learning-based systems live outside the
benchmark tree under `data/retrace_learn/supervision_train_3000_en/` and
`data/retrace_learn/supervision_dev_400_en/`; they are **not** benchmark test
sets. The legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

Each split stores `scenarios.jsonl` plus a small `manifest.json` and a
`README.md` (the realistic split also ships `annotations_template.jsonl`).

## Generation Model

Generation is blueprint-first:

1. `scripts/generate_retrace_bench_blueprints.py` creates deterministic hidden
   blueprints with domain, failure mode, difficulty, scope traps, evidence
   events, memory IDs, and expected statuses.
2. `scripts/render_retrace_bench_dataset.py` renders each blueprint into an
   English workflow trace and task prompts.
3. Hidden labels are copied from the blueprint-derived state, not inferred by a
   language model.
4. `scripts/validate_retrace_bench_dataset.py` checks references, public text,
   task coverage, and distribution gates.

## Scenario Shape

Every scenario includes:

- `workflow_context`
- `public_input.event_trace`
- `public_input.initial_memory`
- four task types: black-box, memory state, evidence retrieval, diagnostic
- `hidden_gold` with expected answer, optional decision, evidence IDs, memory
  states, failure diagnosis, wrong-answer traps, and rubric

Visible scenario text avoids benchmark-specific terminology and never exposes
hidden labels directly.

## Coverage

The generator covers eight enterprise-style domains and eleven memory failure
modes. Difficulty follows a deterministic 15/30/35/20 distribution across L1-L4.
Validation enforces minimum rates for distractors, cross-scope traps, verified
contradictions of stale trusted notes, false-premise rejection, and non-answer
actions.

## Manual Validation

Beyond the automated validators, a recommended manual validation protocol
covers a stratified 8-domain × 11-failure-mode sample of the `main` split. See
[`manual_validation_protocol.md`](manual_validation_protocol.md)
for the procedure and [`manual_validation_sample_88.md`](manual_validation_sample_88.md)
for the enumerated sample / report template. The manual protocol is documented
as planned/recommended and is not claimed as a completed human study until the
report template is filled in.
