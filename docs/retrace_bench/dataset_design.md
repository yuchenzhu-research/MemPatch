# ReTrace-Bench General English Dataset Design

ReTrace-Bench is a general benchmark for persistent-memory reliability in
agentic workflows. It is method-neutral: scenarios are not specific to
ReTrace-Learn, DPA, RAG, CRUD stores, Mem0-style systems, or any single model
family.

## Splits

- `data/retrace_bench/test_800_templateheldout_en/` - 800 scenario
  paper-facing held-out benchmark split.
- `data/retrace_bench/test_800_en/` - 800 scenario prototype/diagnostic split;
  retained for comparison only, not the final benchmark.
- `data/retrace_bench/sample_80_hard_en/` - 80 scenario hard sample for quick
  inspection and baseline smoke checks.
- `data/retrace_bench/sample_20_v2/` - tiny v2 schema smoke fixture.
- `data/retrace_supervision/train_3000_en/` and
  `data/retrace_supervision/dev_400_en/` - synthetic supervision/selection
  pools for future ReTrace-Learn work; not benchmark test sets.

Each split stores `scenarios.jsonl` plus a small `manifest.json`.

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
