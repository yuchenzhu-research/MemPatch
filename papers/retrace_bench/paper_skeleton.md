# ReTrace-Bench — Paper Skeleton

Skeleton for the benchmark paper. Each section points to consolidated prose in
`section_bank.md` and to the canonical source docs.

---

## Title (placeholder)

> ReTrace-Bench: A Benchmark for Agent Memory Revision Reliability under
> Evolving Evidence

## Abstract (placeholder)

One paragraph: agentic systems rely on persistent shared memory; the hard
problem is *revising* beliefs correctly as evidence evolves, not storing them.
ReTrace-Bench isolates memory revision reliability with deterministically
labeled scenarios, four structured task views, a five-way revision-decision
space, an eight-way memory-state space, an eleven-way failure-mode taxonomy, a
template-heldout test split with leakage checks, and an official scorer / public
API / evaluator CLI. Report headline metrics and the gap between deployable
baselines and a gold-replay oracle consistency reference.

## 1 Introduction

- Memory-augmented agents accumulate beliefs that must be revised as evidence
  changes; failures are silent (stale reuse, over/under update, scope leakage).
- Existing long-term / agent-memory benchmarks emphasize recall and retrieval;
  revision *reliability* under evolving, conflicting, scoped, policy-bound
  evidence is under-measured.
- Contributions: (1) task formulation of memory revision reliability; (2) a
  deterministically labeled, auditable dataset with a template-heldout test
  split; (3) four structured task views + canonical metric set; (4) an official
  scoring API, evaluator CLI, and HF artifact; (5) baseline suite with an
  explicit oracle boundary.
- Source prose: `section_bank.md#1-introduction`,
  `docs/retrace_bench/benchmark_paper.md`.

## 2 Task: Agent Memory Revision Reliability

- Definition: at probe time, for each memory entry decide keep / supersede /
  block / restore / forget / quarantine, and justify from visible evidence.
- Five-way decision space; non-answer decisions.
- Source: `section_bank.md#2-task`, `general_taxonomy.py`.

## 3 Benchmark Design

- Four task views (black-box answer, memory-state, evidence-retrieval,
  diagnostic) over a single scenario.
- Scenario anatomy: `public_input` (event trace, initial memory), hidden gold,
  trust levels, visibility scopes, distractors.
- Domains, difficulties, failure modes.
- Source: `section_bank.md#3-benchmark-design`,
  `docs/retrace_bench/dataset_design.md`, `docs/retrace_bench/schema.md`.

## 4 Data Construction

- Deterministic blueprint construction: hidden labels derived from a blueprint,
  not inferred by an LLM (reproducible, auditable).
- Validators: reference integrity, public-text hygiene, task coverage,
  distribution gates (`scripts/validate_retrace_bench_dataset.py`).
- Splits and roles; template-heldout construction and leakage checks.
- Source: `section_bank.md#4-data-construction`,
  `docs/retrace_bench/generation_and_audit_protocol.md`.

## 5 Evaluation Protocol and Metrics

- Headline metrics: `decision_macro_f1`, `non_answer_decision_accuracy`,
  `memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`,
  `stale_reuse_rate`.
- Auxiliary metrics (why `black_box_decision_accuracy` is not headline).
- Prediction schema; strict validation contract; oracle boundary.
- Source: `section_bank.md#5-evaluation-protocol-and-metrics`,
  `scorers_general.py`, `docs/retrace_bench/metrics_v2.md`.

## 6 Baselines

- Sanity (`latest_only`, `retrieve_all`); retrieval/memory (`rag_lexical`,
  `crud_memory`, `mem0_style`, `heuristic_memory_state`); gold-replay oracle
  consistency reference (`retrace_oracle_engine`).
- Source: `section_bank.md#6-baselines`; v1.0 baselines to be regenerated on
  `main_3000_en` (legacy pre-v1 artifacts were removed from the active tree).

## 7 Results

- Main table on the `main` split (`main_3000_en`); stress results on `hard`
  (`hard_300_en`).
- Gap between best deployable baseline and oracle consistency reference.
- Source: `table_plan.md` Table 4.

## 8 Analysis

- Per-failure-mode and per-domain breakdowns; difficulty-level trends.
- Where current baselines collapse (non-answer decisions, evidence grounding).
- Oracle memory-state ceiling (0.968) explanation.
- Source: `table_plan.md` Tables 5–7, baseline results doc.

## 9 Related Work

- Complementary to long-term / agent-memory benchmarks (LongMemEval, MemBench,
  LoCoMo; optionally MemoryAgentBench / EvoMemBench): ReTrace-Bench isolates the
  revision-authorization step rather than recall.
- Source: `section_bank.md#9-related-work`, `benchmark_paper.md`.

## 10 Limitations and Ethics

- Synthetic, deterministically constructed scenarios; English-only release;
  templated generation mitigated by template-heldout split + leakage checks.
- No human-subject data; safety/policy scenarios are illustrative.
- Source: `section_bank.md#10-limitations-and-ethics`.

## 11 Reproducibility / Artifact

- Public API, evaluator CLI, HF dataset, deterministic validators, tests.
- `PYTHONPATH=.` usage note; exact commands.
- Source: `artifact_checklist.md`, `examples/retrace_bench/README.md`.

## Appendix checklist

- Validation gates, leakage checks, oracle consistency, full taxonomy tables,
  prediction schema, per-domain/per-mode breakdowns. See `table_plan.md`
  appendix tables.
