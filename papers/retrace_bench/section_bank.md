# ReTrace-Bench — Section Bank

Consolidated prose for the benchmark paper. Trim/adapt when moving to LaTeX.
Canonical sources are noted per section.

---

## 1 Introduction

Agentic workflows increasingly rely on a persistent, shared memory: beliefs,
facts, task state, and preferences that one or more agents write once and reuse
across many later steps. The hard problem is not *storing* memory but *revising*
it correctly as evidence evolves. A reliable agent must, at probe time, decide
for every memory entry whether to keep, supersede, block, restore, forget, or
quarantine it — and must justify that decision from the visible evidence. When
revision fails, the failures are silent and compounding: stale beliefs are
reused, corrections are missed or over-applied, scoped information leaks across
contexts, and policy-blocked content resurfaces.

Existing long-term and agent-memory benchmarks largely measure recall and
retrieval quality over accumulated context. ReTrace-Bench is complementary: it
isolates the *revision-authorization* step and turns it into an operational
evaluation target. It is method-neutral and evaluation-only — any memory-enabled
agent (LLM-only, RAG, CRUD store, Mem0-style system, or a trained policy) can be
scored — and it does not depend on any specific memory architecture.

Contributions:
1. A task formulation of agent memory revision reliability under evolving
   evidence, with a canonical five-way revision-decision space.
2. A deterministically labeled, auditable dataset: hidden labels are derived
   from a blueprint rather than inferred by an LLM, with a template-heldout test
   split and explicit leakage checks.
3. Four structured task views over a single scenario and a canonical headline /
   auxiliary metric set.
4. An official, stable scoring API, an evaluator CLI that runs no model and
   needs no API keys, example predictions, and a packaged dataset card.
5. A baseline suite (sanity, retrieval, memory-architecture) plus a gold-replay
   oracle consistency reference with an explicit oracle boundary.

*Source:* `docs/retrace_bench/benchmark_paper.md`.

## 2 Task

A scenario presents an agent with a visible event trace and an initial shared
memory, then probes it. For every relevant memory entry the agent must produce a
revision decision and supporting justification. The canonical decision space is
five-way:

- `use_current_memory` — answer from the currently authorized belief;
- `escalate` — defer because evidence is insufficient/conflicting;
- `ask_clarification` — request missing information;
- `refuse_due_to_policy` — withhold due to a policy/permission block;
- `mark_unresolved` — flag the belief as not currently usable.

The last four are *non-answer* decisions, scored by
`non_answer_decision_accuracy`, because correct behavior is frequently *not* to
answer from current memory. This distinguishes a system that reliably abstains
under stale/blocked/uncertain evidence from one that always answers.

*Source:* `benchmark/retrace_bench/general_taxonomy.py`.

## 3 Benchmark Design

**Four task views.** Each scenario is scored from four structured views that
share one underlying state:
- *Black-box task*: the free-text answer a downstream user would receive.
- *Memory-state task*: a map `memory_id -> status` over the eight-way memory
  status space.
- *Evidence-retrieval task*: the set of event IDs that justify the decision.
- *Diagnostic task*: the primary failure mode the scenario probes.

**Scenario anatomy.** `public_input` exposes an `event_trace` (each event has an
ID, actor, type, text, timestamp, trust level, visibility scope, and related
memory IDs) and an `initial_memory` (each entry has an ID, text, source events,
visibility scope, and a distractor flag). Hidden gold (decision, memory state,
evidence IDs, failure diagnosis, answer key facts) is never exposed to models.

**Taxonomy.**
- *Domains (8):* software engineering agent, enterprise multi-tool workflow,
  customer-support CRM, calendar/task workflow, research/knowledge work,
  personal-assistant preference, e-commerce recommendation, data-analysis BI.
- *Difficulties (L1–L4):* single-hop update; multi-hop with distractor;
  conditional validity; cross-scope adversarial audit.
- *Trust levels:* verified, trusted, untrusted.

*Source:* `docs/retrace_bench/dataset_design.md`, `docs/retrace_bench/schema.md`,
`general_taxonomy.py`.

## 4 Data Construction

**Deterministic blueprints.** Each scenario is instantiated from a blueprint
that fixes the gold labels by construction; labels are not inferred by an LLM, so
they are reproducible and auditable. A scenario's public text is generated to be
hygienic (no benchmark-identifying or gold-leaking terms).

**Validators.** `scripts/validate_retrace_bench_dataset.py` enforces reference
integrity (evidence/memory IDs resolve), public-text hygiene, task coverage
(all four views present), and distribution gates (e.g. minimum event counts,
distractor and cross-scope rates, non-answer share, verified-over-trusted rate).

**Splits and roles (ReTrace-Bench v1.0).** Four paper-facing splits, public
names `main` / `hard` / `realistic` / `calibration` (never train / dev /
validation / test):
- `main_3000_en` (`main`, 3000) — controlled benchmark main split; all headline
  numbers come from here. Never train/tune/select on it.
- `hard_300_en` (`hard`, 300) — rule-defined long-context / multi-evidence /
  multi-memory stress split (20–100 events, ≥5 memories, ≥2 evidence per case).
- `realistic_100_en` (`realistic`, 100) — realistic-style workflow split;
  `source_type = realistic_style_synthetic`, `annotation_status = pending`. No
  human validation or public-source provenance is claimed.
- `calibration_80_en` (`calibration`, 80) — smoke / quickstart split only; never
  for model or checkpoint selection or headline claims.

ReTrace-Learn (the method track) uses ReTrace-Bench-derived scenario data with
declared split roles; the same scenario family may be consumed for proposal
learning and evaluation. Split roles are explicit, and leakage-free held-out
evaluation is not claimed where the same gold labels are used for training. The
legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

**Leakage audit.** Every split is de-actionalized and passes a decision-word
leakage audit: no authoritative/verified record contains a decision phrase tied
to one of the five gold decisions, so the gold decision must be recovered by
reasoning over state rather than string matching. The per-split `manifest.json`
records a `leakage_audit_summary`; the audit logic lives in
`benchmark/retrace_bench/generation/release_manifest.py`. Legacy
template-signature reports (`template_signature_report.md`,
`split_leakage_report.md`) are retained for provenance only.

*Source:* `docs/retrace_bench/generation_and_audit_protocol.md`,
`docs/retrace_bench/contamination_policy.md`.

## 5 Evaluation Protocol and Metrics

**Headline metrics** (the numbers a paper should lead with):
- `decision_macro_f1` — macro-F1 over the five-way decision space (primary
  decision metric; robust to the majority `use_current_memory` class).
- `non_answer_decision_accuracy` — accuracy on scenarios whose correct action is
  a non-answer decision.
- `memory_state_accuracy` — accuracy of the predicted `memory_id -> status` map.
- `evidence_f1` — F1 of predicted vs. gold evidence event IDs.
- `failure_diagnosis_accuracy` — accuracy on the eleven-way failure-mode label.
- `stale_reuse_rate` — rate of reusing a stale belief when it should have been
  revised (lower is better).

**Auxiliary metrics** (reported for completeness, not as headline):
`black_box_decision_accuracy` (can be dominated by the majority class),
`decision_balanced_accuracy`, `use_current_memory_accuracy`,
`answer_key_fact_accuracy`, `answer_exact_match`, `format_failure_rate`.

**Prediction schema and strict contract.** A prediction is one JSON object per
line keyed by `scenario_id`, with a `response` containing `decision`,
`memory_state`, `evidence_event_ids`, `failure_diagnosis`, and `answer`
(canonical nested or flat form). In strict mode the official scorer rejects
unknown decision labels, unknown memory-state statuses, evidence IDs absent from
the event trace, and failure-diagnosis values that are neither a canonical label
nor a documented alias; incomplete memory-state coverage of visible memory IDs
is a warning. Validation messages reference only model-visible IDs.

**Oracle boundary.** The gold-replay oracle may read hidden gold and replay the
gold typed revision through the deterministic engine. It is a consistency
reference that (a) confirms the benchmark is solvable from the gold labels and
(b) upper-bounds achievable state/evidence/diagnosis scores. It must never be
listed alongside deployable baselines as a competing system.

*Source:* `benchmark/retrace_bench/scorers_general.py`,
`docs/retrace_bench/metrics_v2.md`.

## 6 Baselines

- *Sanity:* `latest_only` (always trust the most recent write), `retrieve_all`
  (return everything) — calibrate the floor and majority-class behavior.
- *Retrieval / memory architectures:* `rag_lexical` (lexical retrieval),
  `crud_memory` (explicit create/update/delete store), `mem0_style` (an in-repo
  heuristic emulation of a popular memory framework), `heuristic_memory_state`.
- *Oracle consistency reference:* `retrace_oracle_engine` (gold-replay; not
  deployable, reported separately).

These baselines establish that the benchmark is non-trivial: simple
recency/retrieval heuristics collapse on non-answer decisions and evidence
grounding even when they score reasonably on majority-class accuracy.

*Source:*
a v1.0 offline baseline run on `main_3000_en` (to be regenerated; the legacy
`docs/retrace_bench/baseline_results_test_800_templateheldout_en.md` is retained
for provenance only).

## 9 Related Work

ReTrace-Bench is complementary to existing long-term and agent-memory
benchmarks (e.g. LongMemEval, MemBench, LoCoMo, and optionally MemoryAgentBench /
EvoMemBench). Those primarily evaluate recall, retrieval, and long-context
consistency. ReTrace-Bench instead isolates the revision-authorization decision
— whether a belief is still usable given evolving, conflicting, scoped, and
policy-bound evidence — and provides structured, deterministically labeled views
for it. It is positioned as an operational evaluation target that augments rather
than replaces recall-oriented benchmarks.

*Source:* `docs/retrace_bench/benchmark_paper.md`.

## 10 Limitations and Ethics

- Scenarios are synthetic and deterministically constructed; this buys
  reproducible, auditable labels at the cost of natural-language diversity. The
  template-heldout split and leakage probes mitigate template memorization.
- The current public release is English-only.
- Policy/permission scenarios are illustrative and do not encode any real
  organization's policy. No human-subject or private data is used.
- The oracle reference reads hidden gold by design and is never a deployable
  system; reporting it as a baseline would be misleading.
