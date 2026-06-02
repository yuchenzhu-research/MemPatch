# ReTrace-Bench: A Benchmark for Agent Memory Revision Reliability

> **Independent benchmark paper.** ReTrace-Bench is a stand-alone
> benchmark / resource / evaluation paper for agent memory revision
> reliability under evolving evidence. It is **method-neutral** and
> **evaluation-only**: it does **not**
> depend on ReTrace-Learn, DPA, or any specific memory architecture, and it is
> not an evaluation component of ReTrace-Learn. Any memory-enabled agent —
> LLM-only, RAG, CRUD store, Mem0-style system, or a trained policy — can be
> scored on it. It is **complementary** to existing long-term and agent memory
> benchmarks rather than a replacement for them.

## 1. Problem: shared memory revision reliability

Agentic workflows increasingly rely on a **persistent, shared memory**: beliefs,
facts, task state, and preferences that one or more agents write once and reuse
across many later steps. The hard problem is not *storing* memory but
**revising** it correctly as evidence evolves. A reliable agent must, at probe
time, decide for every memory entry whether to keep, supersede, block, restore,
forget, or quarantine it — and must justify that decision from the visible
evidence.

Revision is where memory-enabled agents silently fail. Concretely, ReTrace-Bench
targets reliability under:

- **evolving evidence** — a later verified record overrides an earlier note;
- **scope boundaries** — a look-alike item in another workspace must not leak in;
- **trust / source conflicts** — untrusted or unsupported claims must not defeat
  verified records;
- **policy constraints** — some content (e.g. secrets) must never be stored;
- **forget / restore** — memory must be removed when obsolete and restored when
  a temporary block is lifted;
- **stale memory reuse** — the most common failure: confidently reusing an
  outdated answer that is still lexically plausible.

## 2. Positioning relative to existing benchmarks

Recent benchmarks already probe important aspects of long-term and agent memory,
including long-term conversational memory, knowledge updates, abstention,
factual/reflective memory, and agent memory capability. ReTrace-Bench is
complementary: rather than measuring how much an agent can remember, it isolates
memory-revision reliability under evolving evidence as an operational evaluation
target. Each scenario is scored through structured views: revision decision over
a five-way action space, per-memory state, supporting evidence retrieval, and
failure-mode diagnosis under scope, trust, policy, and stale-reuse traps.

Concretely, related resources sit along adjacent axes:

- **LongMemEval** — long-term conversational memory over extended sessions:
  emphasizes how much an assistant can recall and stay consistent across time.
- **MemBench** — agent memory capability and effectiveness under varied memory
  operations.
- **LoCoMo** — very long-term conversational memory and reasoning over
  multi-session dialogue history.
- **MemoryAgentBench** *(if appropriate)* — agentic memory across read/write/
  update operations and downstream task use.
- **EvoMemBench** *(if appropriate)* — memory behavior under evolving
  information over time.

ReTrace-Bench is complementary to these: it isolates *revision reliability
rather than memory capacity alone*, using structured multi-view scoring over a
typed five-way decision space and an explicit failure-mode taxonomy. We do not
claim to subsume or replace prior work; rather, ReTrace-Bench provides an
operational evaluation target for *what an agent should do to a memory entry
when the evidence changes* — keep, supersede, block, restore, forget, or
quarantine it — and whether that action is grounded in the visible evidence.

## 3. Contribution

ReTrace-Bench contributes **structured workflow traces** and a metric suite that
scores revision behavior directly, as an operational evaluation target
complementary to existing memory benchmarks:

1. **Structured scenarios.** Each scenario is a workflow trace — an
   `event_trace` of timestamped, multi-source events (mixed trust levels,
   actors, and visibility scopes), an `initial_memory` snapshot (including
   distractors), and a hidden, auditable gold (`hidden_gold`) that is never
   exposed in model input. Hidden labels are derived deterministically from a
   blueprint, not inferred by an LLM, so they are reproducible and auditable.
2. **Four task views per scenario.** `black_box_task` (decide/answer),
   `memory_state_task` (classify each memory), `evidence_retrieval_task` (cite
   the minimal supporting events), and `diagnostic_task` (name the failure mode).
3. **Explicit failure-mode taxonomy.** 11 reliability failure modes
   (stale reuse, under/over-update, conflict collapse, scope leakage, policy
   violation, wrong-source attribution, hallucination, unnecessary write,
   failure to forget, failure to release/restore) across 8 enterprise-style
   domains, with difficulty tiers L1–L4.
4. **Leakage-resistant scoring.** Decisions use strict enum matching (no
   substring shortcuts); answers use token-F1 with `must_include` /
   `must_not_include` rubrics so a "retrieve-everything" answer that stuffs the
   gold string into unrelated text does not score well; stale reuse is detected
   even when the stale answer is *paraphrased*.

### Split roles

- **`test_800_templateheldout_en`** is the canonical paper-facing held-out test
  split (800 scenarios, all 8 domains and all 11 failure modes, template-held
  out from train/dev). All headline numbers come from this split.
- **`test_800_en`** is a prototype/diagnostic split only and must **not** be
  used for paper headline numbers.
- **`sample_80_hard_en`** is a calibration / quickstart / smoke split for quick
  inspection and pipeline verification. On Hugging Face it may be exposed as the
  `validation` split for dataset-viewer compatibility only; it is **not** a
  model-selection / checkpoint-selection validation set.
- **`train_3000_en`** and **`dev_400_en`** (under `data/retrace_supervision/`)
  are supervision / selection pools for learning-based systems and are **not**
  benchmark test sets.

## 4. Metrics

The paper-facing **headline metrics** (the constants `HEADLINE_METRICS` in
`benchmark/retrace_bench/scorers_general.py`, computed in `aggregate_metrics`):

- **decision_macro_f1** — *primary decision metric*; macro-averaged F1 over the
  five-way decision space, robust to the dominant `use_current_memory` class.
- **non_answer_decision_accuracy** — accuracy on cases whose correct action is a
  *non-answer* (`escalate`, `ask_clarification`, `refuse_due_to_policy`,
  `mark_unresolved`); a strong test of whether an agent knows when *not* to act.
- **memory_state_accuracy** — fraction of memories given the correct status.
- **evidence_f1** — F1 of cited evidence event IDs vs. the minimal gold set.
- **failure_diagnosis_accuracy** — correctly naming the failure mode.
- **stale_reuse_rate** — rate of reusing a stale/wrong answer (**lower is
  better**; detected even when paraphrased).

**Auxiliary metrics** (`AUXILIARY_METRICS`; reported for completeness, not as
headline numbers):

- **black_box_decision_accuracy** — strict enum decision match. *Reported as an
  auxiliary raw decision signal because it can be dominated by the majority
  `use_current_memory` class*; `decision_macro_f1` is the headline decision
  metric instead.
- **decision_balanced_accuracy** — macro-averaged per-class recall.
- **answer_key_fact_accuracy** — rubric / token-F1 key-fact match for answers.
- **answer_exact_match** — strict normalized equality (diagnostic only; too
  strict for open text).
- **format_failure_rate** — rate of unparseable / missing decisions.
- **per-domain** and **per-failure-mode** breakdowns so weaknesses are
  localizable rather than averaged away.

## 5. Baselines and the oracle boundary

ReTrace-Bench ships a baseline suite spanning sanity, retrieval, and
memory-architecture families:

| baseline | family | reads hidden gold? | role |
| --- | --- | --- | --- |
| `latest_only` | sanity | no | answer with the most recent event |
| `retrieve_all` | sanity | no | dump all trusted events |
| `rag_lexical` | retrieval | no | lexical top-k retrieval, no mutation semantics |
| `crud_memory` | memory store | no | last-write-wins CRUD over visible IDs |
| `mem0_style` | memory store | no | compact fact store with add/update/delete keywords |
| `llm_json_answerer` | API model | no | direct LLM, strict-JSON schema (provider required) |
| `retrace_oracle_engine` | **oracle** | **yes** | **gold-replay consistency reference — not a deployable method** |

**`retrace_oracle_engine` is a gold-replay consistency reference / oracle
consistency diagnostic, not a comparable deployable method.** It is allowed to
read `hidden_gold` (answer, decision, evidence, diagnosis) and replays the gold
typed revision through the deterministic engine; it exists to (a) verify the
benchmark is internally consistent and solvable from the gold labels and
(b) bound the achievable state/evidence/diagnosis scores. It must never be
presented alongside the deployable baselines as if it were a competing system.
The runner enforces this: the oracle is grouped under
`is_oracle=true` / `group="oracle"`, separate from the deployable baselines.

See `baseline_results_test_800_templateheldout_en.md` for current paper-facing
numbers on the canonical held-out split, and `baseline_results_sample_80_hard_en.md`
for the calibration/quickstart split.
