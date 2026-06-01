# ReTrace-Bench: A Benchmark for Agent Memory Revision Reliability

> Benchmark-paper framing notes. ReTrace-Bench is an **evaluation-only**
> benchmark and is independent of any training method (it does **not** depend
> on ReTrace-Learn, DPA, or any specific memory architecture). Any
> memory-enabled agent — LLM-only, RAG, CRUD store, Mem0-style system, or a
> trained policy — can be scored on it.

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

## 2. Gap: what existing benchmarks do *not* measure

Existing memory/agent benchmarks evaluate adjacent but distinct capabilities:

- **Long-context retrieval** (e.g. needle-in-a-haystack style): can a model find
  a fact in a long window? This measures recall over a *static* context, not
  whether the model *revises* a belief when newer, conflicting evidence arrives.
- **Long-term dialogue memory** (e.g. persona / session-history benchmarks):
  consistency of recalled facts across sessions, but typically without
  adversarial scope traps, trust conflicts, or policy constraints.
- **Stale-memory probes** (e.g. update-vs-stale tasks): detect that a fact
  changed, but usually as an isolated single-fact update rather than a typed,
  multi-failure-mode revision over a shared workflow trace.
- **General agent benchmarks** (tool use, web tasks): end-to-end task success,
  where memory reliability is entangled with planning and tool execution and is
  never scored in isolation.

None of these **directly** evaluate *shared memory revision reliability* — the
combination of (a) deciding the correct action under evolving evidence, scope,
trust/source conflict, and policy; (b) producing the correct per-memory state;
(c) grounding the decision in the minimal supporting evidence; and (d)
diagnosing the failure mode when an agent gets it wrong.

## 3. Contribution

ReTrace-Bench fills this gap with **structured workflow traces** and a metric
suite that scores revision behavior directly:

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

The headline split is **`sample_80_hard_en`** (80 scenarios, all 8 domains and
all 11 failure modes, deliberately adversarial). `sample_40_en` is a smaller
quick-sanity fixture. Larger generated splits (`dev_800_en`, `stress_1760_en`)
are artifacts and are intentionally not committed.

## 4. Metrics

Headline metrics (see `docs/retrace_bench` and `benchmark/retrace_bench/scorers_general.py`):

- **decision accuracy** (`black_box_decision_accuracy`) — strict enum match.
- **decision_macro_f1** / **decision_balanced_accuracy** — robust to the
  majority `use_current_memory` class.
- **non_answer_decision_accuracy** — accuracy on cases whose correct action is a
  *non-answer* (`escalate`, `ask_clarification`, `refuse_due_to_policy`,
  `mark_unresolved`); a strong test of whether an agent knows when *not* to act.
- **memory_state_accuracy** — fraction of memories given the correct status.
- **evidence_f1** — F1 of cited evidence event IDs vs. the minimal gold set.
- **failure_diagnosis_accuracy** — correctly naming the failure mode.
- **stale_reuse_rate** — rate of reusing a stale/wrong answer (lower is better;
  detected even when paraphrased).

Reported with **per-domain** and **per-failure-mode** breakdowns so weaknesses
are localizable rather than averaged away.

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
| `retrace_oracle_engine` | **oracle** | **yes** | **upper bound only — not deployable** |

**`retrace_oracle_engine` is an oracle upper bound, not a comparable deployable
method.** It is allowed to read hidden gold to construct the correct typed
revision and routes it through the deterministic engine; it exists to (a) verify
the benchmark is solvable and internally consistent and (b) bound the achievable
score. It must never be presented alongside the deployable baselines as if it
were a competing system. The runner enforces this: the oracle is grouped under
`is_oracle=true` / `group="oracle"`, separate from the deployable baselines.

See `baseline_results_sample_80_hard_en.md` for current numbers.
