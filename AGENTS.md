# ReTrace Agent Instructions

This is the first file every coding model must read before editing this
repository.

## Current State

Dynamic branch, HEAD, smoke, and validation status live only in
`docs/implementation_status.md`. This file contains stable agent instructions,
not volatile run status.

## Canonical Reading Order

Read only these active authority documents before method work:

1. `AGENTS.md`
2. `docs/method_spec_dpa.md`
3. `docs/stage_ab_protocol.md`
4. `docs/implementation_status.md`
5. `docs/upstream_integration.md`
6. `docs/coding_contract.md`
7. `docs/repository_execution_contract.md`
8. `docs/stage_c_report.md`
9. `docs/paper1_blueprint_zh.md`

Legacy planning documents and old raw source-material files are no longer
active authority. Git history preserves them.

## One-Sentence Alignment

ReTrace preserves immutable evidence and changes a belief's eligibility for
current answers only through verified, temporally valid typed defeat paths
computed by deterministic DPA.

## Method Boundary

Paper 1 studies evidence-preserving reversible authorization for evolving agent
memory:

```text
immutable evidence ledger
+ typed belief/condition/evidence-edge graph
+ deterministic Defeat-Path Authorization Algorithm
```

The central research question is:

> Can local typed-edge prediction plus deterministic, auditable DPA authorize
> current belief use more reliably than direct LLM adjudication, while
> preserving original evidence and allowing later reversal?

ReTrace still uses a semantic model in Stage A to propose local evidence edges.
It does not eliminate model judgment. It restricts semantic-model judgment to
local typed proposals and delegates final belief authorization to deterministic
DPA.

Stage A v1 uses effect-triggered authorization revision: the candidate belief is
already evidence-supported, and later evidence changes current authorization
only through an admitted direct local typed effect on the belief, a supplied
required condition, or a supplied grounded replacement. Irrelevant or silent
new evidence must produce an empty edge set and preserve existing authorization;
`UNCERTAIN` is reserved for directly relevant but unresolved evidence.

## Do Not Drift

Do not turn this codebase into:

- generic RAG;
- a Mem0 clone;
- a Graphiti clone;
- CUPMem fixed-slot state tracking;
- RL memory action learning;
- latent memory, memory-token, or learned consolidation work;
- a new benchmark generator;
- a publishable hand-written heuristic scaffold;
- an unconstrained LLM judge that directly rewrites memory.

Do not let legacy flat `RelationPrediction`, `CONDITION`, `SUPPORT`, or
`REQUIRED_BY` semantics govern new runtime method work.

## Canonical Runtime Vocabulary

Only this typed scheme is canonical for new method documentation and runtime
work:

- `DependencyEdge(REQUIRES)`: belief -> condition.
- `EvidenceEdge(BLOCKS)`: evidence -> condition.
- `EvidenceEdge(RELEASES)`: evidence -> condition.
- `EvidenceEdge(SUPERSEDES)`: evidence -> prior belief, with grounded
  `replacement_belief_id`.
- `EvidenceEdge(REAFFIRMS)`: evidence -> belief.
- `EvidenceEdge(UNCERTAIN)`: evidence -> belief.

## Stage Identities

- Stage A, `ReTrace-LLM`: main method path. In the primary controlled track it
  consumes a fixed `SharedCandidateView`, predicts local typed evidence edges,
  admits them through `RevisionGate`, and computes authorization with DPA. It
  does not directly emit final usability verdicts.
- Stage B, `DirectJudge-LLM`: shared-view-controlled direct-adjudication
  baseline. It consumes the same fixed semantic view and directly emits
  `USABLE`, `NOT_USABLE`, or `UNCERTAIN`. It is not an
  `EvidenceEdgeVerifier`, does not use DPA, and is not strict call-budget
  matched.
- Stage C, `ReTrace-Local`: deferred learned local typed-edge verifier using
  the same DPA core. Its initiation has been formally deferred per the analysis in `docs/stage_c_report.md`.

## Safe Next Boundary

The next boundary is the scientific hardening of the STALE pathway and evaluation protocol. It is not Stage C and not another Memora expansion.

Stage A v1 controlled semantics are retained. The per-belief verifier path
(`ControlledReTraceLLM`) is the auditable controlled reference implementation.
It is not a scalable benchmark-execution interface because it repeats the same
evidence context once per candidate belief: O(B) semantic-model calls and
approximately O(B × |E|) repeated prompt tokens.

Benchmark-facing development uses a bounded batched local typed-edge proposal
path that preserves the same typed-edge vocabulary, RevisionGate, and DPA
semantics while reducing repeated semantic-model calls.

The primary external benchmark is the official frozen STALE 400-case dataset
(`STALEproj/STALE`, file `T1_T2_400_FULL.json`, license CC BY 4.0), which is
publicly released and downloaded into the gitignored
`data_external/stale_official_frozen/` directory. STALE end-to-end evaluation
is explicitly labeled as a secondary external answer-level evaluation. The
primary controlled structural-attribution proof remains the fixed
`SharedCandidateView` comparison.

The Memora oracle-conditioned 30-question execution is retained only as an
internal rejected-pilot/adapter-misalignment artifact. It must not be rerun and
must not be cited as a positive method result. No official Memora or FAMA
result exists.

Method-visible STALE fields are `uid`, the ordered `haystack_session`, the
aligned `timestamps`, and the three `probing_queries`. Evaluator-only fields
are `M_old`, `M_new`, `explanation`, `relevant_session_index`, and `type`
(`type` may be used only for post-run stratification, never as method input).
If `M_old` or `M_new` text appears independently inside the genuine haystack
sessions, that haystack text remains legitimately method-visible; the
prohibition is against directly injecting the separate gold fields or
`relevant_session_index` into the method.

Completed and retained:

- validated AB-1A.5 and AB-1B controlled offline protocol;
- provider/cache/manifest scaffolding;
- secondary end-to-end development runner scaffolding;
- official frozen STALE adapter and offline non-leaking Stage A/B wiring
  scaffolding;
- legacy STALE smoke/dry-run scaffolding (kept for history, not the canonical
  STALE entrypoint);
- Ambiguity-and-Scope internal feasibility diagnostic (completed and retained as a feasibility diagnostic).

Deferred or prohibited for the current task:

- Stage C training;
- model training or distillation;
- full official STALE/Memora evaluation;
- prompt tuning or training from official scored examples/evaluator outputs;
- treating Stage A predictions as gold labels;
- claiming official results from mock/dry-run infrastructure.

## Coding Rules

- Standard library first.
- Keep core DPA logic API-free and deterministic.
- Keep benchmark-specific logic in adapters or runners.
- Do not edit `reference/`.
- Do not commit `reference/`, `outputs/`, caches, local environments, generated
  artifacts, benchmark downloads, or API keys.
- Preserve the canonical dataclass contracts in `src/retracemem/schemas.py`.
- Keep all paper-relevant method paths JSON-compatible and provenance-rich.
- Add or update tests for every new behavior.

## Verification

Compile:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
```

Full offline tests:

```bash
.venv/bin/python -m pytest
```

Do not run live providers or official benchmark evaluation unless a later task
explicitly authorizes that stage.
