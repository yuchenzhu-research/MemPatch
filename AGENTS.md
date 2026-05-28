# ReTrace Agent Instructions

This is the first file every coding model must read before editing this
repository.

## Current Phase

Repository branch: `method/retrace-llm-directjudge`.

Stage AB-1A.5 is complete. The repository has an offline, controlled Stage A/B
authorization protocol with auditability and comparison-claim safeguards:

- mandatory `SharedCandidateView.new_evidence`;
- derived `view_fingerprint` over first-class controlled-input fields, with
  metadata excluded;
- traced Stage A edge-verifier calls, including zero-edge calls;
- loud rejection for invalid fixed dependency anchors;
- retained admitted/rejected edge-proposal provenance and gate reasons;
- DirectJudge prompt v1 with explicit current/new evidence;
- optional `model_revision_or_api_version` provenance in both method paths;
- honest reporting that Stage A currently makes N edge-verifier calls and
  Stage B currently makes one direct-adjudication call.

This documentation reset does not begin AB-1B.

## Canonical Reading Order

Read only these active authority documents before method work:

1. `docs/method_spec_dpa.md`
2. `docs/stage_ab_protocol.md`
3. `docs/paper1_blueprint_zh.md`
4. `docs/repository_execution_contract.md`
5. `docs/coding_contract.md`
6. `docs/implementation_status.md`
7. `docs/upstream_integration.md`

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
  the same DPA core. It may begin only after Stage A/B evidence supports the
  structured authorization decomposition. It is not latent-memory learning.

## Safe Next Boundary

The next implementation stage, when explicitly requested later, is AB-1B:
internal development-case evaluator and replay-only runner. Do not start it
during documentation cleanup.

Still not started:

- AB-1B internal development-case evaluator;
- real provider integration;
- live API calls;
- official STALE or Memora evaluation;
- secondary end-to-end experimental execution;
- Stage C training.

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
