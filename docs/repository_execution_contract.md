# Repository Execution Contract

This is the English engineering, reproducibility, upstream-integration, and
handoff contract for ReTrace.

## Authority Order

When instructions conflict, follow this order:

1. `AGENTS.md`
2. `docs/method_spec_dpa.md`
3. `docs/stage_ab_protocol.md`
4. `docs/paper1_blueprint_zh.md`
5. this document
6. `docs/upstream_integration.md`
7. `docs/coding_contract.md`
8. `docs/implementation_status.md`

Legacy planning documents and old source-material files are not active
authority. Do not restore them as required reading.

## Stage Interpretation

Stages AB-1A.5 and AB-1B are complete. AB-1C is the next not-yet-started
future boundary.

Stage identities:

- Stage A, `ReTrace-LLM`: semantic model proposes local typed evidence edges;
  deterministic DPA computes authorization.
- Stage B, `DirectJudge-LLM`: shared-view-controlled direct-adjudication
  baseline; directly predicts candidate-belief usability; not an
  `EvidenceEdgeVerifier`; no DPA.
- Stage C, `ReTrace-Local`: deferred learned local typed-edge verifier using
  the same DPA core; not latent-memory learning.

Do not describe Stage B as strict call-budget matched. Use:

```text
shared-view-controlled direct-adjudication baseline
```

or explicitly:

```text
same fixed semantic input view and model-configurable comparison, with
observed calls/tokens/latency reported; not strict call-budget matched.
```

## Clean-Room Upstream Policy

External repositories are references, baselines, or evaluators. They cannot
redefine the ReTrace method.

`reference/` is the canonical ignored local-upstream clone location. Preserve
that convention unless a future task explicitly authorizes migration.

For every upstream result or wrapper:

- verify repository URL, commit SHA, branch, and license locally;
- record setup status and entrypoints;
- keep raw upstream outputs separate from ReTrace outputs;
- preserve official evaluator outputs when used;
- record any patch with upstream commit SHA and reason;
- prefer wrappers/adapters over copied code;
- never use official evaluation examples, gold labels, or evaluator outputs as
  prompt-development or training data unless an approved, leakage-audited study
  explicitly allows it.

Detailed upstream roles are in `docs/upstream_integration.md`.

Existing `docs/upstreams/*.md` files are read-only upstream reconnaissance
notes. They are not active method authority and are not part of the required
model-reading order.

## Controlled and End-to-End Separation

Primary controlled authorization track:

- Stage A and Stage B consume the same fixed `SharedCandidateView`;
- extraction, induction, retrieval, and answer generation do not differ inside
  the primary attribution comparison;
- calls, tokens, cache behavior, and latency are measured per instance;
- cost is reported honestly as observed.

Secondary end-to-end track:

- may include extraction, requirement induction, retrieval, authorized-basis
  construction, and answer generation;
- must be reported separately because these components add confounds.

## Run Manifest Requirements

Any paper-relevant run must preserve at least:

- run id;
- method name and stage;
- repository commit SHA;
- input split id and checksum;
- upstream repo SHA when external assets are involved;
- model provider, model id, and model revision/API version when available;
- model configuration;
- prompt template hashes;
- parser and response-schema versions;
- cache mode and cache manifest;
- per-instance call counts;
- per-instance token counts;
- per-instance latency;
- output checksum;
- errors, retries, parse failures, and gate rejections.

For controlled Stage A/B comparisons, also preserve:

- `view_fingerprint`;
- comparison regime;
- fine-grained authorization or verdict status;
- admitted and rejected proposal traces where applicable.

## Provider Boundary

All future live model calls must go through the provider/cache/accounting
abstraction in `src/retracemem/providers`, `src/retracemem/cache`, and related
accounting modules.

Forbidden:

```text
random module imports an SDK client and calls a live model directly
```

Required:

- prompt/template version recorded;
- parser/schema version recorded;
- cost and latency measured;
- retry/error outcome recorded;
- cache/replay behavior auditable;
- no live call inside DPA, RevisionGate, memory store, retrieval-free tests, or
  deterministic core logic.

## Leakage and Evaluation Separation

Use only approved development diagnostics for prompt design, parser repair,
controlled harness debugging, and provider bring-up.

Before any frozen official evaluation, lock and checksum:

- prompts and hashes;
- model provider/model revision and generation parameters;
- parser/schema versions;
- candidate-view or retrieval construction protocol;
- DPA configuration;
- cache policy;
- cost-reporting policy;
- evaluator/upstream repository SHAs;
- output directory and manifest format.

Do not inspect, tune against, rewrite prompts from, or synthesize training
examples from final STALE/Memora evaluation instances before configuration is
frozen.

## Safe Future Packets

These are future task shapes, not active work.

### AB-1B (completed)

Internal development-case evaluator and replay-only runner.
Completed with repaired semantics in AB-1B.1.

### AB-1C

One live provider adapter and tiny approved dev-only calls through the provider
boundary, with hard caps and full cache/provenance capture.

### AB-2

Secondary end-to-end internal pipeline, reported separately from primary
controlled attribution.

### AB-3

Frozen official evaluation after method/configuration freeze.

### Stage C

Learned local typed-edge verifier using the same DPA core, only after Stage A/B
supports the structured decomposition.

## Completion Report Requirements

Every coding-model completion report should state:

- starting branch and commit SHA;
- final branch and commit SHA if committed;
- changed files;
- forbidden file areas confirmed untouched;
- exact validation commands and pass/fail results;
- whether live API calls occurred;
- whether official evaluation data was touched or executed;
- whether DPA/core semantics changed;
- whether permitted paper claims changed;
- next safe task boundary.

## Parking Lot Policy

Ideas that would widen scope must not be silently implemented. Park them only
if explicitly requested:

- latent state memory;
- consolidation learning;
- RL memory operations;
- new benchmark construction;
- large new storage/database dependencies;
- batched or budget-normalized alternative A/B experiment variants not yet
  approved.
