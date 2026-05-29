# Stage A/B Protocol

This is the active-stage authority for Stage A/B scientific and engineering
protocol.

## Completed Milestones

### AB-0

Offline Stage A/B contracts, versioned prompts, DirectJudge sibling path, and
mock/replay tests were implemented without real provider SDK dependencies.

### AB-0.5

Fairness and deterministic-grounding hardening were completed:

- DirectJudge consumes the full shared view;
- DirectJudge must output exactly one verdict per candidate belief;
- `PromptRequirementInducer` uses explicit scope identity;
- `SUPERSEDES` replacements must be grounded in current evidence;
- graph ids are deterministic from grounded inputs;
- `SharedCandidateView` validates uniqueness and key consistency.

### AB-1A

Offline controlled attribution harness was completed:

- `SharedCandidateView` includes `new_evidence`, fixed dependency anchors, and
  derived `view_fingerprint`;
- `ControlledReTraceLLM` runs fixed view -> edge verifier -> isolated typed
  graph -> `RevisionGate` -> DPA -> `ControlledMethodResult`;
- DirectJudge and ControlledReTrace share the same fingerprint in provenance;
- per-instance cost is reported by delta accounting.

### AB-1B

Offline internal development-case evaluator and replay-only runner is complete
with repaired semantics:

- internal development cases in `data/internal_dev/controlled_ab_cases.json`
  cover six explicit case types: direct supersession, prerequisite blocking,
  protected unrelated belief, uncertainty, releases smoke, and
  rejected proposal audit;
- cases deserialize into valid `SharedCandidateView` objects with deterministic
  `view_fingerprint` preserved through both methods;
- `src/retracemem/evaluation/controlled_ab.py` implements `load_cases`,
  `run_case`, `compute_metrics`, and `format_report`;
- replay/mock execution only via `MockLLMProvider`; no live API calls;
- observed cost accounting uses `calls.get("total", 0)` instead of
  `sum(calls.values())` to avoid double-counting; cost is captured even on
  method failure via `finally` blocks;
- total annotated belief decisions are computed independently of Stage A
  success; conservative accuracy policy counts failed executions against
  method correctness;
- rejected proposal audit case triggers `RevisionGate` rejection
  (`replacement_belief_id_only_valid_for_supersedes`) rather than parser
  failure; provenance records `admitted=false` and stable `gate_reason`;
- obsolete-memory misuse and protected-belief preservation are computed
  symmetrically for both Stage A and Stage B;
- rollback recovery is `NOT YET OPERATIONALIZED` because the current
  fixed-view controlled interface does not preload prior accepted
  evidence-edge history; the prior `release_rollback` case type is relabelled
  as `releases_smoke`;
- `parse_errors` is incremented only on actual parse/value errors, not
  merely declared;
- `Unsupported Revision Rate` is deliberately deferred and surfaced in the
  report as `NOT YET OPERATIONALIZED`;
- runner `scripts/run_controlled_ab_dev.py` prints prominent disclaimers,
  writes JSON-compatible per-instance results and aggregate summary to
  `outputs/controlled_ab_dev/` (gitignored), and never commits run artifacts;
- 29 offline tests in `tests/evaluation/test_controlled_ab_evaluator.py`.

AB-1B results are internal development protocol diagnostics only. They are
not official benchmark results, are not strict call-budget matched, and make
no claim that ReTrace outperforms DirectJudge.

### AB-1A.5

Offline auditability and comparison protocol lock is complete:

- `SharedCandidateView.new_evidence` is mandatory;
- `view_fingerprint` is derived with versioned canonical JSON and SHA-256 over
  first-class controlled-input fields; metadata is excluded by policy;
- duplicate evidence ids, mismatched current-evidence payloads,
  candidate/replacement overlap, repeated mapping keys, and conflicting
  condition payloads are rejected;
- `EdgePredictionBatch` preserves `model_call_trace_id`, including zero-edge
  invocations;
- `PromptEvidenceEdgeVerifier.verify_edges_with_trace()` returns traced
  batches and `verify_edges()` delegates for compatibility;
- `ControlledReTraceLLM` fails loudly on rejected fixed dependency anchors and
  records admitted anchors, edge proposal admission status, and gate reasons;
- DirectJudge prompt v1 explicitly identifies current/new evidence;
- both paths can record `model_revision_or_api_version`;
- the current protocol states Stage A N calls versus Stage B one call.

## Stage A: ReTrace-LLM

Stage A is the main method path.

In the primary controlled track it consumes a fixed `SharedCandidateView`:

```text
SharedCandidateView
→ PromptEvidenceEdgeVerifier.verify_edges_with_trace
→ isolated typed graph
→ RevisionGate
→ deterministic DPA
→ ControlledMethodResult
```

Stage A predicts local typed evidence edges. It does not directly emit final
usability verdicts. DPA computes final authorization.

Stage A v1 uses effect-triggered authorization revision. The candidate belief
is treated as previously evidence-supported; new evidence changes current
authorization only by producing an admitted direct local typed effect on the
belief, a supplied required condition, or a supplied grounded replacement.
Irrelevant or silent evidence must produce an empty edge set and preserve
authorization. `UNCERTAIN` is reserved for directly relevant but unresolved
evidence.

The primary controlled track does not run extraction, requirement induction,
retrieval, or answer generation.

The shared view is identical at the method-interface level, but prompt exposure
is intentionally method-specific: DirectJudge renders the whole view for one
direct adjudication call, while Stage A renders current evidence, one candidate
belief, replacement candidates, and candidate conditions per edge-verifier call.
This is why the protocol reports observed cost rather than claiming identical
prompt exposure.

## Stage B: DirectJudge-LLM

Stage B is a shared-view-controlled direct-adjudication baseline.

It consumes the same fixed semantic `SharedCandidateView` and directly outputs
one `USABLE`, `NOT_USABLE`, or `UNCERTAIN` verdict per candidate belief.

It is not an `EvidenceEdgeVerifier`, does not produce typed edges, and does not
use DPA.

## SharedCandidateView

The fixed view contains:

- instance and query identifiers;
- query text;
- ordered evidence context;
- current `new_evidence`;
- candidate beliefs;
- candidate replacement beliefs;
- candidate conditions by belief;
- fixed `DependencyEdge(REQUIRES)` anchors by belief;
- derived `view_fingerprint`.

The fingerprint locks the first-class semantic input view so Stage A and Stage
B provenance can be compared without relying on mutable metadata.

## DirectJudge Prompt v1 Correction

DirectJudge prompt v1 explicitly renders the current/new evidence identity,
session id, timestamp, source dataset, source pointer, and text. This prevents
the direct baseline from receiving an ambiguous evidence list where it cannot
tell which evidence is the active update.

## Call Cardinality

The current controlled interface is not strict call-budget matched:

- Per-belief Stage A reference execution makes N semantic edge-verifier calls,
  one per candidate belief.
- Stage B makes one direct-adjudication call over the complete
  `SharedCandidateView`.

Prompts and method interfaces differ by design.

## Stage A Execution Paths

### Controlled attribution reference path (`ControlledReTraceLLM`)

- One candidate belief per verifier call.
- Intended for attribution auditing and small regression cases.
- Call complexity: O(B) calls.
- Token complexity: approximately O(B × |E|) repeated prompt cost.
- Each call repeats the full `new_evidence.text`.

### Scalable batched authorization path (`BatchedControlledReTraceLLM`)

- One local candidate neighborhood per verifier call.
- Returns typed-edge proposals indexed by candidate belief id and/or supplied
  condition id.
- All proposed edges still pass through the same `RevisionGate`.
- DPA runs separately for every candidate belief, identical semantics.
- Does not change canonical edge semantics or final deterministic authorization.
- Must report batch size, calls, tokens, latency, and provenance.
- Call complexity: O(ceil(B/K)) calls for batch size K.
- Token complexity: approximately O(ceil(B/K) × |E_local| + |B|).

Stage B remains unchanged. Comparison must report observed compute rather than
strict matched budget.

## Allowed Comparison Claims

Allowed:

- Stage A and Stage B consume the same fixed semantic `SharedCandidateView` in
  the primary controlled track.
- The same model family, model id, provider, and model revision/API version may
  be configured and recorded.
- Per-instance calls, tokens, cache behavior, and latency are measured.
- The comparison isolates structured authorization versus direct adjudication
  under a fixed semantic input view.

Forbidden before future evidence exists:

- strict matched call budget;
- equal number of model calls;
- identical prompt exposure;
- completed budget-normalized analysis;
- completed official benchmark evaluation;
- superiority over DirectJudge, CUPMem, STALE baselines, Memora systems, or any
  external method.

## Current and Future Boundaries

The Ambiguity-and-Scope internal feasibility diagnostic is complete and retained.

The current next boundary is the scientific hardening of the STALE pathway and evaluation protocol.

The Memora oracle-conditioned 30-question pilot is retained only as an internal rejected-pilot/adapter-misalignment artifact. It must not be rerun and must not be expanded.

The following remain future or deferred only. They do not start automatically:

- official end-to-end Memora evaluation with FAMA scoring;
- frozen official STALE evaluation (which is classified as a secondary answer-level evaluation, while the primary structural-attribution proof remains the fixed SharedCandidateView comparison);
- Stage C learned local typed-edge verifier using the same DPA core.

No official STALE/Memora evaluation, full benchmark run, provider-generalization
project, or Stage C work is part of the immediate feasibility packet.
