# Stage A/B: ReTrace-LLM and DirectJudge-LLM Implementation Plan

Status: draft, pending review and approval before implementation.
Branch: `method/retrace-llm-directjudge` off `main` HEAD `dcf121b`.
Authority: this plan governs Stage A/B implementation only. DPA core
specifications remain in `docs/refactor_plan_defeat_path.md`.

## 1. Scientific Objective

Test whether constrained local typed-edge prediction plus deterministic
DPA outperforms matched direct LLM adjudication for evidence-preserving
reversible belief revision.

The null hypothesis is that the structured DPA decomposition adds overhead
without improving authorization accuracy compared to asking the same model
to make the same decision directly. If Stage A does not outperform Stage B,
the decomposition does not justify its cost and Stage C is not warranted.

## 2. Stage A: ReTrace-LLM (Main Method)

ReTrace-LLM replaces all development-only heuristic/manual fixtures with
generic LLM-backed semantic components while keeping DPA deterministic.

### Components

1. **Generic Typed Belief Extraction** (`PromptTypedBeliefExtractor`)
   - Implements `TypedBeliefExtractor` protocol.
   - Input: `EvidenceNode`, `scope_id`.
   - Output: `list[BeliefNode]` with grounded `source_evidence_ids`.
   - LLM call via `CachedLLMClient` → `ModelCallTrace`.

2. **Generic Requirement/Condition Induction** (`PromptRequirementInducer`)
   - Implements `RequirementInducer` protocol.
   - Input: `BeliefNode`, `tuple[EvidenceNode, ...]`.
   - Output: `list[RequirementProposal]` (each with `ConditionNode` + `DependencyEdge(REQUIRES)`).
   - LLM call via `CachedLLMClient` → `ModelCallTrace`.

3. **Generic Evidence-Edge Prediction** (`PromptEvidenceEdgeVerifier`)
   - Implements `EvidenceEdgeVerifier` protocol.
   - Input: `new_evidence`, `candidate_belief`, `candidate_replacement_beliefs`,
     `candidate_conditions`, `temporal_context`.
   - Output: `list[EvidenceEdge]` (BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN).
   - LLM call via `CachedLLMClient` → `ModelCallTrace`.
   - `SUPERSEDES` edges require a real `replacement_belief_id` from
     `candidate_replacement_beliefs`.

4. **Existing Deterministic DPA and Authorized-Basis Pipeline**
   - `RevisionGate` → `DefeatPathAuthorizationAlgorithm` → `BasisBuilder`.
   - No LLM calls. Purely structural.

### Stage A Data Flow

```
EvidenceNode
  → PromptTypedBeliefExtractor       [LLM call]
  → PromptRequirementInducer          [LLM call per new belief]
  → ImpactCandidateRetriever          [LLM or embedding call]
  → PromptEvidenceEdgeVerifier        [LLM call per impacted belief]
  → RevisionGate                      [deterministic]
  → DefeatPathAuthorizationAlgorithm  [deterministic]
  → BasisBuilder                      [deterministic]
  → answer
```

## 3. Stage B: DirectJudge-LLM (Attribution Baseline)

DirectJudge-LLM is a **sibling method path**, not an `EvidenceEdgeVerifier`.

It uses the same model family/version and comparable evidence/context/call
budget as Stage A, but directly decides memory usability without DPA or
local edge restrictions.

### Why DirectJudge-LLM is Not an EvidenceEdgeVerifier

- `EvidenceEdgeVerifier` proposes typed local edges that DPA consumes. Its
  output is constrained to `list[EvidenceEdge]` and it never decides final
  authorization.
- DirectJudge-LLM makes a **final** usability judgment for each candidate
  belief given the query and evidence context. It does not produce edges,
  conditions, or defeat paths. It produces a direct `USABLE / NOT_USABLE /
  UNCERTAIN` verdict with a rationale.
- Placing DirectJudge inside the DPA pipeline would attribute its gains
  (or losses) to DPA structure when they are actually from the model.

### Stage B Data Flow

```
EvidenceNode
  → same extraction as Stage A (shared preprocessing)
  → same candidate retrieval as Stage A (shared view)
  → DirectJudge prompt: given query + evidence + candidate beliefs,
    decide which beliefs are currently usable [LLM call]
  → answer (using usable beliefs as basis)
```

## 4. Evaluation Tracks

### Primary: Controlled Authorization-Only Track

Stage A and Stage B receive the **same precomputed `SharedCandidateView`**,
which fixes evidence context, query, candidate beliefs, candidate replacement
beliefs, and candidate conditions. Neither method performs its own extraction
or retrieval in this track.

- **Stage A** predicts typed local edges from the view, then feeds them to
  deterministic DPA to authorize or exclude each candidate belief.
- **Stage B** directly predicts a `USABLE / NOT_USABLE / UNCERTAIN` verdict
  for each candidate belief in the same view.
- Primary metrics evaluate **authorization decisions**, not answer-generation
  text.

### Secondary: End-to-End Track

Compares full extraction → retrieval → authorization → answer pathways.
Reported separately because extraction and retrieval introduce additional
confounds beyond the authorization step.

### Primary Metrics

- authorization accuracy
- obsolete-memory misuse rate
- unsupported revision rate
- protected-belief preservation
- rollback recovery
- tokens, calls, latency

## 4a. Fairness Protocol

What must be held fixed between Stage A and Stage B so the comparison
attributes gains to structure rather than model strength:

| Dimension | Constraint |
|-----------|-----------|
| Model family & revision | Same `model_id` and `model_revision_or_api_version` |
| Frozen instances | Same `boundary_audit_dev.jsonl` / `boundary_audit_eval.jsonl` cases |
| Candidate-belief view | Same `SharedCandidateView` (primary controlled track) |
| Token/call budget | Comparable total tokens; Stage B may use fewer calls but same total budget ceiling |
| Answer generation | Same answer-generation pathway wherever attribution requires it |
| Temperature/seed | Same `temperature=0.0`, same `seed` if supported |
| Cache/replay | Both methods use `CachedLLMClient` with same replay-safe cache policy |

### Separate Reporting

When end-to-end extraction differs from controlled authorization-only
attribution, report both:
- **Controlled**: shared `SharedCandidateView`, compare only authorization step.
- **End-to-end**: full pipeline from evidence to answer.

## 5. No-Leakage Protocol

- Prompt/model/config development **only** on `boundary_audit_dev.jsonl`
  (20 cases) and hand-written unit tests.
- No tuning on STALE or Memora final evaluation examples.
- Before frozen evaluation, freeze and record:
  - prompt template text and `prompt_template_hash`;
  - `model_id`, `model_revision_or_api_version`;
  - retrieval settings (limits, scoring);
  - DPA settings (tie-break rules — already deterministic);
  - cache manifests (`JSONLCache` replay logs);
  - token/call budget ceilings.
- `registry/upstreams.lock.yaml` already pins reference commit SHAs.
- `configs/retrace_prompt.yaml` must be updated to reflect typed edge
  labels before frozen evaluation; legacy `CONDITION` / `SUPPORT` labels
  are retired.

## 6. Implementation Waves

### Wave AB-0: Contracts, Prompt Schema, Mock/Replay Tests

AB-0 does **not** add real provider SDK dependencies. It reuses
`BaseLLMProvider`, `MockLLMProvider`, `CachedLLMClient`, `JSONLCache`, and
`CostAccounting` unchanged. Real provider adapters are deferred until the
offline contracts and controlled comparison are approved.

The primary controlled A/B comparison uses a shared `SharedCandidateView`
and does not introduce method-specific LLM or embedding retrieval.

**Files to create:**
- `src/retracemem/methods/__init__.py`
- `src/retracemem/methods/contracts.py`
  (`SharedCandidateView`, `DirectUsabilityStatus`, `DirectUsabilityVerdict`,
  `ControlledMethodResult`)
- `src/retracemem/methods/directjudge.py`
  (DirectJudge-LLM sibling method, not an `EvidenceEdgeVerifier`)
- `src/retracemem/verifier/prompt_typed_belief_extractor.py`
  (implements `TypedBeliefExtractor` via `CachedLLMClient`)
- `src/retracemem/verifier/prompt_requirement_inducer.py`
  (implements `RequirementInducer` via `CachedLLMClient`)
- `src/retracemem/verifier/prompt_evidence_edge_verifier.py`
  (implements `EvidenceEdgeVerifier` via `CachedLLMClient`)
- `prompts/retrace_llm/belief_extraction_v0.txt`
- `prompts/retrace_llm/requirement_induction_v0.txt`
- `prompts/retrace_llm/evidence_edge_prediction_v0.txt`
- `prompts/directjudge/direct_usability_v0.txt`

**Tests to create:**
- `tests/method_contract/__init__.py`
- `tests/method_contract/test_shared_candidate_view.py`
- `tests/method_contract/test_prompt_typed_belief_extractor.py`
- `tests/method_contract/test_prompt_requirement_inducer.py`
- `tests/method_contract/test_prompt_evidence_edge_verifier.py`
- `tests/method_contract/test_directjudge.py`
- `tests/method_contract/test_controlled_ab_fairness.py`

**Forbidden modifications in AB-0:**
- `src/retracemem/schemas.py`
- `src/retracemem/memory/**`, `src/retracemem/tms/**`
- `src/retracemem/backends/**`, `src/retracemem/pipeline.py`
- `src/retracemem/retrieval/**`, `src/retracemem/adapters/**`
- `scripts/**`, `configs/**`, `pyproject.toml`, `reference/**`

**No file ownership conflicts with existing code.**

### Wave AB-0.5: Fairness and Replay Invariants (Hardening)

Locked requirements before AB-1 may begin:

1. **Full shared-view fairness.**
   DirectJudge-LLM in the primary controlled track receives the same evidence
   context, candidate beliefs, candidate replacement beliefs, and candidate
   conditions available to ReTrace-LLM. Its prompt must render all of these.

2. **Complete verdict coverage.**
   DirectJudge must output exactly one verdict for every candidate belief.
   Omissions, duplicates, and unknown belief ids are parser failures.

3. **Explicit scope.**
   `PromptRequirementInducer` must require explicit scope identity derived
   from `belief.metadata["scope_id"]` and may not use `"default"` or
   `"global"` fallback.

4. **Current-evidence supersession grounding.**
   A `SUPERSEDES` replacement must be both:
   - present among supplied `candidate_replacement_beliefs`; and
   - grounded in the current new evidence
     (`new_evidence.evidence_id in replacement.source_evidence_ids`).

5. **Deterministic ids.**
   LLM outputs provide semantic text and labels, not authoritative graph
   object identifiers. Graph ids used by Stage A prompt components must be
   computed deterministically from grounded inputs (evidence_id, scope_id,
   normalized semantic output, edge_type, target_id, prompt_version).

6. **Gate condition.**
   AB-1 remains blocked until AB-0.5 passes full tests offline.

### Wave AB-1: ReTrace-LLM Generic Semantic Components

**Files to implement (from AB-0 stubs):**
- `src/retracemem/verifier/prompt_typed_belief_extractor.py`
- `src/retracemem/verifier/prompt_requirement_inducer.py`
- `src/retracemem/verifier/prompt_evidence_edge_verifier.py`

**Files to create:**
- `src/retracemem/retrieval/prompt_impact_retriever.py`
  (implements `ImpactCandidateRetriever` with embedding or LLM relevance)
- `src/retracemem/retrieval/prompt_query_retriever.py`
  (implements `QueryBeliefRetriever` with embedding or LLM relevance)

**Tests to update:**
- `tests/method_contract/test_prompt_extractor_contract.py`
- `tests/method_contract/test_prompt_inducer_contract.py`
- `tests/method_contract/test_prompt_edge_verifier_contract.py`

### Wave AB-2: DirectJudge-LLM Matched Baseline

**Files to implement (from AB-0 stub):**
- `src/retracemem/methods/directjudge.py`

**Tests to update:**
- `tests/method_contract/test_directjudge_contract.py`

### Wave AB-3: Internal Dev Diagnostics and Matched-Cost Reporting

**Files to create:**
- `scripts/run_retrace_llm_dev.py`
  (runs ReTrace-LLM on `boundary_audit_dev.jsonl`)
- `scripts/run_directjudge_dev.py`
  (runs DirectJudge-LLM on same cases with matched budget)
- `scripts/run_matched_comparison.py`
  (side-by-side comparison report)

**Files to modify:**
- `src/retracemem/evaluation/records.py` — extend for method-trace fields.
- `src/retracemem/evaluation/cost_accounting.py` — add budget-ceiling checks.

**Tests to create:**
- `tests/test_matched_comparison.py`

### Wave AB-4: Frozen Benchmark Runners

**Files to create or modify:**
- `scripts/run_stale_retrace_llm.py`
- `scripts/run_stale_directjudge.py`
- `scripts/run_memora_retrace_llm.py`
- `scripts/run_memora_directjudge.py`

**Contamination checks:**
- `tests/contamination/test_no_eval_in_dev_prompts.py`

**Pre-run freeze checklist:**
- All prompts, model IDs, configs, and cache manifests are committed.
- `run_manifest.json` is emitted with provenance before each official run.

### Stage C (Deferred): ReTrace-Local

- Learned local typed-edge verifier using SFT/LoRA.
- Optional `torch` / `transformers` dependencies only.
- Same DPA core, different verifier.
- Begins only after Stage A/B establish that the structured DPA
  formulation has value.

## 7. Exact Proposed Files and Tests Per Wave

| Wave | New Files | Modified Files | New Tests |
|------|-----------|---------------|-----------|
| AB-0 | `verifier/prompt_typed_belief_extractor.py`, `verifier/prompt_requirement_inducer.py`, `verifier/prompt_evidence_edge_verifier.py`, `methods/directjudge.py`, `configs/retrace_prompt_typed.yaml`, `configs/directjudge_prompt.yaml` | `pyproject.toml` | `tests/method_contract/test_prompt_extractor_contract.py`, `tests/method_contract/test_prompt_inducer_contract.py`, `tests/method_contract/test_prompt_edge_verifier_contract.py`, `tests/method_contract/test_directjudge_contract.py` |
| AB-1 | `retrieval/prompt_impact_retriever.py`, `retrieval/prompt_query_retriever.py` | stubs from AB-0 | method_contract tests |
| AB-2 | — | `methods/directjudge.py` | `test_directjudge_contract.py` |
| AB-3 | `scripts/run_retrace_llm_dev.py`, `scripts/run_directjudge_dev.py`, `scripts/run_matched_comparison.py` | `evaluation/records.py`, `evaluation/cost_accounting.py` | `tests/test_matched_comparison.py` |
| AB-4 | `scripts/run_stale_retrace_llm.py`, `scripts/run_stale_directjudge.py`, `scripts/run_memora_retrace_llm.py`, `scripts/run_memora_directjudge.py`, `tests/contamination/test_no_eval_in_dev_prompts.py` | — | contamination tests |

## 8. Stop Conditions

- No API-backed method code should be implemented until this plan is
  reviewed and approved.
- No official STALE or Memora evaluation until prompt/config freeze.
- No Stage C work until Stage A/B validation is complete.

## 9. Claims Allowed Before Official Evaluation

- DPA core is deterministic and test-closed.
- Development fixtures pass all structural authorization scenarios.
- No accuracy, FAMA, or benchmark-score claims until frozen evaluation
  with committed manifests.

## Audit Findings Summary

### Subagent A: Provider, Cache, and Prompt Infrastructure

- **Reusable infrastructure**: `BaseLLMProvider` (abstract), `MockLLMProvider`
  (deterministic testing), `CachedLLMClient` (replay-safe JSONL cache),
  `JSONLCache` (append-only event logger), `CostAccounting` (token/call/latency
  tracking). All are ready for Stage A/B use.
- **Prompt templates**: `configs/retrace_prompt.yaml` exists but uses legacy
  flat-relation labels (`SUPPORT`, `CONDITION`). Must be replaced with typed
  edge labels for Stage A. `PromptRelationVerifier` in
  `verifier/prompt_verifier.py` is legacy and will not be reused; new
  `PromptEvidenceEdgeVerifier` replaces it.
- **Token/call accounting**: `ModelCallTrace` records all 13 cache-key fields
  plus `prompt_tokens`, `completion_tokens`, `latency_ms`. `CostAccounting`
  aggregates them. Both are ready for budget enforcement.
- **New files needed**: `PromptTypedBeliefExtractor`,
  `PromptRequirementInducer`, `PromptEvidenceEdgeVerifier` (all in
  `verifier/`), `DirectJudge` (in `methods/`), updated prompt configs. DPA
  core remains API-free.
- **Dependencies**: `pyproject.toml` currently has zero runtime dependencies.
  Stage A/B will need an optional `api` group for LLM provider SDKs.

### Subagent B: Typed Pipeline Extension-Point

- **Interfaces Stage A must implement**: `TypedBeliefExtractor.extract()`,
  `RequirementInducer.induce_requirements()`,
  `EvidenceEdgeVerifier.verify_edges()`. Signatures are stable and frozen.
- **Generic semantic extraction**: needed in addition to edge verification.
  The current `ManualTypedBeliefExtractor` is a dev-only fixture;
  `PromptTypedBeliefExtractor` must replace it for paper runs.
- **DirectJudge-LLM placement**: must sit in `methods/directjudge.py` as a
  sibling method path, not contaminating `verifier/` or DPA core.
- **Dev-only fixtures that must remain**: `ManualTypedBeliefExtractor`,
  `ManualRequirementInducer`, `ManualEvidenceEdgeVerifier`,
  `ManualImpactCandidateRetriever`, `ManualQueryBeliefRetriever`. All are
  forbidden for paper main-result runners.
- **`ReTracePipeline` and `ReTraceBackend`**: accept injected components via
  explicit constructor. Stage A uses `PromptTyped*` components; Stage B uses
  `DirectJudge` path. Both flow through the same `EvaluationRecord` output.

### Subagent C: Benchmark, Dev/Eval, and Contamination

- **Dev splits**: `boundary_audit_dev.jsonl` (20 cases) for prompt development.
  `boundary_audit_eval.jsonl` frozen — never inspected during tuning.
- **Frozen sets**: STALE and Memora data under `reference/` pinned by
  `registry/upstreams.lock.yaml`. Must remain untouched until config freeze.
- **Runner invocation**: existing `run_stale.py`, `run_memora.py` are
  retrieval-baseline smoke runners. Stage A/B will need new method-specific
  runners that inject `PromptTyped*` or `DirectJudge` components.
- **Contamination**: no `tests/contamination/` directory exists yet. Must be
  created in Wave AB-4 with at minimum an exact-copy check between dev
  prompts and eval instances.
- **Run manifest fields**: `ModelCallTrace` already records `model_id`,
  `model_revision_or_api_version`, `prompt_template_hash`, and
  `eligible_for_replay`. A top-level `run_manifest.json` should be emitted
  before each official evaluation with frozen config checksums.

### Subagent D: Matched Baseline and Paper-Claim

- **Stage A ReTrace-LLM**: generic semantic extraction → typed-edge prediction
  → deterministic DPA. The model proposes local edges; DPA decides.
- **Stage B DirectJudge-LLM**: same model family/version, same frozen
  instances, comparable token/call budget, but directly decides memory
  usability without DPA or local edge constraints.
- **Stage C ReTrace-Local**: deferred. Same DPA, different verifier (learned
  local classifier). Begins only after A/B validation.
- **Fixed between A and B**: model family, model revision, frozen evidence/query
  instances, candidate-belief view, temperature/seed, token budget ceiling,
  answer-generation pathway, cache/replay policy. This ensures gains are
  attributed to DPA structure, not model strength.
- **Why DirectJudge is a baseline, not a verifier**: it makes final usability
  decisions rather than proposing local typed edges. Placing it inside DPA
  would misattribute its performance to the structural decomposition.
- **Claims before evaluation**: only structural correctness of DPA core. No
  accuracy, FAMA, or benchmark-score claims until frozen evaluation with
  committed manifests.
