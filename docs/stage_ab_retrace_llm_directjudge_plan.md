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

## 4. Fairness Protocol

What must be held fixed between Stage A and Stage B so the comparison
attributes gains to structure rather than model strength:

| Dimension | Constraint |
|-----------|-----------|
| Model family & revision | Same `model_id` and `model_revision_or_api_version` |
| Frozen instances | Same `boundary_audit_dev.jsonl` / `boundary_audit_eval.jsonl` cases |
| Candidate-belief view | Same extraction output (shared preprocessing) or clearly justified shared protocol |
| Token/call budget | Comparable total tokens; Stage B may use fewer calls but same total budget ceiling |
| Answer generation | Same answer-generation pathway wherever attribution requires it |
| Temperature/seed | Same `temperature=0.0`, same `seed` if supported |
| Cache/replay | Both methods use `CachedLLMClient` with same replay-safe cache policy |

### Separate Reporting

When end-to-end extraction differs from controlled authorization-only
attribution, report both:
- **Controlled**: shared extraction, compare only authorization step.
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

### Wave AB-0: Contracts, Prompt Schema, Cache/Manifests, Dev Tests

**Files to create:**
- `src/retracemem/verifier/prompt_typed_belief_extractor.py`
  (stub implementing `TypedBeliefExtractor`, gated by `CachedLLMClient`)
- `src/retracemem/verifier/prompt_requirement_inducer.py`
  (stub implementing `RequirementInducer`, gated by `CachedLLMClient`)
- `src/retracemem/verifier/prompt_evidence_edge_verifier.py`
  (stub implementing `EvidenceEdgeVerifier`, gated by `CachedLLMClient`)
- `src/retracemem/methods/directjudge.py`
  (DirectJudge-LLM method path, not an `EvidenceEdgeVerifier`)
- `configs/retrace_prompt_typed.yaml`
  (typed edge labels: BLOCKS, RELEASES, SUPERSEDES, REAFFIRMS, UNCERTAIN)
- `configs/directjudge_prompt.yaml`
  (DirectJudge prompt config)

**Files to modify:**
- `pyproject.toml` — add optional `api` dependency group
  (e.g. `google-generativeai`, `openai`, `anthropic`).

**Tests to create:**
- `tests/method_contract/test_prompt_extractor_contract.py`
- `tests/method_contract/test_prompt_inducer_contract.py`
- `tests/method_contract/test_prompt_edge_verifier_contract.py`
- `tests/method_contract/test_directjudge_contract.py`

**No file ownership conflicts with existing code.**

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
