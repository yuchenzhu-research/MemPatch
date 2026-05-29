# ReTrace Companion v3: Reproducible Integration, Audit, and Model-Handoff Contract

**Repository target:** `yuchenzhu-research/ReTrace`
**Audited integration snapshot:** `integration/retrace-v1-complete` @ `5e8d6e2d1a494d572d6d0fa929595bb198154390`
**Purpose:** engineering and model-handoff authority accompanying the Paper 1 blueprint v3. This document governs implementation truth, provenance, leakage avoidance, stage transitions, and safe task delegation. It does not replace the scientific blueprint.

---

## 1. Authority Order

A coding agent must read and reconcile current repository files before editing. After adoption of this v3 contract, the intended authority order is:

1. `AGENTS.md` - first-read current task boundary and reading order;
2. `docs/method_spec_dpa.md` - canonical typed runtime and deterministic DPA semantics;
3. `docs/stage_ab_protocol.md` - controlled Stage A/B protocol and metric boundary;
4. `docs/paper1_blueprint_zh.md` - scientific claims and experiment interpretation;
5. `docs/repository_execution_contract.md` - engineering/reproducibility/handoff rules;
6. `docs/coding_contract.md` - file/package/test/commit rules;
7. `docs/implementation_status.md` - factual current implementation status;
8. `docs/upstream_integration.md` - upstream clean-room and licensing rules;
9. `docs/stage_c_report.md` - go/no-go status only after corrected against this contract.

Legacy raw source materials must not remain active instruction authority after canonical docs are updated.

---

## 2. Non-Negotiable Method Identity

Paper 1 is:

```text
immutable evidence ledger
+ typed belief / condition / evidence-edge graph
+ deterministic Defeat-Path Authorization Algorithm (DPA)
```

Stage identities:

- **Stage A - `ReTrace-LLM`:** semantic model proposes local typed evidence edges; Gate admits structurally valid proposals; DPA computes final authorization.
- **Stage B - `DirectJudge-LLM`:** same-view-controlled direct-adjudication baseline; directly emits `USABLE / NOT_USABLE / UNCERTAIN`; not an edge verifier; no DPA.
- **Stage C - `ReTrace-Local`:** deferred learned local typed-edge verifier; must not start until A/B evidence supports it and development-safe human-audited labels exist.

Do not convert Paper 1 into latent memory, RL memory operations, learned consolidation, a new benchmark paper, a CUPMem fork, generic RAG, or an LLM that directly rewrites memory.

Canonical runtime vocabulary:

```text
DependencyEdge(REQUIRES)
EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
```

---

## 3. Critical Audit Finding at Snapshot `5e8d6e2`

The integration branch contains useful v1 scaffolding, but it is not yet scientifically validated v1. Active docs must not state that all phases are fully operational without qualification.

### 3.1 Implemented scaffolding that may be retained

- provider/cache/manifest extension code;
- end-to-end runner entrypoints;
- answer generator interface;
- adapter/wrapper entrypoints for STALE and Memora;
- Stage C defer/no-go document shell;
- existing validated AB-1B controlled evaluator foundation.

### 3.2 Blocking or high-priority corrections before live comparison

#### Provider correctness

`HTTPLLMProvider` currently routes Gemini-style chat completion requests to a URL that does not match Google's documented OpenAI-compatible REST path. The documented endpoint contains `/v1beta/openai/chat/completions`. Provider-specific credential selection must also not silently select a Gemini key for an OpenAI request or vice versa.

Required action:
- fix endpoint selection and provider-specific key lookup;
- add mocked HTTP contract tests for each supported provider route;
- run a capped dev-only live smoke only after tests pass and credentials are available;
- record manifests without secrets.

#### End-to-end fairness and component validity

Current end-to-end mock execution relies on manual extraction, manual requirement induction, manual edge outputs and manual retrieval. This is valid for deterministic integration testing only, not paper results.

The live runner uses overlap-based retrieval. Treat this as a development baseline until its scientific role, candidate recall/precision and impact on scope-expansion failure are measured. Do not call it the paper-facing retrieval method without evidence.

Required action:
- preserve mock mode as integration tests;
- clearly tag manual/overlap components as development-only or secondary-track baseline;
- ensure end-to-end A/B comparisons share extraction, induction, retrieval and answer generation and differ only in authorization;
- record per-component and per-method costs separately.

#### STALE adapter status

The current STALE runner uses a development-fixture pipeline, manually installs an old belief, may create a fallback mock dataset under the upstream reference tree, and can mock the official judge. This is an adapter invocation smoke test, not official ReTrace evaluation and not Stage A/B evidence.

Required action:
- rename and document mock mode as adapter smoke/dry-run;
- never write fallback generated inputs under `reference/`;
- output all generated artifacts under ignored ReTrace `outputs/` locations;
- connect actual Stage A and Stage B pipeline paths before any frozen official run;
- preserve unmodified official evaluator behavior for real evaluation.

#### Memora adapter status

The current Memora wrapper defaults to `ReTracePipeline.for_development_fixture()` and the runner invokes upstream tooling in a manner that may write generated results inside the upstream clone. This is adapter smoke only.

Required action:
- prevent tracked or untracked mutation of `reference/Memora` during ReTrace execution wherever feasible through copied/sandboxed input/output roots or wrapper-level output redirection;
- require a non-fixture Stage A/B pipeline for any reported evaluation;
- keep mock judge runs explicitly labelled smoke/dry-run;
- use FAMA official evaluation only after frozen configurations exist.

#### Stage C data leakage / false gold risk

A Stage A live trace is not a gold evidence graph. Official benchmark execution items must not be recycled into Stage C training labels.

Required action:
- revise `docs/stage_c_report.md` so that Stage C requires development-safe, human-audited typed-edge annotations;
- never describe self-generated admitted edges as gold;
- never train on official STALE/Memora scored examples or evaluator feedback.

---

## 4. Mandatory Immediate Scientific Gate Before Heavy Evaluation

The internal **Ambiguity-and-Scope Feasibility Gate** has been implemented, executed, and retained as a feasibility diagnostic. Before any future expensive official execution or Stage C planning, this gate's findings must be reviewed.

### 4.1 Purpose

Answer the central question early:

```text
Does Stage A reduce unsupported scope expansion and overconfident revision
relative to Stage B when both use the same configured base model?
```

### 4.2 Required internal development split

Create a versioned, clearly internal-only diagnostic set, initially 30-50 cases, with categories:

- clear supersession;
- clear prerequisite blocking;
- protected unrelated belief;
- temporary constraint versus persistent preference;
- current-state change versus historical fact;
- tentative intention / possible future change;
- insufficient evidence requiring `UNCERTAIN/UNRESOLVED`;
- multi-belief scope-expansion traps;
- stateful release only when prior blocker state actually reaches execution.

Each case must contain:

- evidence history and current evidence;
- candidate beliefs;
- candidate conditions and fixed `REQUIRES` anchors;
- expected fine-grained Stage A authorization;
- comparable Stage B target;
- protected belief annotations;
- ambiguity/abstention-required annotation;
- human-written rationale;
- source classification (`newly_authored_dev`, `STALE_generation_inspired_dev`, `Memora_update_inspired_dev`);
- checksum and version.

Do not call this a new benchmark in Paper 1. It is a development diagnostic split for falsifying the method hypothesis early.

### 4.3 Required metrics

For Stage A and Stage B separately:

- authorization/comparable-verdict accuracy;
- stale-blocking accuracy;
- protected-belief preservation;
- abstention accuracy;
- unsupported confident revision rate on ambiguity-labelled items;
- execution/parsing/provenance failure counts;
- observed calls, tokens, cache behavior and latency.

Report clearly: same semantic view; same configured base model when available; prompts/interfaces differ; not strict call-budget matched.

### 4.4 Go/no-go interpretation

- If A preserves more protected beliefs or abstains better without materially harming stale blocking, proceed to external pilot/frozen evaluation.
- If A and B are both near-perfect, strengthen the diagnostic set before drawing method conclusions.
- If both are poor, diagnose model/prompt/input representation before spending official evaluation cost.
- If B clearly outperforms A, mark a blocking scientific risk; do not start Stage C or conceal the negative outcome behind infrastructure work.

---

## 5. Correct External Benchmark Pathway

### 5.1 STALE / CUPMem

Role:
- STALE: nearest end-to-end benchmark for implicit invalidation, including SR, PR and IPA.
- CUPMem: released external competitor/baseline that must be compared through official code after verification.

Required sequence:

1. clean-room adapter smoke without official-result claims;
2. development-safe or generated pilot for pipeline debugging;
3. configuration freeze;
4. official Stage A and Stage B final-answer runs;
5. official evaluator invocation unchanged;
6. optional CUPMem official reproduction.

Prohibited:
- tuning prompts on official scored cases;
- using official judged outputs as Stage C labels;
- calling mock judge output an official result.

### 5.2 Memora

Role:
- long-horizon final-answer evaluation with FAMA and `memory_presence` / `forgetting_absence` components.

Required sequence:

1. wrapper/adapter smoke in isolated output roots;
2. non-fixture Stage A/B pipeline validation;
3. frozen provider/model/prompt/retrieval configuration;
4. official evaluator run and raw-output preservation.

Prohibited:
- mutation of upstream inputs or evaluator semantics;
- training or prompt tuning on official evaluation questions/feedback;
- claiming FAMA results from mocked evaluator behavior.

---

## 6. Provider, Cache, Manifest, and Secret Rules

All model calls must pass through the approved provider/cache/accounting layer. No module may directly instantiate an SDK or issue HTTP calls outside that boundary.

Every paper-relevant run must record:

```text
run_id
repository_commit_sha
method/stage/comparison_regime
input split version and checksum
upstream SHA/license when applicable
provider/model/revision
prompt template hashes
parser/schema versions
generation parameters
cache mode and cache manifest
per-instance traces/costs/errors
output checksums
```

Never store API keys or authorization headers in logs, manifests, test outputs, committed data or docs.

Live-call rule:
- require explicit `--live` or equivalent opt-in;
- enforce hard caps on calls and tokens;
- use internal development cases first;
- cache and replay the exact calls;
- do not begin official scored evaluation in the same task that debugs provider behavior.

---

## 7. Upstream Clean-Room Rules

Use `reference/` only as an ignored, read-only local clone location for upstream repositories. ReTrace scripts should not create fallback data, run artifacts, or result files inside upstream clones as part of ordinary execution.

Tracked records should preserve:

- repository URL;
- commit SHA and branch;
- license artifact and verified role;
- wrapper/adaptor entrypoints;
- patches if unavoidable;
- input/output checksums for paper-relevant runs.

Keep separate:

```text
outputs/upstream_raw/
outputs/normalized/
outputs/retrace_runs/
outputs/manifests/
outputs/reports/
```

Official evaluator logic must not be silently patched. If a wrapper is required, document it and retain raw official evaluator outputs.

---

## 8. Required Next Coding Packet

The next authorized coding packet should not be “live benchmark evaluation.” It is:

### Packet P0/P1 - Repair and Early Feasibility Test (Factual Update: Feasibility Gate is completed)

Factual status: The Ambiguity-and-Scope internal feasibility diagnostic has already been executed/retained. The current packet focuses on:

1. Safety mode gating (strict separation between replay, dev-live, and official-eval modes).
2. Canonical per-session ingestion (defaulting ingest_chunk_size = 1) and explicit approximate/non-canonical metadata tagging for chunk sizes > 1.
3. Focused tests for runner safety gating and chunk policy.

Only after this safety repair packet is complete and approved may a separate task authorize STALE/Memora frozen official execution.

---

## 9. Documentation Synchronization Requirements

After repository adoption of this v3 direction, synchronize active docs:

- `AGENTS.md`: replace “all phases fully operational” with precise scaffold/smoke/validated status and set P0/P1 as next boundary.
- `docs/implementation_status.md`: distinguish implemented interfaces, mock/dry-run validation, live validation, and official results.
- `docs/paper1_blueprint_zh.md`: update status and add ambiguity/scope feasibility gate.
- `docs/repository_execution_contract.md`: update completed/future boundary and leakage rules.
- `docs/stage_c_report.md`: replace self-generated “gold” requirement with human-audited development-safe annotation policy.
- `README.md`: do not claim official results or operational live provider until demonstrated.

---

## 10. Completion-Report Contract for Coding Models

Every coding-model completion report must state:

- starting branch/SHA and final branch/SHA;
- exact files changed;
- exact tests and runner commands executed;
- whether calls were mock, replay, dev-live, or frozen official;
- whether official benchmark data was accessed and for what purpose;
- whether any upstream clone was mutated;
- whether prompts/method logic were changed after viewing official results;
- actual A/B metrics and honest interpretation;
- whether Stage C remains deferred.

No completion report may use the words “official evaluation complete” for a mocked judge/dry-run or adapter smoke.

---

## 11. Final Handoff Principle

The v1 scaffold is valuable, but it is not yet evidence for the paper claim. The next priority is to fix boundary violations and run an early, real, discriminative A/B diagnostic test.

The project succeeds only if it can honestly answer:

> Under the same semantic input and base-model configuration, does local typed-edge proposal plus deterministic DPA reduce unsupported suppression and overconfident revision relative to direct adjudication, and does that advantage later survive frozen external evaluation?

Until that answer exists, preserve the method, preserve the evidence, preserve the evaluation boundary, and do not train Stage C.
