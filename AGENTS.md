# MemPatch Agent Instructions

Read `AGENTS.md` then `README.md` before method work. Dynamic branch/HEAD status lives only in `README.md`.

## Unified Paper

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI: an LLM agent must rapidly integrate new evidence with prior memory states — superseding, blocking, releasing, marking uncertain, reaffirming, or leaving unchanged affected memories — and produce the correct current `memory_state`.

One paper, one story:

- MemPatch-Bench defines scenarios, `hidden_gold`, and the `response` evaluation interface.
- MemPatch scaffold learns benchmark-compatible revision responses.
- DPA (`authorize(...)`) authorizes proposals deterministically; benchmark scores the outcome.
- Benchmark-grounded feedback supplies training signal from `memory_state` / evidence / diagnostic metrics.

Do not frame MemPatch-Bench and the scaffold as two papers or two tracks.

## Benchmark response interface

Paper-facing fields:

- `response.decision`
- `response.memory_state` (scored vs `hidden_gold.expected_memory_state`)
- `response.evidence_event_ids`
- `response.failure_diagnosis`

Canonical `hidden_gold`: `expected_decision`, `expected_answer`, `expected_memory_state`, `expected_failure_diagnosis`, `expected_evidence_event_ids`, `counterevidence_event_ids`, `rubric`, `decision_aliases`, `stale_or_wrong_answers`.

## Scaffold implementation roles

Not paper contributions — internal pipeline only:

1. **Scenario View Builder** (`src/retrace_learn/runtime/graph_extractor.py`) — event_trace → revision view
2. **Revision Response Policy** (`src/retrace_learn/runtime/learned_proposer.py`) — revision view → benchmark-compatible response
3. **Benchmark-grounded feedback** (`src/retrace_learn/runtime/reward.py`) — metrics → training signal

DPA maps internal belief statuses to benchmark `memory_state` labels. ReTrace-Engine (Parser + RevisionGate + DPA + audit via `authorize(...)`) is an internal commit path, not a standalone module.

## Configuration baselines

- `ReTrace-Prompt` — API baseline proposes typed actions over a fixed revision view, routes through ReTrace-Engine
- `DirectJudge-API` — API baseline predicts final usability status, bypasses ReTrace-Engine
- `retrace_learn` config — full trainable scaffold (View Builder → Response Policy → feedback)

## Public API

Sole authorization entrypoint:

```python
def authorize(
    view: SharedCandidateView,
    proposal_batches: tuple[EvidenceProposalBatch, ...],
    *,
    audit_metadata: dict[str, Any] | None = None,
) -> AuthorizationResult:
    ...
```

Neither DPA nor RevisionGate should be invoked directly. The model proposes; DPA authorizes.

Multi-agent wrappers: `commit_subagent_submission(...)`, `commit_submission_sequence(...)`.

## Canonical typed edges

- `DependencyEdge(REQUIRES)`: belief → condition
- `EvidenceEdge(BLOCKS|RELEASES|SUPERSEDES|REAFFIRMS|UNCERTAIN)`

DPA precedence: `SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED`

## Training boundary

Learned policy sees only method-visible inputs (revision view, new evidence, candidate beliefs/conditions/anchors). Never sees `hidden_gold` or evaluator final statuses during live inference.

Only human-approved reviewed cases for live training export. Scripts must not auto-promote pending cases.

## Experiment tiers

E0 — oracle/replay kernel validation  
E1 — fixed-candidate revision evaluation (primary comparison)  
E2 — learned Revision Response Policy training/eval  
E3 — closed-loop multi-agent workflow  
E4 — STALE/CUPMem external validation (isolated under `experiments/`)

## Do not drift

- No generic orchestration framework, agent voting, or duplicate `authorize` kernels
- No DPA semantic changes without demonstrated deterministic bugs
- No STALE gold fields in method inputs; no official scored cases for prompt tuning
- No generic RAG / Mem0 / Graphiti clone

## Clean worktree

Do not commit: `artifacts/`, `analysis/`, `local/`, caches, checkpoints, predictions, API keys, generated corpora.

Before commit:
```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache .pycache_compile
```

Preserve `src/retracemem/schemas.py` dataclass contracts.

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```
