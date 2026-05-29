# ReTrace

ReTrace is a research codebase for evidence-preserving reversible authorization in evolving agent memory.

The method core is:

```text
immutable EvidenceNode ledger
+ typed BeliefNode / ConditionNode graph
+ DependencyEdge(REQUIRES)
+ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
+ RevisionGate structural admission
+ deterministic Defeat-Path Authorization Algorithm
```

ReTrace is designed as a **pluggable authorization kernel** that can be integrated with any external memory database or agent runtime. It does not perform memory storage, retrieval, or agent orchestration itself; instead, it consumes candidate views and deterministically adjudicates belief eligibility.

## Implemented Core

- Deterministic DPA over admitted typed edges.
- Stage A `ReTrace-LLM`: local typed-edge proposal plus RevisionGate plus DPA.
- Stage B `DirectJudge-LLM`: direct shared-view adjudication baseline.
- Sole public entrypoint `authorize(...)` in the root namespace.

## Offline Validation

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
.venv/bin/python -m pytest
```

## Integration Example

External memory producers can easily request authorization from ReTrace using the public kernel function:

```python
from retracemem import (
    authorize,
    EvidenceProposalBatch,
)
from retracemem.methods.contracts import SharedCandidateView

# 1. Construct the shared view
view = SharedCandidateView(...)

# 2. Package proposed typed edges
proposals = (
    EvidenceProposalBatch(
        edges=tuple(predicted_edges),
        model_call_trace_id="call_trace_uuid",
    ),
)

# 3. Request authorization
result = authorize(view, proposals)

print(result.authorized_belief_ids)
print(result.excluded_belief_ids)
print(result.trace)
```

## Non-Claims

The repository currently does not establish:

- Stage A superiority over Stage B;
- official STALE or Memora scores;
- paper-facing retrieval validity;
- Stage C training labels;
- benchmark-general live-provider performance.
