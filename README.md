# ReTrace

ReTrace is a research codebase for multi-agent / subagent shared-memory revision authorization in evolving memory.

The method core is:

```text
immutable EvidenceNode ledger
+ typed BeliefNode / ConditionNode graph
+ DependencyEdge(REQUIRES)
+ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
+ RevisionGate structural admission
+ deterministic Defeat-Path Authorization Algorithm
```

ReTrace is designed as a **pluggable authorization kernel** that controls which revisions submitted by multiple subagents are authorized to alter the shared usable memory basis.

## Implemented Core

- Deterministic DPA over admitted typed edges.
- Stage A `ReTrace-LLM`: local typed-edge proposal plus RevisionGate plus DPA.
- Stage B `DirectJudge-LLM`: direct shared-view adjudication baseline.
- Sole public entrypoint `authorize(...)` in the root namespace.
- Multi-agent shared-memory submission and commit interface.

## Offline Validation

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
.venv/bin/python -m pytest
```

## Integration Example

External subagents can submit memory revisions to the shared memory layer:

```python
from retracemem import (
    authorize,
    EvidenceProposalBatch,
)
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.multiagent.commit import commit_subagent_submission

# Call commit_subagent_submission(submission)
```

## Implemented State and Milestones

Established components:
    - Semantically reviewed-candidate generation workflow.
    - Strict evidence-grounding Stage C parser.
    - Live-smoke preflight boundary.

Not yet established:
    - Approved smoke examples, live prompt results, trained Stage C policy, frozen primary-test results, and official external-validation scores.

*Note: Any committed artifacts in the repository (such as stale run summaries) are development artifacts and do not represent final paper results.*

## Next Steps

The next steps for this repository are:
1. Complete repository cleanup and governance.
2. Regenerate the dev70 dataset evaluation with strict/canonicalized schemas.
3. Perform system analysis and ablation study.
4. Set up and evaluate `ReTrace-AdaptiveProposer` with API-ICL.


