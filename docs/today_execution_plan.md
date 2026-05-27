# Today Execution Plan

Date: 2026-05-27

Objective: finish the first research-code version today. "Finish" means a
complete local smoke loop that demonstrates ReTrace's intended logic, not full
paper-scale benchmark results.

## Final State By End Of Day

The repository should support these commands:

```bash
python3 scripts/run_boundary_audit.py --method retrieval_baseline
python3 scripts/run_boundary_audit.py --method retrace_heuristic
python3 scripts/run_stale.py --limit 3 --method retrieval_baseline
python3 scripts/run_memora.py --limit 3 --method retrieval_baseline
env PYTHONPYCACHEPREFIX=.pycache_compile python3 -m compileall -q retracemem tests scripts
```

The first two commands are the most important. STALE and Memora only need smoke
coverage today.

## Milestone 0: Freeze The Contracts

Status: mostly complete.

Do not change these unless absolutely necessary:

- `retracemem/schemas.py`
- `retracemem/backends/base.py`
- `retracemem/evaluation/jsonl.py`
- `retracemem/evaluation/records.py`

Acceptance:

- Existing smoke checks still pass.
- All later modules use `EvaluationRecord` for output.

## Milestone 1: BoundaryAudit Data

Create:

```text
data/boundary_audit/minimal.jsonl
```

Write 20 cases with these buckets:

- 4 `SUPERSEDE`
- 4 `BLOCK`
- 4 `CONDITION`
- 4 `NONE`
- 4 `UNCERTAIN`

Required fields:

```json
{
  "case_id": "ba_001",
  "bucket": "BLOCK",
  "old_belief": "The user commutes by bicycle.",
  "new_evidence": "The user broke their leg and will be in a cast for six weeks.",
  "query": "How should the user commute tomorrow?",
  "expected_relation": "BLOCK",
  "expected_authorized": false,
  "condition": "cycling ability",
  "protected_beliefs": [
    "The user likes Thai food."
  ],
  "notes": "Broken leg should affect cycling, not food preference."
}
```

Acceptance:

- File is valid JSONL.
- Every case has a unique `case_id`.
- Buckets are balanced enough for smoke reporting.
- No external data needed.

## Milestone 2: Heuristic Relation Verifier

Create:

```text
retracemem/verifier/heuristic_verifier.py
tests/test_heuristic_verifier.py
```

Class:

```python
class HeuristicRelationVerifier:
    def verify(self, new_evidence, candidate_belief, context=None) -> RelationPrediction:
        ...
```

Minimum rules:

- Address/location replacement → `SUPERSEDE`
- Injury/cast/recovery + mobility/commute/sport belief → `BLOCK`
- "only if", "after", "when cleared", recovery condition → `CONDITION`
- "not sure", "unclear", "maybe", "might have changed" → `UNCERTAIN`
- obvious unrelated topics → `NONE`
- repeated support language → `SUPPORT`

Implementation constraints:

- Standard library only.
- Deterministic.
- Fail closed to `UNCERTAIN` when evidence is too ambiguous.
- Include `rationale`, `span`, and `confidence`.

Acceptance:

- Tests cover all six primary labels except `REQUIRED_BY`.
- Invalid or empty inputs do not crash.
- Output is always a `RelationPrediction`.

## Milestone 3: ReTrace Pipeline

Create:

```text
retracemem/pipeline.py
tests/test_pipeline.py
```

Class:

```python
class ReTracePipeline:
    def __init__(self, verifier=None) -> None: ...
    def reset_user(self, user_id: str) -> None: ...
    def add_belief(self, user_id: str, belief: Belief) -> None: ...
    def ingest_evidence(self, user_id: str, evidence: EpisodicEvidence) -> list[RelationPrediction]: ...
    def authorized_basis(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, str]]: ...
    def answer(self, user_id: str, query: str, limit: int = 10) -> EvaluationRecord: ...
```

First-version behavior:

- User has an `EpisodeLedger` and `BeliefStore`.
- `add_belief` seeds prior beliefs for diagnostic cases.
- `ingest_evidence` appends evidence and verifies it against all existing
  beliefs.
- Gate-accepted relations are stored.
- `authorized_basis` delegates to `BasisBuilder`.
- `answer` returns deterministic shell text and an `EvaluationRecord`.

Do not implement embeddings, graph search, async, or API calls.

Acceptance:

- Broken-leg evidence blocks bike commute but preserves food preference.
- Supersede old address blocks only old address.
- Uncertain evidence removes default authorization without adding replacement.
- Output can be written by `write_jsonl`.

## Milestone 4: BoundaryAudit Runner

Replace placeholder:

```text
scripts/run_boundary_audit.py
```

CLI:

```bash
python3 scripts/run_boundary_audit.py \
  --cases data/boundary_audit/minimal.jsonl \
  --method retrace_heuristic \
  --output outputs/boundary_audit/retrace_heuristic.jsonl
```

Methods:

- `retrieval_baseline`
- `retrace_heuristic`

For `retrieval_baseline`:

- store old belief and new evidence as raw text;
- answer with deterministic retrieval shell.

For `retrace_heuristic`:

- seed `old_belief`;
- optionally seed protected beliefs;
- ingest `new_evidence`;
- build authorized basis;
- compare observed authorization to expected authorization.

Summary printed to stdout:

```text
cases_total
relation_correct
authorization_correct
protected_beliefs_preserved
unsupported_revision_count
output_path
```

Acceptance:

- Runner exits 0.
- JSONL output exists.
- Summary metrics are printed.
- No API key needed.

## Milestone 5: STALE Smoke Runner

Improve:

```text
scripts/run_stale.py
```

CLI:

```bash
python3 scripts/run_stale.py --reference-root reference/STALE --limit 3 --method retrieval_baseline
```

Behavior:

- discover `*_MAIN.json`;
- load first file;
- take first `--limit` samples;
- ingest sessions into `RetrievalBaselineBackend`;
- answer dim1/dim2/dim3 queries;
- write JSONL under `outputs/stale/`.

Acceptance:

- If no MAIN files exist, print a clear message and exit 0.
- If files exist, emit JSONL.
- No official judge required today.

## Milestone 6: Memora Smoke Runner

Improve:

```text
scripts/run_memora.py
```

CLI:

```bash
python3 scripts/run_memora.py --reference-root reference/Memora --limit 3 --method retrieval_baseline
```

Behavior:

- discover persona roots;
- use the first persona root by default;
- load sessions and questions;
- ingest chronological sessions;
- answer first `--limit` questions;
- write JSONL under `outputs/memora/`.

Acceptance:

- If no data roots exist, print a clear message and exit 0.
- If data roots exist, emit JSONL.
- No FAMA judge required today.

## Milestone 7: Verification

Run:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile python3 -m compileall -q retracemem tests scripts
```

Run direct smoke tests if pytest is unavailable.

If pytest exists:

```bash
python3 -m pytest -q
```

Acceptance:

- compileall passes.
- runner commands do not crash.
- no generated outputs are committed.

## Milestone 8: Commit And Push

Use these commits:

```text
Add first-version execution docs
Add BoundaryAudit diagnostic cases
Add heuristic ReTrace verifier
Add ReTrace smoke pipeline
Add BoundaryAudit runner
Add benchmark smoke runners
```

Push:

```bash
git push origin main
```

## Work Allocation For Future Models

### Gemini-3.5 Flash

Use for small deterministic edits:

- add JSONL cases;
- add simple tests;
- update docs;
- improve CLI argument parsing;
- fix compile errors.

Do not ask it to redesign the architecture.

### Opus 4.7

Use for larger integration:

- implement `ReTracePipeline`;
- improve heuristic verifier;
- wire benchmark smoke runners;
- review consistency across modules.

Do not let it add unrelated memory frameworks or heavy dependencies.

## Stop Conditions

Stop adding features when:

- BoundaryAudit runs end to end;
- STALE and Memora smoke runners do not crash;
- JSONL output is unified;
- compileall passes;
- docs explain what remains for full experiments.

Anything beyond that belongs to the next iteration.

