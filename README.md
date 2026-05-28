# ReTrace

ReTrace is the working codebase for ICLR 2027 Paper 1:

> Evidence-preserving reversible authorization for evolving agent memory.

ReTrace is not latent memory learning, RL memory-action training, or a generic
memory framework clone. It preserves original evidence and changes whether a
belief may govern current answers through typed, auditable defeat paths.

The method core is:

```text
immutable EvidenceNode ledger
→ BeliefNode / ConditionNode graph
→ DependencyEdge(REQUIRES)
→ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
→ RevisionGate structural admission
→ deterministic Defeat-Path Authorization Algorithm
```

## Current Status

The active branch is `integration/retrace-v1-complete`.

Completed through V1-Complete/AB-3:

- typed DPA execution spine;
- offline Stage A/B contracts, prompts, sibling DirectJudge path, and mock/replay tests;
- fairness and deterministic-grounding hardening;
- offline controlled attribution harness;
- auditability and comparison-protocol lock;
- replay-only internal evaluation;
- real provider adaptation, budget caps, and run manifest configuration;
- secondary end-to-end multi-step internal evaluation;
- official STALE and Memora evaluation adapter runners;
- Stage C go/no-go deferral report (`docs/stage_c_report.md`).

## Canonical Docs

- `AGENTS.md`: first-read instructions for coding models.
- `docs/method_spec_dpa.md`: technical authority for runtime semantics.
- `docs/stage_ab_protocol.md`: active Stage A/B protocol authority.
- `docs/paper1_blueprint_zh.md`: canonical Chinese scientific blueprint.
- `docs/repository_execution_contract.md`: reproducibility and handoff contract.
- `docs/coding_contract.md`: package-boundary and editing rules.
- `docs/implementation_status.md`: concise live repository status.
- `docs/stage_c_report.md`: Stage C go/no-go recommendation report.
- `docs/upstream_integration.md`: upstream roles and clean-room integration.

## How to Run

### Compile and Run Tests
```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest
```

### Run End-to-End Pipeline Dev Test
```bash
.venv/bin/python scripts/run_end_to_end_dev.py
```

### Run Official STALE Evaluation
```bash
# Offline Mock Evaluation
.venv/bin/python scripts/run_stale_official_eval.py --limit 1

# Live Evaluation (requires OPENAI_API_KEY)
.venv/bin/python scripts/run_stale_official_eval.py --live --limit 5
```

### Run Official Memora Evaluation
```bash
# Offline Mock Evaluation
.venv/bin/python scripts/run_memora_official_eval.py --limit 1

# Live Evaluation (requires OPENAI_API_KEY)
.venv/bin/python scripts/run_memora_official_eval.py --live --limit 3
```

