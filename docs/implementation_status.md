# Implementation Status

Last updated: 2026-05-28.

This document is the concise live repository status. It is not a scientific
storytelling document.

## Repository State

- Repository: `yuchenzhu-research/ReTrace`
- Active branch checked before this documentation reset:
  `method/retrace-llm-directjudge`
- Local HEAD checked before editing:
  `20b48b919106959f6d66e882b540525c41046983`
- Remote branch checked before editing:
  `origin/method/retrace-llm-directjudge` at
  `20b48b919106959f6d66e882b540525c41046983`
- Verified starting commit message:
  `Implement Stage AB-1A.5 auditability and protocol lock`
- Working tree before editing: clean.

## Completed

Completed before this documentation reset:

- typed DPA execution spine on `main`;
- AB-0 offline Stage A/B contracts, versioned prompts, DirectJudge sibling path,
  and mock/replay tests;
- AB-0.5 fairness and deterministic-grounding hardening;
- AB-1A offline controlled attribution harness;
- AB-1A.5 offline auditability and comparison protocol lock.

AB-1A.5 completed behavior includes:

- `SharedCandidateView.new_evidence` is mandatory;
- `view_fingerprint` is derived and hashes first-class controlled-input fields,
  with metadata excluded;
- traced Stage A edge-verifier output preserves `model_call_trace_id`, including
  zero-edge invocations;
- fixed supplied `DependencyEdge` anchor rejection fails loudly;
- predicted evidence-edge admitted/rejected decisions and gate reasons are
  retained in provenance;
- DirectJudge prompt v1 explicitly identifies current/new evidence;
- `model_revision_or_api_version` can be recorded in both method paths;
- the protocol truthfully reports Stage A N calls versus Stage B one call in
  the current controlled interface.

## Not Started

Do not treat any of these as implemented:

- AB-1B internal development-case evaluator or replay-only runner;
- real provider integration;
- live API calls;
- official STALE or Memora evaluation;
- secondary end-to-end experimental execution;
- Stage C training;
- learned local typed-edge verifier results.

## Documentation Reset Scope

This task changes documentation only. It does not modify runtime source code,
tests, prompts, scripts, configs, datasets, registry files, provider/cache code,
retrieval, backend, pipeline, `RevisionGate`, or DPA.

## Verification From This Task

Recorded from this documentation-reset task:

- Compileall: passed with
  `env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts`.
- Pytest: `228 passed in 0.67s` with `.venv/bin/python -m pytest`.
- Live API calls: none.
- Official STALE/Memora evaluation: none.
