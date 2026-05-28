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

AB-1B completed behavior includes:

- 6 internal development controlled authorization cases covering
  direct supersession, prerequisite blocking, protected unrelated belief,
  uncertainty, release/rollback recovery, and rejected proposal audit;
- case deserialization into valid `SharedCandidateView` objects with
  deterministic `view_fingerprint`;
- replay/mock execution of both Stage A (`ControlledReTraceLLM`) and
  Stage B (`DirectJudgeLLM`) using `MockLLMProvider` — no live API calls;
- controlled A/B metric computation: authorization accuracy, obsolete-misuse
  count/rate, protected-belief preservation, rollback recovery, fine-grained
  status breakdown, verdict breakdown, observed cost (calls/tokens/cache/latency
  reported separately for each stage);
- `Unsupported Revision Rate` deliberately deferred — requires explicit
  annotation of valid defeat-path structure and unambiguous denominator;
- parser-level and execution errors surfaced in results, not silently dropped;
- replay-only runner (`scripts/run_controlled_ab_dev.py`) with prominent
  disclaimers: internal protocol check only, not an official benchmark,
  not strict call-budget matched, no claim that ReTrace outperforms DirectJudge;
- JSON-compatible per-instance results and aggregate summary;
- output written to `outputs/controlled_ab_dev/` (gitignored);
- 25 new tests in `tests/evaluation/test_controlled_ab_evaluator.py`.

## Not Started

Do not treat any of these as implemented:

- real provider integration;
- live API calls;
- official STALE or Memora evaluation;
- secondary end-to-end experimental execution;
- Stage C training;
- learned local typed-edge verifier results.

## Verification From AB-1B

- Compileall: passed with
  `env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts`.
- Pytest: `253 passed` with `.venv/bin/python -m pytest`.
- Replay-only runner: passed with
  `.venv/bin/python scripts/run_controlled_ab_dev.py`.
- Live API calls: none.
- Official STALE/Memora evaluation: none.
- DPA, RevisionGate, schemas, providers, cache, retrieval, backend, pipeline:
  not modified.
