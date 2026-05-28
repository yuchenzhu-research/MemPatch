# Implementation Status

Last updated: 2026-05-28.

This document is the concise live repository status. It is not a scientific
storytelling document.

## Repository State

- Repository: `yuchenzhu-research/ReTrace`
- Active branch: `method/retrace-llm-directjudge`
- AB-1B implementation commit:
  `aa796f409fbdc8c9edbc48d3ac003b2f4b0baf7d`
- AB-1B.1 repair: commit `639a75622d15335568d0862a578f68d5750f41ad`
- AB-1B.2 cleanup: current local HEAD (to be updated after commit).

## Completed

- typed DPA execution spine on `main`;
- AB-0 offline Stage A/B contracts, versioned prompts, DirectJudge sibling path,
  and mock/replay tests;
- AB-0.5 fairness and deterministic-grounding hardening;
- AB-1A offline controlled attribution harness;
- AB-1A.5 offline auditability and comparison protocol lock;
- AB-1B offline internal development-case evaluator and replay-only runner
  (repaired in AB-1B.1).

AB-1B repaired behavior includes:

- 6 internal development controlled authorization cases covering
  direct supersession, prerequisite blocking, protected unrelated belief,
  uncertainty, releases smoke, and rejected proposal audit;
- case deserialization into valid `SharedCandidateView` objects with
  deterministic `view_fingerprint`;
- replay/mock execution of both Stage A (`ControlledReTraceLLM`) and
  Stage B (`DirectJudgeLLM`) using `MockLLMProvider` — no live API calls;
- observed cost uses `calls.get("total", 0)` not `sum(calls.values())`;
  cost is captured even on method failure;
- total annotated belief decisions computed independently of Stage A success;
  conservative accuracy denominators;
- rejected proposal audit triggers RevisionGate rejection, not parser failure;
  provenance records `admitted=false` and stable `gate_reason`;
- obsolete-memory misuse and protected-belief preservation computed
  symmetrically for Stage A and Stage B;
- rollback recovery NOT YET OPERATIONALIZED (fixed-view interface does not
  preload prior accepted evidence-edge history);
- `parse_errors` incremented only on actual parse failures;
- `Unsupported Revision Rate` deliberately deferred;
- replay-only runner with prominent disclaimers;
- JSON-compatible per-instance results and aggregate summary;
- output written to `outputs/controlled_ab_dev/` (gitignored);
- 27 tests in `tests/evaluation/test_controlled_ab_evaluator.py`.

## Not Started

Do not treat any of these as implemented:

- AB-1C live provider adapter;
- real provider integration;
- live API calls;
- official STALE or Memora evaluation;
- secondary end-to-end experimental execution;
- Stage C training;
- learned local typed-edge verifier results.

## Verification From AB-1B.1 Repair

- Compileall: passed with
  `env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts`.
- Pytest: `255 passed` with `PYTHONPATH=. .venv/bin/pytest`.
- Replay-only runner: passed with
  `PYTHONPATH=. .venv/bin/python scripts/run_controlled_ab_dev.py`.
- Live API calls: none.
- Official STALE/Memora evaluation: none.
- DPA, RevisionGate, schemas, providers, cache, retrieval, backend, pipeline:
  not modified.
