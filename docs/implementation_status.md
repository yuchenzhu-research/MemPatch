# Implementation Status

Last updated: 2026-05-28.

This document is the concise live repository status. It is not a scientific
storytelling document.

## Repository State

- Repository: `yuchenzhu-research/ReTrace`
- Active branch: `experiment/retrace-ab-feasibility`
- Starting scaffold: `integration/retrace-v1-complete` @
  `5e8d6e2d1a494d572d6d0fa929595bb198154390`
- Current packet: architecture convergence after the batched Memora
  oracle-conditioned diagnostic smoke.

## Validated Offline

- Typed DPA execution spine.
- Stage A/B controlled contracts, versioned prompts, DirectJudge sibling path,
  and mock/replay tests.
- Fairness and deterministic-grounding hardening.
- AB-1A / AB-1A.5 controlled attribution and auditability protocol.
- AB-1B internal development-case evaluator and replay-only runner.
- Batched Stage A typed-edge proposal and deterministic DPA execution path.

## Implemented Scaffolding, Not Yet Scientific Results

- `HTTPLLMProvider`, cache/accounting, capped-provider wrapper, and run manifest
  infrastructure exist.
- End-to-end development runner exists. Replay mode uses manual fixtures; live
  mode uses prompt components plus overlap retrieval. This is development
  scaffolding, not validated paper-facing retrieval or primary attribution
  evidence.
- STALE adapter/runner entrypoints exist. Current offline/mock execution is
  adapter smoke/dry-run only and is not an official STALE result.
- Stage C remains deferred.

## Memora Oracle-Conditioned Authorization Diagnostic

- Memora released data is present (600 questions, 10 personas, 3 periods).
- `scripts/run_memora_development_eval.py` is a development diagnostic runner.
- Current runner is oracle-conditioned diagnostic only: candidate beliefs
  originate from Memora evaluation annotations (`memory_evidence` /
  `forgetting_evidence`), not from end-to-end memory extraction.
- Batched Stage A has been implemented and offline validated.
- A one-question SiliconFlow DeepSeek-V4-Pro live smoke completed with zero
  errors, four total model calls, and one Stage A batched authorization call.
- This establishes execution connectivity and batching feasibility only.
- No official Memora result exists.

## Explicit Non-Claims

The repository currently has no verified result showing:

- Stage A outperforms Stage B;
- a live provider path is scientifically validated;
- official STALE evaluation is complete;
- official Memora evaluation is complete;
- overlap retrieval is a validated paper-facing retrieval method;
- Stage A traces can be treated as gold typed-edge labels.

## Current Next Boundary

The current next boundary is code convergence and persistent write/read
separation, not another large live run. In particular, ReTrace should separate
write-time memory construction/update from query-time read/evaluation, share
deterministic authorization execution across Stage A paths, and keep the Memora
oracle-conditioned diagnostic separate from any future official agent
evaluation.

## Verification at Starting Scaffold

The integration scaffold reported offline compile/test and mock runner success,
but those runs were mock/replay/smoke validation only. Fresh validation for this
branch must be recorded after the current changes.
