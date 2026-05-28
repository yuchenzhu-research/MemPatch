# Implementation Status

Last updated: 2026-05-28.

This document is the concise live repository status. It is not a scientific
storytelling document.

## Repository State

- Repository: `yuchenzhu-research/ReTrace`
- Active branch: `experiment/retrace-ab-feasibility`
- Starting scaffold: `integration/retrace-v1-complete` @
  `5e8d6e2d1a494d572d6d0fa929595bb198154390`
- Current packet: v3 authority adoption, provider/smoke safety repair, and an
  internal Ambiguity-and-Scope Stage A/B feasibility diagnostic.

## Validated Offline

- Typed DPA execution spine.
- Stage A/B controlled contracts, versioned prompts, DirectJudge sibling path,
  and mock/replay tests.
- Fairness and deterministic-grounding hardening.
- AB-1A / AB-1A.5 controlled attribution and auditability protocol.
- AB-1B internal development-case evaluator and replay-only runner.

## Implemented Scaffolding, Not Yet Scientific Results

- `HTTPLLMProvider`, cache/accounting, capped-provider wrapper, and run manifest
  infrastructure exist, but real provider behavior must be repaired and
  validated before live feasibility claims.
- End-to-end development runner exists. Replay mode uses manual fixtures; live
  mode uses prompt components plus overlap retrieval. This is development
  scaffolding, not validated paper-facing retrieval or primary attribution
  evidence.
- STALE adapter/runner entrypoints exist. Current offline/mock execution is
  adapter smoke/dry-run only and is not an official STALE result.
- Memora wrapper/runner entrypoints exist. Current offline/mock execution is
  adapter smoke/dry-run only and is not an official Memora result.
- Stage C remains deferred.

## Explicit Non-Claims

The repository currently has no verified result showing:

- Stage A outperforms Stage B;
- a live provider path is scientifically validated;
- official STALE evaluation is complete;
- official Memora evaluation is complete;
- overlap retrieval is a validated paper-facing retrieval method;
- Stage A traces can be treated as gold typed-edge labels.

## Current Next Boundary

Before any official benchmark evaluation or Stage C work, run an internal Stage
A/B Ambiguity-and-Scope feasibility diagnostic on fixed `SharedCandidateView`
inputs after provider and smoke-run safety fixes.

## Verification at Starting Scaffold

The integration scaffold reported offline compile/test and mock runner success,
but those runs were mock/replay/smoke validation only. Fresh validation for this
branch must be recorded after the current changes.
