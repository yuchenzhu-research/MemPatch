# Implementation Status

Last updated: 2026-05-28.

This document is the concise live repository status. It is not a scientific
storytelling document.

## Repository State

- Repository: `yuchenzhu-research/ReTrace`
- Active branch: `refactor/retrace-minimal-core`
- Starting scaffold: `integration/retrace-v1-complete` @
  `5e8d6e2d1a494d572d6d0fa929595bb198154390`
- Current packet: Memora oracle-conditioned Stage A/B diagnostic failure
  attribution after first 30-question quantitative run.
- Current verified base: `76905510362df74841c18ae99ddcabdcc16fafcd`.

## Validated Offline

- Typed DPA execution spine.
- Stage A/B controlled contracts, versioned prompts, DirectJudge sibling path,
  and mock/replay tests.
- Fairness and deterministic-grounding hardening.
- AB-1A / AB-1A.5 controlled attribution and auditability protocol.
- AB-1B internal development-case evaluator and replay-only runner.
- Batched Stage A typed-edge proposal and deterministic DPA execution path.
- Shared proposal strategy interface for backend ingestion.
- Memora oracle-conditioned authorization metrics, DirectJudge belief-id parser
  hardening, and offline Stage A provenance failure analyzer.

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
- A 30-question weekly/all-personas SiliconFlow DeepSeek-V4-Pro diagnostic
  completed. It is diagnostic-only and oracle-conditioned, not official Memora
  or FAMA evaluation.
- Run facts: 30 questions, 1 Stage B parser-format error, 119 total model calls,
  240786 total tokens. Stage A used 30 authorization calls plus 30 answer calls;
  Stage B used 30 adjudication calls plus 29 answer calls.
- Stage A metrics on the current diagnostic mapping:
  - memory preservation accuracy: 0.96875 (31/32);
  - forgetting suppression accuracy: 0.034482758620689655 (4/116);
  - balanced authorization accuracy: 0.5016163793103449;
  - overall item accuracy: 0.23648648648648649;
  - uncertain rate: 0.033783783783783786.
- Stage B metrics on the same current diagnostic mapping:
  - memory preservation accuracy: 1.0 (32/32);
  - forgetting suppression accuracy: 0.7565217391304347 (87/115);
  - balanced authorization accuracy: 0.8782608695652174;
  - overall item accuracy: 0.8095238095238095;
  - uncertain rate: 0.0;
  - execution errors: 1.
- Stage A minus Stage B:
  - memory preservation: -0.03125;
  - forgetting suppression: -0.7220389805;
  - balanced authorization accuracy: -0.3766444903;
  - overall item accuracy: -0.5730373230.
- Batched Stage A execution was operationally successful but substantially
  underperformed DirectJudge on the current diagnostic metric.
- Stage A preserved nearly all `forgetting_absence` candidates. Read-only
  provenance audit found 112 Stage A forgetting false positives; all had an
  admitted `REAFFIRMS` edge and final `AUTHORIZED` status. The 4 Stage A
  correctly excluded forgetting candidates were `UNRESOLVED`, not defeat-path
  exclusions.
- Current interpretation route: **Route A — adapter/target misalignment
  dominates**. Memora `forgetting_absence` is an answer-surface absence
  criterion, while the current adapter maps it to candidate belief authorization.
  Failure attribution is therefore pending a conflict-grounded authorization
  subset definition.
- No new large live run is justified until that subset definition is audited.
- No official Memora result exists.

## Explicit Non-Claims

The repository currently has no verified result showing:

- Stage A outperforms Stage B;
- Stage A is scientifically acceptable on the current naive Memora mapping;
- a live provider path is scientifically validated;
- official STALE evaluation is complete;
- official Memora evaluation is complete;
- overlap retrieval is a validated paper-facing retrieval method;
- Stage A traces can be treated as gold typed-edge labels.

## Current Next Boundary

The current next boundary is not another live run. The next work should define
and audit a filtered conflict-grounded authorization subset for Memora-style
forgetting targets. Do not tune Stage A on the current overall
`forgetting_absence` metric, do not run a 150-question expansion, and do not
change DPA or RevisionGate unless a deterministic offline repro shows a
Gate/DPA failure.

## Canonical Entrypoints

- Offline validation:
  `env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts`
  and `.venv/bin/python -m pytest`.
- Current demo replay:
  `.venv/bin/python scripts/run_memora_development_eval.py --mode replay --period weekly --persona academic_researcher --limit-questions 2 --stage-a-execution batched`.
- Offline Stage A failure analysis for an existing diagnostic report:
  `.venv/bin/python scripts/analyze_memora_oracle_failure_modes.py --report outputs/memora_oracle_diag_batched_metrics_weekly_30/memora_development_report.json --output-dir outputs/analysis`.
- Reference regression paths remain in tests and development scripts but are
  not the default demo path.

## Verification at Starting Scaffold

The integration scaffold reported offline compile/test and mock runner success,
but those runs were mock/replay/smoke validation only. Fresh validation for this
branch must be recorded after the current changes.
