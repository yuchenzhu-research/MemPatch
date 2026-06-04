# ReTrace-Bench — Human Validation Plan

Operational plan for executing the protocol in `human_validation_protocol.md`.

## Goal

Produce paper-grade evidence that ReTrace-Bench gold labels are human-agreeable
and that the task is solvable from the visible evidence, with reported IAA and a
human upper bound — without fabricating any results.

## Roles

| Role | Who | Responsibility |
|---|---|---|
| Annotation lead | dataset author | builds packets, runs scoring, adjudicates |
| Annotator A | real human | independent annotation of the 200-packet |
| Annotator B | real human (ideally not the author) | independent annotation |
| Adjudicator | senior author / third human | resolves A/B disagreements |

LLMs may assist with tooling only, never as an annotator.

## Timeline / milestones

1. **Packet freeze.** Generate packets deterministically (seed 2027):
   ```bash
   python scripts/export_human_annotation_packet.py
   ```
   Record the packet manifest hash; do not regenerate mid-study.
2. **Pilot (Level 1).** One author annotates `quick_audit_50`; sanity-check the
   sheet/codebook and fix any unclear items. This is an internal audit, not a
   paper claim.
3. **Independent annotation (Level 2).** Annotator A and Annotator B each fill
   their own copy of `paper_validation_200_sheet.csv` from the public packet.
4. **Scoring.** Lead runs `scripts/score_human_annotations.py` over both files.
5. **Adjudication.** Disagreements (different decision/diagnosis, or low evidence
   F1) are reviewed; the adjudicator records resolved labels in a third file.
6. **Reporting.** Re-run scoring including the adjudicated file; copy the results
   into the paper and update `human_validation_status.md`.

## Acceptance targets (guidance, not gold-matching mandates)

These are **reporting** targets to interpret results; annotators must label
honestly even if a target is missed.

- decision Cohen's kappa ≥ 0.6 (substantial) desirable;
- solvable_rate high (most scenarios solvable from visible evidence);
- ambiguity_rate low; flagged-ambiguous items reviewed for possible gold fixes;
- human upper-bound joint clearly above the offline baselines (all of which are
  0.0 joint — see `v1_1_offline_baseline_report.md`).

If decision kappa or solvable rate is poor on a pattern, that pattern's gold is
re-examined rather than the human labels being discarded.

## Handling disagreements

- **Decision/diagnosis mismatch:** adjudicator decides; if gold itself looks
  wrong, file a gold-fix note (do not silently edit gold without recording it).
- **Evidence sets differ:** compare against the minimal-evidence rule; the
  smaller correct set wins.
- **Memory-state differs:** adjudicate per `memory_id`.

## What this plan does NOT do

- It does not call any model API.
- It does not let LLM outputs count as human annotations.
- It does not claim completion. Until ≥2 real humans finish the 200-packet and
  scoring is run, status remains "protocol prepared; human annotation pending".
