# paper1_balanced420 Error Analysis Summary

**Run**: DeepSeek-V3 via SiliconFlow, conflict_aware constrained, zero-shot
**Dataset**: paper1_balanced (420 episodes, 14 failure families × 30, 2 domains × 210)
**Date**: 2026-05-31

---

## Global Result

| Stage | Final Status Accuracy | Stale Propagation | Parser Errors |
|-------|----------------------|-------------------|---------------|
| **A (ReTrace)** | **99.14%** | **0.00%** | **0.00%** |
| B (DirectJudge) | 30.00% | 55.56% | 16.67% |

ReTrace achieves a **3.3× accuracy advantage** over the DirectJudge baseline across 420 balanced episodes spanning 14 failure types and 2 domains.

---

## Strongest Evidence for ReTrace

1. **Zero stale propagation**: ReTrace never allows a superseded or blocked belief to remain USABLE. DirectJudge has 55.6% stale propagation.

2. **Perfect accuracy on 10/14 families**: Including the hardest multi-action types (multi_action_supersedes_blocks, multi_action_supersedes_releases) and critical safety types (stale_propagation, direct_supersession).

3. **Zero parser/grounding errors**: 100% valid JSON, 100% valid target grounding, zero repair attempts needed.

4. **99.6% multi-action recall**: When gold requires multiple simultaneous revision actions, the model recalls 99.6% of them.

5. **Both domains equivalent**: No significant accuracy difference between software_engineering and research_workflow.

---

## Hardest Stage A Failure Families

| Rank | Family | Accuracy | Errors | Dominant Error |
|------|--------|----------|--------|----------------|
| 1 | temporary_blocker_recovery | 86.67% | 4/30 | Spurious UNCERTAIN alongside BLOCKS |
| 2 | blocks_uncertain | 98.33% | 1/60 | UNCERTAIN instead of BLOCKS on condition |
| 3 | evidence_conflict | 98.33% | 1/60 | Over-scoped UNCERTAIN to unaffected belief |
| 4 | multi_action_supersedes_releases | 98.89% | 1/90 | Spurious UNCERTAIN on dependent belief |

All errors are **under-update or uncertainty errors** — the model is too conservative, never too aggressive. This is a favorable failure mode for safety-critical memory systems.

---

## Dominant Stage B Failure Modes

1. **Stale propagation** (150 beliefs): DirectJudge keeps superseded beliefs as USABLE.
2. **Omitted verdicts** (210 beliefs): DirectJudge fails to produce verdicts for all beliefs in multi-belief episodes.
3. **Over-update** (57 beliefs): DirectJudge incorrectly marks usable beliefs as NOT_USABLE.
4. **Uncertainty collapse** (30 beliefs): DirectJudge defaults to UNCERTAIN for ambiguous evidence.

These failures are **structural** — DirectJudge lacks typed revision edges and deterministic DPA resolution.

---

## Action-Level Metric Interpretation

action_type_match (0.452) and exact_action_match (0.403) are low despite 99.1% final-status accuracy because:
- Gold uses **minimal** action sets (only critical revision actions)
- Model produces **590 REAFFIRMS** vs gold's 30 (518 in submissions where gold expects NO_REVISION)
- REAFFIRMS is DPA-harmless but penalized by strict action matching

This is a **metric artifact**, not a real action-quality weakness. See [action_metric_interpretation.md](../analysis/paper1_balanced420/action_metric_interpretation.md) for full analysis.

**Paper framing**: Present as a limitation/interpretation point. The primary metric is final-status accuracy.

---

## Implications for Stage C API-ICL

1. **temporary_blocker_recovery** is the priority training family:
   - 4/7 Stage A errors come from this family
   - ICL examples should demonstrate BLOCKS-only pattern (no spurious UNCERTAIN)
   - Should explicitly document that UNCERTAIN is permanent and non-reversible

2. **REAFFIRMS inflation** can be addressed by:
   - Training signal that penalizes unnecessary REAFFIRMS in first submissions
   - Or accepting REAFFIRMS as valid and using relaxed action matching

3. **Expected Stage C gains**:
   - temporary_blocker_recovery: 86.7% → ~97%+ (learnable pattern with 2 distinct error modes)
   - Other families: already at 98-100%, marginal gains expected
   - Overall: 99.1% → ~99.5%+ is plausible with ICL

---

## What Still Must Be Done Before Final Paper Claims

1. **Stage C API-ICL run**: Confirm that ICL examples improve temporary_blocker_recovery accuracy.

2. **Additional model providers**: Run with at least one more model (e.g., Claude, GPT-4) to show the ReTrace advantage is model-independent.

3. **Confidence intervals**: 420 episodes may be sufficient for aggregate metrics but family-level confidence intervals should be reported (30 per family).

4. **Action metric reporting**: Decide on paper framing for action_type_match. Options:
   - Report both final-status and action-level metrics with interpretation
   - Define a "critical action match" metric that excludes REAFFIRMS
   - Report multi_action_recall as the primary action-level metric

5. **Stage B improvement**: Current 30% is likely unfairly low due to parse errors. Consider a Stage B variant with structured output or retry logic.

6. **STALE/CUPMem external validation**: Planned but not yet executed.

7. **Ablation studies**: Demonstrate the contribution of each component (typed edges, DPA, gate) via ablation.

---

## File Index

| File | Content |
|------|---------|
| [global_metrics.md](../analysis/paper1_balanced420/global_metrics.md) | Full metric tables |
| [failure_type_breakdown.md](../analysis/paper1_balanced420/failure_type_breakdown.md) | Per-family accuracy breakdown |
| [stage_a_error_cases.md](../analysis/paper1_balanced420/stage_a_error_cases.md) | All 7 Stage A error cases with root cause |
| [stage_b_error_taxonomy.md](../analysis/paper1_balanced420/stage_b_error_taxonomy.md) | Stage B error transition analysis |
| [temporary_blocker_recovery_deep_dive.md](../analysis/paper1_balanced420/temporary_blocker_recovery_deep_dive.md) | TBR family root cause analysis |
| [action_metric_interpretation.md](../analysis/paper1_balanced420/action_metric_interpretation.md) | Action-level metric interpretation |
