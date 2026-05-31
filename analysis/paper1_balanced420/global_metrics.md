# Global Metrics — paper1_balanced420

**Run**: `paper1_balanced420_deepseek_v3_conflict_aware_live_20260531_174458`
**Dataset**: paper1_balanced (420 episodes, 14 failure families × 30 each, 2 domains × 210)
**Model**: deepseek-ai/DeepSeek-V3 via siliconflow
**Stage A variant**: conflict_aware constrained (zero-shot)
**Temperature**: 0.0, seed 42, deterministic decoding

---

## Stage A (ReTrace: Typed Proposal → DPA → authorize)

| Metric | Value |
|--------|-------|
| **Final status accuracy (micro)** | **0.9914** |
| **Final status accuracy (macro)** | **0.9873** |
| Episode exact match rate | 0.9833 |
| Over-update (stale propagation) rate | 0.0000 |
| Under-update rate | 0.0143 |
| Uncertainty error rate | 0.0074 |
| Grounding error rate | 0.0000 |
| Valid output rate | 1.0000 |
| Parser error rate | 0.0000 |
| First-pass valid JSON rate | 1.0000 |
| Repair attempt rate | 0.0000 |

### Action-Level Metrics

| Metric | Value | Note |
|--------|-------|------|
| action_type_match | 0.4520 | See [action_metric_interpretation.md](action_metric_interpretation.md) |
| exact_action_match | 0.4033 | |
| evidence_grounding | 0.4520 | |
| target_grounding | 1.0000 | |
| no_revision_match | 0.4944 | |
| false_no_revision_rate | 0.0000 | |
| multi_action_recall | 0.9958 | |

---

## Stage B (DirectJudge: API model directly predicts usability)

| Metric | Value |
|--------|-------|
| **Final status accuracy** | **0.3000** |
| Over-update rate | 0.5556 |
| Stale propagation rate | 0.5556 |
| Under-update rate | 0.5643 |
| Uncertainty error rate | 0.4444 |
| Grounding error rate | 0.0000 |
| Valid output rate | 0.8333 |
| Canonicalization rate | 0.0000 |

---

## Summary

Stage A (ReTrace) achieves **99.14% final-status accuracy** with **zero stale propagation** and **zero parser errors** across 420 balanced episodes. The deterministic DPA kernel correctly resolves all typed proposals into final authorization statuses.

Stage B (DirectJudge baseline) achieves only **30% accuracy**, demonstrating that without the typed revision vocabulary and deterministic DPA, a direct LLM judgment approach fails catastrophically on multi-step memory revision tasks. Stage B's 55.6% stale propagation rate means it frequently keeps superseded/blocked beliefs as USABLE.

The **3.3× accuracy gap** (99.1% vs 30.0%) is the primary quantitative evidence for the ReTrace method.
