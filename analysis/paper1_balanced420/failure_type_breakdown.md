# Failure Type Breakdown — paper1_balanced420

Per-family Stage A and Stage B accuracy across all 14 failure types (30 episodes each).

## Stage A Accuracy by Family

| Family | Beliefs | Correct | Accuracy | Errors | Error Rate |
|--------|---------|---------|----------|--------|------------|
| direct_supersession | 60 | 60 | **1.0000** | 0 | 0.0% |
| stale_propagation | 90 | 90 | **1.0000** | 0 | 0.0% |
| scope_expansion | 60 | 60 | **1.0000** | 0 | 0.0% |
| cross_agent_conflict | 60 | 60 | **1.0000** | 0 | 0.0% |
| duplicate_evidence | 30 | 30 | **1.0000** | 0 | 0.0% |
| ambiguous_update | 30 | 30 | **1.0000** | 0 | 0.0% |
| multi_action_supersedes_blocks | 90 | 90 | **1.0000** | 0 | 0.0% |
| reaffirms_only | 30 | 30 | **1.0000** | 0 | 0.0% |
| no_revision | 30 | 30 | **1.0000** | 0 | 0.0% |
| target_ambiguity | 90 | 90 | **1.0000** | 0 | 0.0% |
| multi_action_supersedes_releases | 90 | 89 | **0.9889** | 1 | 1.1% |
| blocks_uncertain | 60 | 59 | **0.9833** | 1 | 1.7% |
| evidence_conflict | 60 | 59 | **0.9833** | 1 | 1.7% |
| **temporary_blocker_recovery** | **30** | **26** | **0.8667** | **4** | **13.3%** |

**10 of 14 families are perfect (1.0).** All 7 belief-level errors concentrate in 4 families, with `temporary_blocker_recovery` contributing 4/7 (57%) of all Stage A errors.

---

## Stage B Accuracy by Family

| Family | Beliefs | B Correct | B Accuracy | B Error Rate |
|--------|---------|-----------|------------|--------------|
| **reaffirms_only** | 30 | 30 | **1.0000** | 0.0% |
| **no_revision** | 30 | 30 | **1.0000** | 0.0% |
| duplicate_evidence | 30 | 24 | 0.8000 | 20.0% |
| cross_agent_conflict | 60 | 30 | 0.5000 | 50.0% |
| scope_expansion | 60 | 30 | 0.5000 | 50.0% |
| blocks_uncertain | 60 | 30 | 0.5000 | 50.0% |
| evidence_conflict | 60 | 30 | 0.5000 | 50.0% |
| target_ambiguity | 90 | 30 | 0.3333 | 66.7% |
| temporary_blocker_recovery | 30 | 9 | 0.3000 | 70.0% |
| direct_supersession | 60 | 0 | 0.0000 | 100.0% |
| stale_propagation | 90 | 0 | 0.0000 | 100.0% |
| ambiguous_update | 30 | 0 | 0.0000 | 100.0% |
| multi_action_supersedes_blocks | 90 | 0 | 0.0000 | 100.0% |
| multi_action_supersedes_releases | 90 | 0 | 0.0000 | 100.0% |

**Only 2 of 14 families achieve 100% Stage B accuracy** (both are trivial single-belief no-change families). Stage B achieves 0% on 5 families including core revision types (direct_supersession, stale_propagation, multi_action).

---

## Domain Breakdown

Both domains (software_engineering, research_workflow) contain 210 episodes each. Stage A errors appear in both domains:
- software_engineering: 3 error episodes (tbr_v8, multi_action_sr_v13, evidence_conflict_v8)
- research_workflow: 4 error episodes (tbr_v1, tbr_v3, tbr_v14, blocks_uncertain_v3)

No significant domain effect on Stage A accuracy.

---

## Key Takeaway

The balanced420 dataset confirms that ReTrace (Stage A) handles **all 14 failure types** with high accuracy. The only systematic weakness is `temporary_blocker_recovery` at 86.7%, where the model's UNCERTAIN action in the blocking submission prevents the RELEASES action from fully recovering the belief to AUTHORIZED. See [temporary_blocker_recovery_deep_dive.md](temporary_blocker_recovery_deep_dive.md) for root cause analysis.
