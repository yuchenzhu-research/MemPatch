# Stage A Error Cases — paper1_balanced420

Total Stage A belief-level errors: **7 out of 810** (0.86% error rate).
Concentrated in **5 episodes** across 4 failure families.

---

## Error Case 1: temporary_blocker_recovery_v8 (SE)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_software_engineering_temporary_blocker_recovery_v8` |
| **failure_type** | temporary_blocker_recovery |
| **domain** | software_engineering |
| **belief_id** | `b_..._temporary_blocker_recovery_v8_1` |
| **gold status** | AUTHORIZED |
| **Stage A status** | UNRESOLVED |
| **error class** | under-update (USABLE → UNCERTAIN) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1 |
| 2 | BLOCKS → condition_1, **UNCERTAIN → belief_1** |
| 3 | RELEASES → condition_1 |

### Gate Decisions

All 4 edges admitted (ok). No gate rejections.

### Likely Cause: **Spurious UNCERTAIN in Sub 2**

The model correctly proposes BLOCKS on the condition in Sub 2, but also emits UNCERTAIN on the belief. Gold expects only BLOCKS on the condition. After RELEASES in Sub 3 restores the condition, the belief should return to AUTHORIZED. However, the UNCERTAIN edge on the belief creates an UNRESOLVED_UNCERTAIN precedence that persists even after the condition is released.

**DPA semantics**: UNCERTAIN edges create permanent UNRESOLVED status (UNRESOLVED_UNCERTAIN precedence). RELEASES only clears the PREREQUISITE_BLOCK from the condition dependency, but cannot clear the direct UNCERTAIN edge on the belief.

**Root cause**: Prompt/proposer issue — the model over-applies UNCERTAIN to the belief when it should only BLOCKS the condition.

**Suggested fix**: Stage C training should teach the policy to use BLOCKS-only on conditions without adding UNCERTAIN on the dependent belief, unless the evidence genuinely creates uncertainty about the belief's truth value independent of the condition.

---

## Error Case 2: temporary_blocker_recovery_v1 (RW)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_research_workflow_temporary_blocker_recovery_v1` |
| **failure_type** | temporary_blocker_recovery |
| **domain** | research_workflow |
| **belief_id** | `b_..._temporary_blocker_recovery_v1_1` |
| **gold status** | AUTHORIZED |
| **Stage A status** | UNRESOLVED |
| **error class** | under-update (USABLE → UNCERTAIN) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1 |
| 2 | BLOCKS → condition_1, **UNCERTAIN → belief_1** |
| 3 | RELEASES → condition_1 |

### Gate Decisions

All 4 edges admitted. Same pattern as v8 above.

### Likely Cause: **Same spurious UNCERTAIN pattern**

Identical to v8. The model emits UNCERTAIN on the belief alongside BLOCKS on the condition. The UNCERTAIN creates permanent UNRESOLVED status that RELEASES cannot clear.

---

## Error Case 3: temporary_blocker_recovery_v3 (RW)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_research_workflow_temporary_blocker_recovery_v3` |
| **failure_type** | temporary_blocker_recovery |
| **domain** | research_workflow |
| **belief_id** | `b_..._temporary_blocker_recovery_v3_1` |
| **gold status** | AUTHORIZED |
| **Stage A status** | BLOCKED |
| **error class** | under-update (USABLE → NOT_USABLE) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1 |
| 2 | BLOCKS → condition_1 |
| 3 | **REAFFIRMS → belief_1** (instead of RELEASES → condition_1) |

### Gate Decisions

All 3 edges admitted.

### Likely Cause: **Missed RELEASES in Sub 3**

The model correctly identifies that the blocker was temporary and tries to recover the belief in Sub 3, but uses REAFFIRMS on the belief instead of RELEASES on the condition. REAFFIRMS cannot clear a PREREQUISITE_BLOCK — only RELEASES on the condition can do that.

**Root cause**: Prompt/proposer issue — the model confuses REAFFIRMS (which strengthens a belief's evidence) with RELEASES (which clears a condition-level block). The model correctly *intends* recovery but selects the wrong action type.

**Suggested fix**: The prompt/ICL examples should emphasize that REAFFIRMS targets beliefs and cannot clear condition blocks. Only RELEASES → condition can clear a PREREQUISITE_BLOCK.

---

## Error Case 4: temporary_blocker_recovery_v14 (RW)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_research_workflow_temporary_blocker_recovery_v14` |
| **failure_type** | temporary_blocker_recovery |
| **domain** | research_workflow |
| **belief_id** | `b_..._temporary_blocker_recovery_v14_1` |
| **gold status** | AUTHORIZED |
| **Stage A status** | UNRESOLVED |
| **error class** | under-update (USABLE → UNCERTAIN) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1 |
| 2 | **UNCERTAIN → belief_1**, BLOCKS → condition_1 |
| 3 | RELEASES → condition_1 |

### Gate Decisions

All 4 edges admitted. Same pattern as v8 and v1.

### Likely Cause: **Spurious UNCERTAIN (same as v8/v1)**

Third instance of the same pattern. The model emits UNCERTAIN + BLOCKS in Sub 2 when gold expects only BLOCKS. The UNCERTAIN creates permanent UNRESOLVED status.

---

## Error Case 5: multi_action_supersedes_releases_v13 (SE)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_software_engineering_multi_action_supersedes_releases_v13` |
| **failure_type** | multi_action_supersedes_releases |
| **domain** | software_engineering |
| **belief_id** | `b_..._multi_action_supersedes_releases_v13_dep` |
| **gold status** | AUTHORIZED |
| **Stage A status** | UNRESOLVED |
| **error class** | under-update (USABLE → UNCERTAIN) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1, RELEASES → condition_1 |
| 2 | **UNCERTAIN → belief_dep** |
| 3 | SUPERSEDES → belief_1, RELEASES → condition_1 |

### Gate Decisions

All edges admitted.

### Likely Cause: **Spurious UNCERTAIN on belief_dep**

The model applies UNCERTAIN to `belief_dep` in Sub 2 when no gold action targets this belief (gold expects no action on `belief_dep`). The UNCERTAIN creates permanent UNRESOLVED status for a belief that should remain AUTHORIZED.

**Root cause**: Prompt/proposer over-caution — the model sees evidence about a related belief and applies UNCERTAIN to a dependent belief that the evidence does not actually affect.

---

## Error Case 6: evidence_conflict_v8 (SE)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_software_engineering_evidence_conflict_v8` |
| **failure_type** | evidence_conflict |
| **domain** | software_engineering |
| **belief_id** | `b_..._evidence_conflict_v8_2` |
| **gold status** | AUTHORIZED |
| **Stage A status** | UNRESOLVED |
| **error class** | under-update (USABLE → UNCERTAIN) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_1 |
| 2 | UNCERTAIN → belief_1, **UNCERTAIN → belief_2** |

### Gate Decisions

All 3 edges admitted.

### Likely Cause: **Over-application of UNCERTAIN to belief_2**

Gold expects UNCERTAIN only on belief_1 (the conflicted belief). The model extends UNCERTAIN to belief_2 as well, even though the evidence conflict does not affect belief_2. This is a scope-expansion error: the model treats conflicting evidence as affecting all beliefs rather than just the targeted one.

**Root cause**: Prompt/proposer issue — the model fails to scope the UNCERTAIN action to only the belief directly affected by the conflicting evidence.

---

## Error Case 7: blocks_uncertain_v3 (RW)

| Field | Value |
|-------|-------|
| **episode_id** | `ep_paper1_research_workflow_blocks_uncertain_v3` |
| **failure_type** | blocks_uncertain |
| **domain** | research_workflow |
| **belief_id** | `b_..._blocks_uncertain_v3_main` |
| **gold status** | BLOCKED |
| **Stage A status** | UNRESOLVED |
| **error class** | uncertainty error (NOT_USABLE → UNCERTAIN via DPA mapping) |

### Typed Actions

| Sub | Actions |
|-----|---------|
| 1 | REAFFIRMS → belief_main |
| 2 | UNCERTAIN → belief_main, UNCERTAIN → belief_other |

### Gate Decisions

All 3 edges admitted.

### Likely Cause: **UNCERTAIN instead of BLOCKS on condition**

Gold expects the model to BLOCKS a condition associated with `belief_main`, which would produce a BLOCKED (PREREQUISITE_BLOCK) status. Instead, the model uses UNCERTAIN directly on the belief, producing UNRESOLVED (UNRESOLVED_UNCERTAIN). Both map to the same "not usable" space in Stage B, but in Stage A's fine-grained DPA status, UNRESOLVED ≠ BLOCKED.

**Note**: When mapped through `STATUS_MAP_A_TO_COMPARABLE`, BLOCKED → NOT_USABLE and UNRESOLVED → UNCERTAIN, so this is scored as an error. If the scoring used a coarser "not safely usable" equivalence class, this case might be considered correct. This is a legitimate DPA-level semantic distinction, not a scoring artifact.

**Root cause**: The model uses UNCERTAIN (a belief-level hedge) when the gold expects BLOCKS (a condition-level invalidation). The model correctly identifies that the belief should not be used, but expresses this through the wrong mechanism.

---

## Summary of Error Patterns

| Pattern | Count | Episodes |
|---------|-------|----------|
| Spurious UNCERTAIN alongside correct BLOCKS | 3 | tbr_v8, tbr_v1, tbr_v14 |
| Missed RELEASES (used REAFFIRMS instead) | 1 | tbr_v3 |
| Over-scoped UNCERTAIN to unaffected belief | 2 | multi_action_sr_v13, evidence_conflict_v8 |
| UNCERTAIN instead of BLOCKS on condition | 1 | blocks_uncertain_v3 |

**All 7 errors are under-update or uncertainty errors. Zero over-updates. Zero stale propagation.**

The dominant failure mode (4/7) is in the `temporary_blocker_recovery` family where the model mishandles the BLOCKS → RELEASES recovery sequence, either by adding spurious UNCERTAIN edges or by using REAFFIRMS instead of RELEASES.
