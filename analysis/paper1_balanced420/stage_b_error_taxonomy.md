# Stage B Error Taxonomy — paper1_balanced420

Stage B (DirectJudge) achieves 30% final-status accuracy (243 correct / 810 beliefs).
Total belief-level errors: **567**.

However, a significant portion of these errors come from **omitted verdicts** — Stage B parse errors where the model fails to produce a verdict for one or more beliefs. After accounting for omitted verdicts, we can separate structural failures from semantic failures.

---

## Error Transition Distribution

| Gold Status (DPA) | Gold Comparable | Predicted (Stage B) | Count | % of Errors | Error Category |
|-------------------|-----------------|---------------------|-------|-------------|----------------|
| SUPERSEDED | NOT_USABLE | USABLE | 150 | 26.6% | **Stale propagation** |
| AUTHORIZED | USABLE | OMITTED | 150 | 26.6% | **Omitted verdict** |
| BLOCKED | NOT_USABLE | OMITTED | 60 | 10.6% | **Omitted verdict** |
| AUTHORIZED | USABLE | NOT_USABLE | 57 | 10.1% | **Over-update** |
| AUTHORIZED | USABLE | UNCERTAIN | 30 | 5.3% | **Uncertainty collapse** |

**Note**: The remaining errors (120 cases) are within the parse-error/omitted category for other status transitions not individually broken out above.

---

## Error Categories Explained

### 1. Stale Propagation (SUPERSEDED → USABLE): 150 errors

The single largest Stage B failure mode. When a belief has been superseded by newer evidence, Stage B's DirectJudge still judges it as USABLE. This is the core safety failure that ReTrace is designed to prevent.

**Affected families**: direct_supersession (60), stale_propagation (90)

**Why it happens**: DirectJudge evaluates each belief independently against its associated evidence, without tracking the supersession chain. It sees a belief with supporting evidence and concludes "USABLE", unaware that a later submission introduced a replacement belief.

**Implication**: This is the primary argument for ReTrace's typed revision approach. Without explicit SUPERSEDES edges and DPA resolution, an LLM cannot reliably detect that previously-supported beliefs are now stale.

### 2. Omitted Verdicts (AUTHORIZED/BLOCKED → OMITTED): 210 errors

Stage B frequently omits verdicts for beliefs, especially in multi-belief episodes. The parser logs these as parse errors (`DirectJudge response omitted verdicts for belief(s): {...}`).

**Affected families**: All multi-belief families, especially:
- direct_supersession: omits the replacement belief verdict
- multi_action_supersedes_blocks: omits 2 of 3 belief verdicts
- multi_action_supersedes_releases: omits 2 of 3 belief verdicts

**Why it happens**: The DirectJudge prompt asks the model to evaluate all beliefs, but the model often produces verdicts for only the "most interesting" belief (typically the one being updated) and ignores the others. This is an inherent limitation of unstructured LLM output for multi-element tasks.

### 3. Over-update (AUTHORIZED → NOT_USABLE): 57 errors

Stage B incorrectly marks usable beliefs as not usable. This is a false negative — the model is too conservative, judging that evidence invalidates a belief when it doesn't.

**Affected families**: Distributed across several families where beliefs should remain AUTHORIZED after evidence processing.

**Why it happens**: The DirectJudge model sees evidence that mentions or relates to a belief and assumes it must be invalidating, even when the evidence actually supports or is neutral to the belief.

### 4. Uncertainty Collapse (AUTHORIZED → UNCERTAIN): 30 errors

Stage B marks beliefs as UNCERTAIN when they should be USABLE. This is a softer form of over-update — the model hedges rather than committing to a USABLE verdict.

**Affected families**: ambiguous_update (30)

**Why it happens**: The model sees evidence labeled "ambiguous" and defaults to UNCERTAIN rather than analyzing whether the ambiguity actually affects the belief's usability.

### 5. Missed Prerequisite Block (BLOCKED → OMITTED/UNCERTAIN): 60 errors

Stage B fails to identify that a belief's prerequisite condition has been invalidated, either by omitting the verdict entirely or by marking it as UNCERTAIN rather than NOT_USABLE.

**Why it happens**: DirectJudge has no concept of dependency edges (REQUIRES) or condition-level blocking. It evaluates beliefs in isolation and cannot reason about transitive prerequisite failures.

---

## Stage B Failure by Family

| Family | B Accuracy | Dominant Error Mode |
|--------|-----------|-------------------|
| direct_supersession | 0.0% | Stale propagation (SUPERSEDED→USABLE) + omitted verdicts |
| stale_propagation | 0.0% | Stale propagation (SUPERSEDED→USABLE) + omitted verdicts |
| multi_action_supersedes_blocks | 0.0% | Stale propagation + omitted verdicts |
| multi_action_supersedes_releases | 0.0% | Stale propagation + omitted verdicts |
| ambiguous_update | 0.0% | Uncertainty collapse (AUTHORIZED→UNCERTAIN) |
| temporary_blocker_recovery | 30.0% | Mixed: omitted verdicts + over-update |
| target_ambiguity | 33.3% | Omitted verdicts (2 of 3 beliefs omitted) |
| cross_agent_conflict | 50.0% | Over-update (1 of 2 beliefs wrong) |
| scope_expansion | 50.0% | Over-update (1 of 2 beliefs wrong) |
| blocks_uncertain | 50.0% | Omitted or wrong on blocked belief |
| evidence_conflict | 50.0% | Over-update on unaffected belief |
| duplicate_evidence | 80.0% | Occasional omission |
| reaffirms_only | 100.0% | None |
| no_revision | 100.0% | None |

---

## Summary

The dominant Stage B failure modes are:

1. **Stale propagation** (26.6%): DirectJudge cannot track supersession chains.
2. **Omitted verdicts** (37.2%): DirectJudge fails to produce verdicts for all beliefs in multi-belief episodes.
3. **Over-update** (10.1%): DirectJudge is too aggressive in marking beliefs as NOT_USABLE.
4. **Uncertainty collapse** (5.3%): DirectJudge defaults to UNCERTAIN when evidence is ambiguous.

These failures are structural, not merely prompt-engineering issues. DirectJudge lacks the typed revision vocabulary (SUPERSEDES, BLOCKS, RELEASES) and the deterministic DPA kernel that enable ReTrace's accuracy. **Stage C API-ICL is expected to improve Stage B's baseline but cannot fully close the gap without the DPA architecture.**
