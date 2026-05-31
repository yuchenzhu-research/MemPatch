# Temporary Blocker Recovery Deep Dive — paper1_balanced420

The `temporary_blocker_recovery` family is the weakest Stage A family at **86.7% accuracy** (26/30 beliefs correct). This document analyzes all 4 error cases.

---

## Family Design

Each `temporary_blocker_recovery` episode has:
- 1 belief with 1 prerequisite condition (REQUIRES dependency)
- 3 submissions:
  1. **Initial evidence**: reaffirms the belief (condition is satisfied)
  2. **Blocking evidence**: shows the condition is temporarily violated
  3. **Recovery evidence**: shows the condition is restored

**Gold expects**:
- Sub 1: REAFFIRMS → belief (or NO_REVISION)
- Sub 2: BLOCKS → condition
- Sub 3: RELEASES → condition

**Gold final status**: AUTHORIZED (belief recovers after temporary block)

---

## Error Cases

### Pattern A: Spurious UNCERTAIN (3 of 4 errors)

| Episode | Sub 2 Actions (Predicted) | Sub 2 Actions (Gold) | Final Status |
|---------|--------------------------|---------------------|--------------|
| SE_v8 | BLOCKS→cond, **UNCERTAIN→belief** | BLOCKS→cond | UNRESOLVED |
| RW_v1 | BLOCKS→cond, **UNCERTAIN→belief** | BLOCKS→cond | UNRESOLVED |
| RW_v14 | **UNCERTAIN→belief**, BLOCKS→cond | BLOCKS→cond | UNRESOLVED |

**What goes wrong**: The model correctly identifies that the condition should be BLOCKED. But it *also* applies UNCERTAIN to the belief, reasoning (plausibly) that if the condition is failing, the belief's truth is uncertain.

**Why this is wrong**: In ReTrace's DPA semantics, UNCERTAIN creates a permanent UNRESOLVED_UNCERTAIN status on the belief that cannot be cleared by RELEASES on the condition. RELEASES only clears the PREREQUISITE_BLOCK path. The UNCERTAIN edge persists, keeping the belief at UNRESOLVED even after the condition is released.

**DPA precedence chain**:
```
After Sub 2: PREREQUISITE_BLOCK (from BLOCKS→cond) + UNRESOLVED_UNCERTAIN (from UNCERTAIN→belief)
After Sub 3: RELEASES clears PREREQUISITE_BLOCK → remaining: UNRESOLVED_UNCERTAIN
Final: UNRESOLVED (from UNRESOLVED_UNCERTAIN precedence)
```

**Diagnosis**: **Prompt/proposer issue**, not DPA/gate issue. The DPA and gate correctly process all edges. The problem is that the model over-applies UNCERTAIN when BLOCKS alone would suffice.

The model's reasoning is semantically plausible — "if the condition is failing, the belief is uncertain" — but it misunderstands the DPA invariant: a BLOCKS→condition is sufficient to mark the belief as not-currently-usable via the REQUIRES dependency. Adding UNCERTAIN→belief is redundant and harmful because UNCERTAIN has no corresponding "un-uncertain" action.

### Pattern B: Missed RELEASES (1 of 4 errors)

| Episode | Sub 3 Actions (Predicted) | Sub 3 Actions (Gold) | Final Status |
|---------|--------------------------|---------------------|--------------|
| RW_v3 | **REAFFIRMS→belief** | RELEASES→cond | BLOCKED |

**What goes wrong**: The model correctly identifies that the belief should recover in Sub 3. But it uses REAFFIRMS→belief instead of RELEASES→condition. REAFFIRMS adds a supporting evidence edge to the belief but cannot clear a PREREQUISITE_BLOCK.

**DPA precedence chain**:
```
After Sub 2: PREREQUISITE_BLOCK (from BLOCKS→cond)
After Sub 3: REAFFIRMS→belief adds REAFFIRMS edge, but PREREQUISITE_BLOCK > AUTHORIZED
Final: BLOCKED (PREREQUISITE_BLOCK precedence wins)
```

**Diagnosis**: **Prompt/proposer issue** — the model confuses the target type. REAFFIRMS targets beliefs; RELEASES targets conditions. The model's intent is correct (recovery) but the action vocabulary selection is wrong.

---

## Correct Cases (26/30)

All 26 correct cases follow the gold pattern exactly:
1. Sub 1: REAFFIRMS → belief (harmless, correct intent)
2. Sub 2: BLOCKS → condition (correct, no spurious UNCERTAIN)
3. Sub 3: RELEASES → condition (correct)

The model succeeds when it:
- Uses BLOCKS-only in Sub 2 without adding UNCERTAIN
- Uses RELEASES→condition in Sub 3 instead of REAFFIRMS→belief

---

## Root Cause Summary

| Root Cause | Count | Fix |
|-----------|-------|-----|
| **Prompt/proposer: spurious UNCERTAIN** | 3 | Stage C training; ICL examples showing BLOCKS-only for temporary blockers |
| **Prompt/proposer: wrong action type (REAFFIRMS vs RELEASES)** | 1 | Stage C training; emphasize RELEASES targets conditions |
| DPA/gate issue | 0 | — |
| Gold/data issue | 0 | — |
| Missing condition target | 0 | — |

**All 4 errors are prompt/proposer issues.** The DPA and gate work correctly in all cases. The gold labels and data are correct.

---

## Implications for Stage C

1. **ICL examples**: Include at least one `temporary_blocker_recovery` example that demonstrates the correct BLOCKS→RELEASES pattern without UNCERTAIN.
2. **Training signal**: The 26 correct cases provide positive training signal; the 4 error cases provide negative signal for the spurious UNCERTAIN pattern.
3. **Action vocabulary documentation**: The prompt should explicitly state that UNCERTAIN creates permanent status that cannot be reversed, unlike BLOCKS which can be reversed by RELEASES.
4. **Expected improvement**: This failure pattern is highly learnable — the model already gets 26/30 correct, and the 4 errors follow just 2 distinct patterns. Stage C API-ICL should eliminate most of these errors.
