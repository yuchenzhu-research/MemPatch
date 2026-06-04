# ReTrace-Bench (internal "v1.1") — Validation Report

**Generator:** `scripts/generate_retrace_bench_final.py`
**Generation seed:** `2027` (deterministic; not API/decoding randomness)
**Canonical data root:** `data/retrace_bench_v1_1/`
**Validator:** `scripts/validate_retrace_bench_dataset.py` (schema + leakage + pattern semantics)
**Gold oracle:** `scripts/check_retrace_bench_gold_oracle.py` (scorer self-consistency)

All five splits were regenerated from scratch with seed `2027` and re-validated.
Re-running the generator with the same seed reproduces these splits byte-for-byte.

## 1. Scenario counts

| Split | Dir | Count | Public? |
|---|---|---|---|
| main | `main_3000_en` | 3000 | yes |
| hard | `hard_500_en` | 500 | yes |
| realistic | `realistic_200_en` | 200 | yes (secondary / stress) |
| calibration | `calibration_80_en` | 80 | yes (smoke only) |
| private_hidden | `private_hidden_200_en` | 200 | **no** (private eval) |

Public total = **3780**. Including private hidden = **3980**.

## 2. Validator errors / warnings per split

| Split | Errors | Warnings | Notes |
|---|---|---|---|
| main | 0 | 0 | clean |
| hard | 0 | 0 | clean |
| realistic | 0 | 200 | every row warns `synthetic_gold_unreviewed` (expected) |
| calibration | 0 | 0 | clean |
| private_hidden | 0 | 0 | clean |

**No schema errors and no gold-leakage errors on any split.** The only warnings
are the 200 realistic-split `synthetic_gold_unreviewed` notices, which are
intentional: realistic gold is synthetic and has **not** been human-reviewed.

## 3. Difficulty distributions

| Split | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| main | 750 | 750 | 750 | 750 |
| hard | – | – | 250 | 250 |
| realistic | – | – | 133 | 67 |
| calibration | – | 40 | 40 | – |
| private_hidden | – | 100 | 100 | – |

Hard is **L3/L4 only** as required. Main spans L1–L4 evenly.

## 4. Decision distributions

### hard (Part-3 balancing target enforced)

| Decision | Count | Share | Target range | OK |
|---|---|---|---|---|
| use_current_memory | 250 | 50.0% | 45–55% | ✓ |
| mark_unresolved | 100 | 20.0% | 15–25% | ✓ |
| ask_clarification | 60 | 12.0% | 10–15% | ✓ |
| refuse_due_to_policy | 50 | 10.0% | 8–12% | ✓ |
| escalate | 40 | 8.0% | 5–10% | ✓ |

### main / realistic / calibration / private_hidden

| Split | use_current_memory | mark_unresolved | refuse_due_to_policy |
|---|---|---|---|
| main | 2200 (73.3%) | 600 (20.0%) | 200 (6.7%) |
| realistic | 147 (73.5%) | 40 (20.0%) | 13 (6.5%) |
| calibration | 59 (73.8%) | 16 (20.0%) | 5 (6.3%) |
| private_hidden | 147 (73.5%) | 40 (20.0%) | 13 (6.5%) |

**Observation (documented, not a blocker):** the non-hard splits are
`use_current_memory`-dominant and currently use only three decision labels
(`use_current_memory`, `mark_unresolved`, `refuse_due_to_policy`); they do not
emit `ask_clarification` / `escalate`. The full five-way decision balance and
the anti-recency / minimal-evidence pressure are enforced on the **hard** split,
which is the headline discrimination split. Main is intentionally broader and
easier. This skew is a deterministic property of the current generator; widening
the non-hard decision space is recorded as a future-robustness item rather than
changed here (changing it would alter the seed-`2027` canonical splits).

## 5. Pattern distributions

All **15** canonical workflow patterns are present in every split.

- **hard:** max pattern share = `ci_failed_after_claim` at 110/500 = **22.0%**
  (< 25% cap). Remaining patterns range 2–10%.
- **main:** perfectly uniform — 200 each across all 15 patterns.
- **realistic / calibration / private_hidden:** near-uniform (13–14 / 5–6 per
  pattern respectively).

## 6. Failure-mode distributions (primary_failure_mode)

Eight canonical failure modes appear across splits. Representative (hard):

| Failure mode | hard count |
|---|---|
| conflict_collapse | 146 |
| under_update | 86 |
| stale_memory_reuse | 66 |
| scope_leakage | 60 |
| policy_violation | 50 |
| wrong_source_attribution | 42 |
| over_update | 38 |
| failure_to_release_or_restore | 12 |

Main covers the same eight modes with a broader, flatter distribution. The
diagnostic label space (`expected_failure_diagnosis`) covers all canonical
failure-diagnosis values; every emitted value is a canonical enum member
(checked by the validator and by `test_expected_decisions_and_diagnoses_are_valid_enums`).

## 7. Average required evidence count

| Split | avg | min | max |
|---|---|---|---|
| main | 1.733 | 1 | 3 |
| hard | 1.690 | 1 | 3 |
| realistic | 1.755 | 1 | 3 |
| calibration | 1.788 | 1 | 3 |
| private_hidden | 1.755 | 1 | 3 |

Hard satisfies **avg required evidence > 1.0** (no single-event shortcut on
average); minimal-evidence exactness is separately enforced by the scorer and
gold oracle.

## 8. Leakage audit summary

The validator's public-text leakage gate passed with **zero leakage errors** on
all splits. Confirmed that none of the following appear in any model-facing
public view (`public_input`, task prompts, `workflow_context`):

`hidden_gold`, `metadata`, `validation_notes`, `source_pointers`,
`is_distractor`, `primary_failure_mode`, `pattern_trap_type`,
`canonical_failure_mode`, and the full `expected_answer` text.

Every `expected_evidence_event_ids` entry resolves to a real `event_id` in the
scenario's `event_trace`, and every `expected_memory_state` key resolves to a
real `memory_id` in `initial_memory` (enforced by validator + unit tests).

## 9. Cross-split ID overlap audit

| Check | Result |
|---|---|
| Shared `scenario_id` across splits | **0** |
| Shared `event_id` across splits | **0** |
| Shared `memory_id` across splits | **0** |

All five splits are fully disjoint in scenario, event, and memory identifiers, so
there is no cross-split contamination between public splits or into the private
hidden split.

## 10. Realistic split status

`realistic_200_en` is **`synthetic_gold_unreviewed`**. Its gold is synthetic and
has **not** been validated by human annotators. Per the AAAI plan it is included
only as a secondary / stress-style split with a limitation note, and it must not
be used as a headline result until real human validation is completed (see
`human_validation_protocol.md` / `human_validation_status.md`). The validator
emits one warning per realistic row to make this status impossible to overlook.

## 11. Reproduction

```bash
python scripts/generate_retrace_bench_final.py --seed 2027 --out data/retrace_bench_v1_1
for s in main_3000_en hard_500_en realistic_200_en calibration_80_en private_hidden_200_en; do
  python scripts/validate_retrace_bench_dataset.py --data data/retrace_bench_v1_1/$s/scenarios.jsonl
done
```
