# ReTrace-Bench — Data Construction Statement

This statement describes how ReTrace-Bench is constructed and validated. It is
intended for the dataset card, the resource paper, and any public-facing
documentation.

## Summary

ReTrace-Bench is a **real-world-inspired, pattern-driven, controlled-synthetic**
benchmark for evaluating memory-revision reasoning. It is built with a
**deterministic generation pipeline** under a **fixed public seed (2027)** from a
library of **workflow-level memory-revision patterns**, and it is validated with
schema checks, leakage checks, cross-split ID audits, and gold-oracle replay.

> ReTrace-Bench is constructed with a deterministic, pattern-driven generation
> pipeline. A fixed public seed controls scenario instantiation and split
> construction, enabling exact reproduction of the released splits. The seed is
> not used as a source of stochastic model evaluation; it only fixes the dataset
> construction process.

## What "deterministic, pattern-driven, controlled synthetic" means

- **Pattern-driven, not random.** Scenarios are instantiated from a curated set of
  workflow-level memory-revision patterns (e.g. a later message supersedes an
  earlier decision, a prerequisite is blocked then released, an instruction is
  reaffirmed, evidence is genuinely ambiguous). Each pattern carries a
  specification for the correct decision, the resulting memory state, the
  grounding evidence IDs, and the failure-mode diagnosis. The benchmark is **not**
  "randomly generated"; the seed only fixes which surface realizations of these
  patterns are emitted and how they are assigned to splits.
- **Controlled synthetic construction is necessary.** ReTrace-Bench evaluates
  *structured* memory-revision labels — the decision, the post-revision memory
  state, the supporting evidence IDs, and the failure diagnosis. Obtaining
  reliable hidden gold for all four of these jointly requires controlled
  construction; it is not something that can be read off raw text at scale.
- **Real user logs are not used.** Two reasons: (1) privacy — real multi-agent /
  user logs cannot be released openly; and (2) hidden gold for the structured
  revision labels above would be extremely hard to obtain and verify on
  uncontrolled logs. The patterns are *inspired by* real workflow memory-revision
  behavior, but the released instances are synthetic.
- **Deterministic and reproducible.** With seed 2027 and the frozen generation
  code, the released splits (`main` 3000, `hard` 500, `realistic` 200,
  `calibration` 80) reproduce exactly. The seed governs **dataset construction
  only**; it is never used as a knob for stochastic model evaluation. Model
  evaluation randomness (decoding temperature, sampling) is a separate concern
  documented in `statistical_reporting_note.md`.

## Validation performed at construction time (automatic)

- **Schema validation** — every scenario conforms to the published schema.
- **Leakage checks** — hidden gold fields are never exposed in the public view;
  evaluation inputs contain no gold labels.
- **Cross-split ID audits** — split membership is disjoint; IDs are grounded and
  unique; no scenario appears in more than one canonical split.
- **Gold-oracle replay** — feeding the hidden gold back as a "perfect prediction"
  yields core metrics = 1.0 and format-failure rate = 0.0, confirming the
  scorer and the gold are mutually consistent.

These automatic checks are run on the full benchmark. They are **necessary but not
sufficient**: they establish internal consistency, not human-judged label
quality. Human validation is a separate, still-pending step (see
`human_validation_protocol.md`, `human_validation_status.md`).

## Per-split construction status

- **`main` (3000)** — canonical evaluation split; deterministic; automatically
  validated.
- **`hard` (500)** — difficulty-balanced canonical split; deterministic;
  automatically validated; distribution targets met (15 patterns covered, max
  single-pattern share < 25%, L3/L4 balanced).
- **`realistic` (200)** — more naturalistic surface realizations. **Status:
  `synthetic_gold_unreviewed`** — the gold for this split has passed automatic
  checks but has **not** yet been confirmed by independent human validation. It
  must be reported as `synthetic_gold_unreviewed` until human validation is
  completed.
- **`calibration` (80)** — small **smoke-only** split for wiring/format checks,
  not a headline evaluation split.
- **`private_hidden` (200)** — held-out contamination probe; **never published**
  to Hugging Face or committed to GitHub; local/private only.

## Reproduction

```bash
# Regenerate the canonical splits deterministically (seed 2027), then validate:
python scripts/generate_retrace_bench_final.py      # frozen generation logic
for s in main hard realistic calibration; do
  python scripts/validate_retrace_bench_dataset.py \
    --data data/retrace_bench_v1_1/${s}_*_en/scenarios.jsonl
done
python scripts/check_retrace_bench_gold_oracle.py \
  --data data/retrace_bench_v1_1/hard_500_en/scenarios.jsonl --out /tmp/oracle.json
```
