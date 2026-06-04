# ReTrace-Bench `hard_300_en` (v1.0.0)

Rule-defined long-context / multi-evidence / multi-memory stress split of
ReTrace-Bench v1.0 (public split name: **`hard`**). It pressures structured
memory revision beyond coarse decision accuracy. Difficulty is defined by
deterministic structural rules, **not** by cherry-picking model failures.

- **Scenarios:** 300
- **Events per scenario:** 20-100 (avg 52.157)
- **Memories per scenario:** >= 5 (avg 6.0)
- **Required evidence events per scenario:** >= 2 (avg 2.453)
- **Source type:** `controlled_synthetic`
- **Annotation status:** `synthetic_gold`
- **Benchmark version:** `1.0.0`

## Hard criteria

Each case satisfies at least three of: >=2 evidence events; >=3 changed memory
states; block/release/restore lifecycle; policy/consent/scope boundary;
non-answer decision; delayed contradiction; no single authoritative-event
shortcut. Cross-scope distractors are present but **not universal**.

## Benchmark hygiene / leakage audit

Authoritative (verified/trusted) records are de-actionalized; the gold decision
must be inferred from the described state. Decision-word leakage audit:
`scenarios_with_decision_word_leak = 0`
(`clean = true`).

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_hard_300.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/hard_300_en/scenarios.jsonl
```
