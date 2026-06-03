# ReTrace-Bench `main_3000_en` (v1.0.0)

Primary controlled benchmark split of ReTrace-Bench v1.0 (public split name:
**`main`**). It provides broad coverage across the 8 domains, 11 memory-revision
failure modes, 5 revision decisions, and 4 difficulty tiers, and is used for the
main benchmark results.

- **Scenarios:** 3000
- **Source type:** `controlled_synthetic`
- **Annotation status:** `synthetic_gold` (controlled synthetic gold)
- **Benchmark version:** `1.0.0`
- **Schema:** `retrace_bench_general_1`
- **Training targets:** none (evaluation-only; `hidden_gold` is evaluation gold)

## Benchmark hygiene / leakage audit

Authoritative (verified/trusted) records are **de-actionalized**: each states a
fact or status and never begins with a final action verb (`Escalate…`,
`Refuse…`, `Ask for clarification…`, `Mark … unresolved`, `Use current
memory`). The gold decision must be recovered by reasoning over the described
state, not copied from a word.

Decision-word leakage audit over authoritative records:
`scenarios_with_decision_word_leak = 0`
(`clean = true`).

## Scale

- avg / max events per scenario: 7.965 / 10
- avg / max memories per scenario: 3.424 / 5
- avg required evidence events per scenario: 1.0

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_main_3000.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl
```
