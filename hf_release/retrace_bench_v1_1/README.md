---
license: cc-by-4.0
language:
- en
pretty_name: ReTrace-Bench
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- benchmark
- memory-revision
- evaluation
configs:
- config_name: main
  data_files: main/scenarios.jsonl
- config_name: hard
  data_files: hard/scenarios.jsonl
- config_name: realistic
  data_files: realistic/scenarios.jsonl
- config_name: calibration
  data_files: calibration/scenarios.jsonl
---

# ReTrace-Bench

ReTrace-Bench evaluates **agent memory-revision reliability**: whether a system
processes new evidence to update, block, release, reaffirm, or reject memory
states without introducing stale, out-of-scope, or policy-invalid beliefs. It
scores not only the final decision but also memory-state tracking, minimal
evidence grounding, and failure diagnosis.

> **Evaluation-only.** ReTrace-Bench is a benchmark for *evaluating* systems.
> Do **not** train models on it, and in particular do **not** train the separate
> *ReTrace-Learn* method on this data. Doing so invalidates results.

## Splits (public release)

| split | cases | purpose |
|---|---|---|
| `main` | 3000 | broad coverage across domains, difficulties, failure modes |
| `hard` | 500 | L3/L4 adversarial; minimal-evidence, no latest-event shortcut |
| `realistic` | 200 | realistic-style stress split; **`synthetic_gold_unreviewed`** |
| `calibration` | 80 | smoke / quickstart only |

**Public total: 3780 cases.**

- `realistic` is `synthetic_gold_unreviewed`: its gold has **not** been human
  reviewed yet. Treat it as a secondary/stress split with a limitation note, not
  a headline split, until human validation is recorded.
- `calibration` is **smoke / quickstart only** — it is not a model-selection /
  checkpoint-selection validation set and must not be used to tune or select
  systems, nor for headline claims.
- A `private_hidden` split (200 cases) exists for private evaluation and is **not
  part of this public release**.

## Format

Each line is a JSON scenario object (the native format consumed by the official
evaluator). It includes a gold-free public input plus the `hidden_gold` block
used for scoring. The public-facing model input must be taken through the
official public view; do not feed `hidden_gold` or internal fields to a model.

## Scoring

Use the official evaluator from the GitHub repository
([https://github.com/yuchenzhu-research/ReTrace](https://github.com/yuchenzhu-research/ReTrace)):

```bash
python scripts/evaluate_retrace_bench_predictions.py \
  --data <split>/scenarios.jsonl --predictions <your_predictions>.jsonl
```

Core metrics: `decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`,
`minimal_evidence_exact_match`, `failure_diagnosis_accuracy`,
`joint_revision_success`, `format_failure_rate`.

## Licensing

- **Dataset:** CC BY 4.0 (see `DATASET_LICENSE.md`).
- **Code** (evaluator/validators/scripts on GitHub): MIT. The code and dataset
  licenses are separate.

## Provenance

- Deterministically generated with seed `2027`.
- Release version `1.1.0`.
- Code, schema, validators, and reproducible baselines: [https://github.com/yuchenzhu-research/ReTrace](https://github.com/yuchenzhu-research/ReTrace).

## Citation

```bibtex
@misc{retracebench,
  title  = {ReTrace-Bench: Evaluating Agent Memory-Revision Reliability},
  author = {ReTrace authors},
  year   = {2026},
  note   = {Evaluation-only benchmark. \url{https://github.com/yuchenzhu-research/ReTrace}}
}
```
