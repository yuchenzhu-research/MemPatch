---
license: cc-by-4.0
language:
- en
pretty_name: MemPatch
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- rapid-memory-integration
- memory-revision
- evaluation
configs:
- config_name: default
  data_files:
  - split: main
    path: main/scenarios.jsonl
  - split: hard
    path: hard/scenarios.jsonl
---

# MemPatch

MemPatch supports **Rapid Memory Integration (RMI)** evaluation for the paper
**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM
Agents**. Each scenario tests whether an LLM agent integrates new evidence into
`memory_state` labels (`current`, `outdated`, `blocked`, `unresolved`, etc.)
without stale reuse, scope leakage, or policy-invalid beliefs.

The released data has two public splits: `main` and `hard`. The stress-only
`realistic` rows and smoke-only `calibration` rows are not part of the public HF
release used for paper-facing evaluation.

> **Evaluation-only.** Do not train the MemPatch Revision Module policy on this
> data. Doing so contaminates benchmark results.

## Public Rows

| public_split_name | rows | purpose |
|---|---:|---|
| `main` | 3000 | broad coverage across domains, difficulties, failure modes |
| `hard` | 500 | L3/L4 adversarial; minimal-evidence, no latest-event shortcut |

**Public total: 3500 rows.**

- `realistic` remains a secondary stress subset until human validation is
  recorded and is not included in this HF release.
- `calibration` remains smoke / quickstart only and is not included in this HF
  release.
- Private hidden rows are not part of this public release.

## Format

Each line in `main/scenarios.jsonl` or `hard/scenarios.jsonl` is a JSON scenario
object with a gold-free `public_input` and a `hidden_gold` block used by the
official scorer. The public-facing model input must be taken through the
official public view; do not feed `hidden_gold` or internal fields to a model.

The benchmark-compatible prediction interface is:

```json
{
  "scenario_id": "case-000001",
  "response": {
    "decision": "use_current_memory",
    "memory_state": {"m1": "current", "m2": "outdated"},
    "evidence_event_ids": ["e2", "e5"],
    "failure_diagnosis": "stale_memory_reuse",
    "answer": "..."
  }
}
```

## Scoring

Use the official evaluator from the GitHub repository:

```bash
python scripts/evaluate_retrace_bench_predictions.py \
  --data main/scenarios.jsonl --predictions <your_predictions>.jsonl
```

Core metrics: `decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`,
`minimal_evidence_exact_match`, `failure_diagnosis_accuracy`,
`joint_revision_success`, `stale_reuse_rate`, `format_failure_rate`.

## Licensing

- **Dataset:** CC BY 4.0 (see `DATASET_LICENSE.md`).
- **Code:** MIT, hosted at https://github.com/yuchenzhu-research/MemPatch.

## Provenance

- Deterministically generated with seed `2027`.
- Release version `1.1.0`.
- Code, schema, validators, and benchmark runner:
  https://github.com/yuchenzhu-research/MemPatch.

## Citation

```bibtex
@misc{mempatch2026,
  title  = {MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents},
  author = {MemPatch authors},
  year   = {2026},
  note   = {Evaluation-only benchmark release. \url{https://github.com/yuchenzhu-research/MemPatch}}
}
```
