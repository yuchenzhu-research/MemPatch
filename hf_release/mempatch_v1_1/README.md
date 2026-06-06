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

MemPatch supports **Rapid Memory Integration (RMI)** evaluation. Each scenario tests whether an LLM agent integrates new evidence into `memory_state` labels (`current`, `outdated`, `blocked`, `unresolved`, etc.) without stale reuse, scope leakage, or policy-invalid beliefs.

The public release contains two splits only: `main` and `hard`. Stress-only `realistic` and smoke-only `calibration` rows are **not** part of this public release.

> **Evaluation-only.** Do not train the MemPatch Revision Module policy on this data.

## Public Rows

| public_split_name | rows | purpose |
|---|---:|---|
| `main` | 3000 | broad coverage across domains, difficulties, failure modes |
| `hard` | 500 | L3/L4 adversarial; minimal-evidence, no latest-event shortcut |

**Public total: 3500 rows.**

There is **no train split** in this release.

## Format

Each line in `main/scenarios.jsonl` or `hard/scenarios.jsonl` is a JSON scenario with gold-free `public_input` and scorer-only `hidden_gold`.

Prediction interface:

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

Use the official evaluator script from the anonymous artifact repository:

```bash
python scripts/evaluate_mempatch_predictions.py \
  --data main/scenarios.jsonl --predictions <your_predictions>.jsonl
```

Core metrics: `decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`, `joint_revision_success`, `stale_reuse_rate`.

## Licensing

- **Dataset:** CC BY 4.0 (see `DATASET_LICENSE.md`).
- **Code:** MIT (anonymous artifact repository, blind review).

## Citation

```bibtex
@misc{mempatch2026,
  title  = {MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Evaluation-only benchmark release.}
}
```
