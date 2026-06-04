# ReTrace-Bench `realistic_100_en` (v1.0.0)

Realistic-style workflow split of ReTrace-Bench v1.0 (public split name:
**`realistic`**).

> **Annotation is NOT done in this pass.** These scenarios are
> `realistic_style_synthetic` (generated from realistic workflow patterns, not
> collected from public sources). `annotation_status` is **`pending`**.
> `hidden_gold` fields are intentionally empty — there is **no synthetic gold
> and no fabricated human annotation**. Use `annotations_template.jsonl` for the
> human annotation pass.

- **Scenarios:** 100
- **Events per scenario:** 15-50 (avg 31.46)
- **Memories per scenario:** 4-8
- **Source type:** `realistic_style_synthetic`
- **Annotation status:** `pending`
- **Benchmark version:** `1.0.0`

## Category mix

- `software_engineering_agent`: 40
- `customer_support_crm`: 20
- `research_knowledge_work`: 15
- `calendar_task_workflow`: 15
- `enterprise_multi_tool_workflow`: 10

## Files

- `scenarios.jsonl` — realistic-style scenarios (de-actionalized, no gold).
- `annotations_template.jsonl` — one empty row per scenario for human annotation.
- `manifest.json` — split manifest + leakage audit summary.

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_realistic_100.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/realistic_100_en/scenarios.jsonl
```
