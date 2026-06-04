# ReTrace-Bench paper-grade human validation packet (200)

**Do not open the `_gold.jsonl` file before annotating.** Annotators work
only from `paper_validation_200_public.jsonl` (gold-free) and record answers
in `paper_validation_200_sheet.csv` (one row per `scenario_id`).

## Files
- `paper_validation_200_public.jsonl` — gold-free scenarios (annotator input).
- `paper_validation_200_sheet.csv` — empty entry sheet, one row per scenario.
- `paper_validation_200_gold.jsonl` — hidden gold, **scoring lead only**.

## Procedure
1. Read `docs/retrace_bench/human_annotation_codebook.md` first.
2. At least two independent human annotators each fill their own copy of the sheet.
3. Set a unique `annotator_id` per person (LLMs may NOT be annotators).
4. Adjudicate disagreements, then run `scripts/score_human_annotations.py`.

## Coverage of this packet
- scenarios: 200
- splits: {'calibration': 10, 'hard': 90, 'main': 80, 'realistic': 20}
- decision labels: {'use_current_memory': 149, 'ask_clarification': 12, 'escalate': 8, 'mark_unresolved': 21, 'refuse_due_to_policy': 10}
- patterns covered: 15/15
- difficulties: {'L2': 33, 'L3': 80, 'L4': 72, 'L1': 15}
- distinct domains: 8
- hard rows: 90 (target >= 50)
- non-answer gold rows: 51 (target >= 40)
