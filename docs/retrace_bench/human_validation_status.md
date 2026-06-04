# ReTrace-Bench — Human Validation Status

**Current status: `automatic validation complete; human annotation pending`**

Validation levels (see `human_validation_protocol.md`):
- **Level 1 — Automatic validation:** **COMPLETE** on the full benchmark (schema,
  leakage, ID grounding, cross-split disjointness, gold-oracle replay = 1.0).
- **Level 2 — Author / expert audit:** **PENDING** (no audit recorded).
- **Level 3 — Independent human validation:** **PENDING** (no real human
  annotators have completed the 200-packet).
- **Level 4 — Human upper bound:** **PENDING** (optional).

No real human annotations have been collected in this cleanup pass. The full
validation package (protocol, codebook, plan, packets, export + scoring scripts)
is prepared and tested, but **no human has annotated the data yet**, so **no
human validation results, IAA, or human upper bound may be cited**. AI/LLM
outputs have not been and must not be used as a substitute for human annotation.

## What exists now
- Protocol: `human_validation_protocol.md`
- Codebook: `human_annotation_codebook.md`
- Plan: `human_validation_plan.md`
- Packets (seed 2027): `annotation_packets/retrace_bench_v1_1/`
  - `quick_audit_50_public.jsonl` / `quick_audit_50_gold.jsonl`
  - `paper_validation_200_public.jsonl` / `paper_validation_200_gold.jsonl`
  - `paper_validation_200_sheet.csv` / `paper_validation_200_readme.md`
- Export script: `scripts/export_human_annotation_packet.py`
- Scoring script: `scripts/score_human_annotations.py`
- Empty results template: `human_validation_results_template.md`

## What is NOT done
- No human annotators have completed the sheet.
- No inter-annotator agreement has been computed on real data.
- No human upper bound exists yet.
- LLM/AI outputs have **not** been used as a substitute for human annotation and
  must not be.

## How to advance the status
1. Have ≥ 2 real humans complete `paper_validation_200_sheet.csv` independently.
2. Run `scripts/score_human_annotations.py --annotations <fileA> <fileB>`.
3. The script will set the status to `quick author audit completed` (1 annotator)
   or `paper-grade human validation completed` (≥ 2 annotators) and write the
   real results. Update this file to match.
