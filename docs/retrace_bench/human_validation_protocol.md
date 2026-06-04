# ReTrace-Bench — Human Validation Protocol

This protocol governs human validation of ReTrace-Bench (internal "v1.1") for the
AAAI benchmark/resource paper. It defines **what** humans annotate, **how**
agreement is measured, and the **integrity rules** that keep the validation
credible.

> **Integrity (non-negotiable):** AI assistants / LLMs (ChatGPT, Claude, Gemini,
> etc.) may **not** be represented as human annotators. LLM-based checking, if
> done, is labelled *LLM pilot validation*, never *human validation*. Do not
> fabricate inter-annotator agreement. If no real humans have annotated yet, the
> status is "protocol prepared; human annotation pending".

## Two validation levels

### Level 1 — Quick author audit (internal)
- **Packet:** `annotation_packets/retrace_bench_v1_1/quick_audit_50_*`
- **Size:** 50 examples, stratified across splits, all 15 patterns, and
  non-answer decisions.
- **Purpose:** fast internal quality signal during development.
- **Limitation:** **not** sufficient to claim full human validation in the paper.

### Level 2 — Paper-grade human validation
- **Packet:** `annotation_packets/retrace_bench_v1_1/paper_validation_200_*`
- **Size:** 200 examples (current packet: ≥90 hard, ≥40 non-answer, all 5
  decisions, all 15 patterns, 8 domains, L1–L4 with L3/L4 emphasis).
- **Annotators:** **at least 2 independent real humans**, preferably including
  someone who is not the dataset author.
- **Adjudication:** a third pass resolves disagreements; record adjudicated
  labels separately.
- **Reported:** inter-annotator agreement (IAA) and human upper-bound metrics.

## Sampling strategy (paper-grade)

The export script (`scripts/export_human_annotation_packet.py`, seed 2027)
enforces, from the public splits only (`main`/`hard`/`realistic`/`calibration` —
**never** `private_hidden`):

- prioritize `main` + `hard`;
- ≥ 50 hard examples;
- ≥ 40 non-answer gold decisions;
- all five decision labels (`use_current_memory`, `mark_unresolved`,
  `ask_clarification`, `refuse_due_to_policy`, `escalate`);
- all 15 workflow patterns;
- multiple domains; both L3 and L4.

Realized coverage is printed by the export script and echoed in
`paper_validation_200_readme.md`.

## What annotators see / do not see

The public packet (`*_public.jsonl`) is produced through
`benchmark.retrace_bench.public_view` and contains only:
`scenario_id`, `split`, `domain`, `difficulty`, `workflow_context`,
`public_input.event_trace`, `public_input.initial_memory`, the four task views
(`black_box_task`, `memory_state_task`, `evidence_retrieval_task`,
`diagnostic_task`), and the `allowed_labels` spaces.

It must **never** contain: `hidden_gold`, `expected_*`, `metadata`,
`validation_notes`, `source_pointers`, `is_distractor`, `primary_failure_mode`,
`pattern_trap_type`, `canonical_failure_mode`. The export script self-checks this
and refuses to write a packet that leaks any of them.

## Annotation fields

Each annotator fills one row per scenario in `paper_validation_200_sheet.csv`:

1. `annotator_id` — unique per human (LLMs prohibited).
2. `scenario_id`
3. `solvable_from_visible_evidence` — yes / no / uncertain
4. `topic_domain_consistent` — yes / no / uncertain
5. `ambiguous_or_multiple_valid_answers` — yes / no / uncertain
6. `filler_heavy` — yes / no / uncertain
7. `decision_label` — one of the five canonical decisions
8. `answer_short_free_text`
9. `memory_state_json` — `{memory_id: status}` over the canonical statuses
   (`current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, `deleted`,
   `should_not_store`, `restored`)
10. `evidence_event_ids` — minimal supporting event IDs
11. `failure_diagnosis` — one canonical failure-diagnosis label
12. `confidence` — 1–5
13. `notes`

Full definitions are in `human_annotation_codebook.md`.

## Procedure

1. Each annotator reads `human_annotation_codebook.md`.
2. Each annotator independently fills their own copy of the sheet from the
   gold-free public packet (do **not** open `*_gold.jsonl`).
3. The scoring lead runs:
   ```bash
   python scripts/score_human_annotations.py \
     --annotations path/to/alice.csv path/to/bob.csv \
     --gold annotation_packets/retrace_bench_v1_1/paper_validation_200_gold.jsonl
   ```
4. Disagreements are adjudicated; optionally re-score with the adjudicated file.
5. Update `human_validation_status.md` to the achieved level.

## Metrics computed

**Dataset quality:** solvable rate, topic/domain-consistency rate, ambiguity
rate, filler-heavy rate.

**Human vs gold** (via the official scorer): decision accuracy,
memory_state_accuracy, evidence_f1, minimal_evidence_exact_match,
failure_diagnosis_accuracy, joint_revision_success, plus per-pattern and
per-difficulty joint breakdowns and a pooled **human upper bound**.

**Inter-annotator agreement:**
- Cohen's kappa (pairwise) for `decision_label`, `failure_diagnosis`,
  `solvable_from_visible_evidence`, `topic_domain_consistent`,
  `ambiguous_or_multiple_valid_answers`;
- Krippendorff's alpha (nominal) when 3+ annotators are present;
- pairwise evidence F1 / Jaccard for `evidence_event_ids` (not kappa);
- per-memory and macro agreement for `memory_state_json`.

## Outputs

- `outputs/retrace_bench_v1_1/human_validation/human_validation_results.json`
- `outputs/retrace_bench_v1_1/human_validation/human_validation_results.md`
- `docs/retrace_bench/human_validation_results_template.md` (mirror; empty
  template until real annotations exist)
