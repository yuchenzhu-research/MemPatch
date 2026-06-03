# Stage C API-ICL: exemplar contract and status

Stage C is `ReTrace-Learn`: it still proposes **typed actions only**,
and the RevisionGate + deterministic DPA / `authorize(...)` path is unchanged.
API-ICL is the first Stage C variant — it conditions the proposer on a few
in-context exemplars of well-formed typed revisions. Only the proposer changes.

```
Stage C API-ICL proposer (k approved exemplars in context)
    -> RevisionGate -> deterministic DPA / authorize(...) -> commit result
```

## Current status (honest)

- **Offline replay / mock**: working (`scripts/evaluate.py stage-c --smoke`, or the
  Stage C runner with `--generations-dir`). Decoded generations are replayed
  through the canonical parser + shared DPA path.
- **API-ICL**: **not wired to a turnkey live CLI** and **fail-closed**. It cannot
  run without a *human-approved* exemplar pack. This is intentional per
  `AGENTS.md` (ReTrace-Learn Paper Training Boundary): no development-candidate
  episode may be promoted for smoke/training without a recorded human decision,
  and no run may see gold typed targets or evaluator final statuses.
- **Hosted-FT / Open LoRA-SFT**: offline data-prep / example configs only; not
  claimed as runnable here.

The exemplar loader/validator/selector in
`src/retracemem/evaluation/multiagent/stage_c_icl.py` is implemented and tested;
the remaining next step is to wire an approved pack + provider into a proposer
that emits typed actions for live ICL evaluation against Stage A ZeroShot on the
same dev cases.

## Required exemplar pack format

A pack is a single JSON file. It is accepted by `load_approved_exemplars()` only
when **all** of the following hold (otherwise it fails closed):

1. `source_manifest_sha256` is present (hash of the immutable review-pack manifest).
2. `approval.decision == "approved"`, with `approval.reviewer` and
   `approval.reviewed_at` recorded.
3. Every exemplar contains only method-visible fields (no gold / evaluator fields).
4. Every action, including `NO_REVISION`, cites visible new evidence via
   non-empty `evidence_ids`.

```json
{
  "pack_id": "stage_c_icl_<name>",
  "split": "development_only",
  "source_manifest_sha256": "<sha256 of the immutable review-pack manifest>",
  "approval": {
    "decision": "approved",
    "reviewer": "<human reviewer id>",
    "reviewed_at": "<ISO-8601 timestamp>"
  },
  "exemplars": [
    {
      "exemplar_id": "ex_supersede_001",
      "failure_category": "missed_SUPERSEDES",
      "candidate_view_summary": "<bounded, method-visible candidate context>",
      "submission_evidence": [
        {"evidence_id": "ev_2", "content": "<new evidence text>"}
      ],
      "proposed_actions": [
        {
          "action_type": "SUPERSEDES",
          "target_belief_id": "b_cfg_1",
          "replacement_belief_id": "b_cfg_2",
          "target_condition_id": null,
          "evidence_ids": ["ev_2"],
          "rationale": "<why, citing the new evidence>"
        }
      ]
    }
  ]
}
```

### Forbidden fields (hard fail / anti-leakage)

The loader recursively rejects any of these keys anywhere in the pack:
`gold_final_status(es)`, `final_belief_statuses`, `gold_typed_targets`,
`gold_revision_targets`, `evaluator_status`, `gold_snapshot`, `belief_statuses`,
`relevant_session_index`, `m_old`, `m_new`.

### Canonical action vocabulary

`SUPERSEDES`, `BLOCKS`, `RELEASES`, `UNCERTAIN`, `REAFFIRMS`, `NO_REVISION`.

## Example template

`fixtures/stage_c_exemplars/EXAMPLE_pending_template.json` is a **synthetic,
non-approved** template (`approval.decision = "pending"`). It demonstrates the
schema and deliberately fails closed if passed to `load_approved_exemplars()`.
Do not use it for live runs; replace it with real review-pack-derived,
human-approved content first.

## Do not

- Do not auto-promote pending packs to approved.
- Do not source exemplars from the eval/test split or from gold labels.
- Do not change RevisionGate or DPA for Stage C — only the proposer adapts.
