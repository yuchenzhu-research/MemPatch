"""Prompt construction for the typed revision proposal."""
from __future__ import annotations

import json
from typing import Any

from mempatch.dpa.methods.contracts import SharedCandidateView


CANONICAL_ACTION_HELP = (
    "Preferred final action_type values and required fields:\n"
    "- BLOCKS: target_condition_id + evidence_ids\n"
    "- UNCERTAIN: target_belief_id + evidence_ids\n"
    "- REAFFIRMS: target_belief_id + evidence_ids\n"
    "- NO_REVISION: all target/replacement ids null (still cite new evidence)\n"
    "Restricted action_type values:\n"
    "- SUPERSEDES: target_belief_id + replacement_belief_id + evidence_ids\n"
    "- RELEASES: target_condition_id + evidence_ids\n"
)


def _view_payload(view: SharedCandidateView) -> dict[str, Any]:
    conditions = {
        bid: [{"condition_id": c.condition_id, "text": c.text} for c in conds]
        for bid, conds in view.candidate_conditions_by_belief
    }
    deps = {
        bid: [
            {"belief_id": d.belief_id, "condition_id": d.condition_id}
            for d in edges
        ]
        for bid, edges in view.dependency_edges_by_belief
    }
    return {
        "query": view.query,
        "as_of_evidence_id": view.new_evidence.evidence_id,
        "evidence_context": [
            {
                "evidence_id": evidence.evidence_id,
                "timestamp": evidence.timestamp,
                "text": evidence.text,
            }
            for evidence in view.evidence_context
        ],
        "new_evidence": {
            "evidence_id": view.new_evidence.evidence_id,
            "text": view.new_evidence.text,
        },
        "candidate_beliefs": [
            {
                "belief_id": b.belief_id,
                "proposition": b.proposition,
                "source_evidence_ids": list(b.source_evidence_ids),
            }
            for b in view.candidate_beliefs
        ],
        "candidate_replacement_beliefs": [
            {"belief_id": b.belief_id, "proposition": b.proposition}
            for b in view.candidate_replacement_beliefs
        ],
        "candidate_conditions_by_belief": conditions,
        "dependency_edges_by_belief": deps,
    }


def build_proposer_prompt(view: SharedCandidateView) -> str:
    payload = _view_payload(view)
    return (
        "You are the MemPatch Revision Module Response Policy (Step 2). Given the "
        "revision view and evidence ledger up to as_of_evidence_id, output ONLY a "
        "JSON array of typed patch "
        "actions forming r_raw for DPA-Consistent Projection into a benchmark "
        "response (decision, memory_state, evidence_event_ids, failure_diagnosis).\n\n"
        f"{CANONICAL_ACTION_HELP}\n"
        "Use BLOCKS, UNCERTAIN, REAFFIRMS, or NO_REVISION unless the view "
        "explicitly supports a restricted action. Do not emit SUPERSEDES unless "
        "candidate_replacement_beliefs contains a valid replacement_belief_id; "
        "copy that belief_id exactly from the view. Do not emit RELEASES unless "
        "the view contains an explicit release target. Invalid or missing IDs "
        "make the entire action array fail closed.\n"
        "Every evidence_id must be copied exactly from evidence_context. Use the "
        "minimal supporting evidence set; never invent an ID.\n"
        "Each action object has keys: action_type, target_belief_id, "
        "target_condition_id, replacement_belief_id, evidence_ids, rationale.\n\n"
        f"CONTEXT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\nACTIONS_JSON:"
    )
