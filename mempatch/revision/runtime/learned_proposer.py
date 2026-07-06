"""Revision Response Policy — MemPatch Revision Module Step 2.

``r_raw ← πθ(V)``: given revision view ``V``, produce a raw benchmark-compatible
revision response. Typed patch actions map to ``response.decision``,
``response.memory_state``, ``response.evidence_event_ids``, and
``response.failure_diagnosis`` after DPA-Consistent Projection (Step 4).

* :class:`LearnedTypedRevisionProposer` wraps a text ``generate_fn`` and parses
  its completion into validated actions via the shared fail-closed parser.
* :class:`ScriptedProposer` replays a fixed action list (smoke test / oracle).

Both return a :class:`ProposalOutput` carrying the raw completion, parse result,
and validated actions for the runtime kernel and benchmark-grounded feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from mempatch.dpa.methods.contracts import SharedCandidateView

from mempatch.revision.schemas import RevisionAction
from mempatch.revision.runtime.dpa_runtime import ParseResult, parse_actions

GENERATE_FN = Callable[[str], str]


class TypedRevisionProposer(Protocol):
    policy_variant: str

    def propose(
        self,
        view: SharedCandidateView,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalOutput:
        ...


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


@dataclass(frozen=True)
class ProposalOutput:
    raw_text: str
    parse_result: ParseResult

    @property
    def actions(self) -> tuple[RevisionAction, ...]:
        return self.parse_result.actions

    @property
    def parsing_valid(self) -> bool:
        return self.parse_result.valid_json and self.parse_result.schema_valid


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


def actions_to_json(actions: list[RevisionAction]) -> str:
    """Render actions as the canonical JSON-array completion."""
    return json.dumps([a.to_dict() for a in actions], ensure_ascii=False)


class LearnedTypedRevisionProposer:
    """Prompt a text model and parse its completion into typed actions."""

    policy_variant = "learned_prompt"

    def __init__(self, generate_fn: GENERATE_FN) -> None:
        self._generate = generate_fn

    def propose(
        self,
        view: SharedCandidateView,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalOutput:
        prompt = build_proposer_prompt(view)
        raw_text = self._generate(prompt)
        return ProposalOutput(raw_text=raw_text, parse_result=parse_actions(raw_text))


class ScriptedProposer:
    """Replay a fixed list of actions (smoke test / oracle teacher forcing)."""

    policy_variant = "scripted"

    def __init__(self, actions: list[RevisionAction]) -> None:
        self._actions = list(actions)

    def propose(
        self,
        view: SharedCandidateView,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalOutput:
        raw_text = actions_to_json(self._actions)
        return ProposalOutput(raw_text=raw_text, parse_result=parse_actions(raw_text))


class PromptProposer:
    """Prompt-based proposer backed by a text ``generate_fn``."""

    policy_variant = "prompt_proposer"

    def __init__(self, generate_fn: GENERATE_FN) -> None:
        self._proposer = LearnedTypedRevisionProposer(generate_fn)

    def propose(
        self,
        view: SharedCandidateView,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalOutput:
        return self._proposer.propose(view, metadata=metadata)
