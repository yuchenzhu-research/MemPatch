"""Module 2: Typed Revision Proposer.

Given a current memory graph + new evidence + candidate beliefs / replacements /
condition anchors, propose typed revision actions from the canonical vocabulary.

* :class:`LearnedTypedRevisionProposer` wraps a text ``generate_fn`` (a trained
  2B/4B model) and parses its completion into validated actions via the shared
  fail-closed parser. The model is external; this owns prompt assembly + parsing.
* :class:`ScriptedProposer` replays a fixed action list (used for the smoke test
  and for oracle/teacher-forcing rollouts).

Both return a :class:`ProposalOutput` carrying the raw completion, the parse
result, and the validated actions, so it can feed straight into the runtime and
the reward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from retracemem.methods.contracts import SharedCandidateView

from retrace_learn.schemas import RevisionAction
from retrace_learn.runtime.dpa_runtime import ParseResult, parse_actions

GENERATE_FN = Callable[[str], str]

CANONICAL_ACTION_HELP = (
    "Allowed action_type values and required fields:\n"
    "- SUPERSEDES: target_belief_id + replacement_belief_id + evidence_ids\n"
    "- BLOCKS: target_condition_id + evidence_ids\n"
    "- RELEASES: target_condition_id + evidence_ids\n"
    "- UNCERTAIN: target_belief_id + evidence_ids\n"
    "- REAFFIRMS: target_belief_id + evidence_ids\n"
    "- NO_REVISION: all target/replacement ids null (still cite new evidence)\n"
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
        "new_evidence": {
            "evidence_id": view.new_evidence.evidence_id,
            "text": view.new_evidence.text,
        },
        "candidate_beliefs": [
            {"belief_id": b.belief_id, "proposition": b.proposition}
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
        "You are the ReTrace-Learn typed revision proposer. Given the memory "
        "graph context and the new evidence, output ONLY a JSON array of typed "
        "revision actions.\n\n"
        f"{CANONICAL_ACTION_HELP}\n"
        "Each action object has keys: action_type, target_belief_id, "
        "target_condition_id, replacement_belief_id, evidence_ids, rationale.\n\n"
        f"CONTEXT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\nACTIONS_JSON:"
    )


def actions_to_json(actions: list[RevisionAction]) -> str:
    """Render actions as the canonical JSON-array completion (SFT target text)."""
    return json.dumps([a.to_dict() for a in actions], ensure_ascii=False)


class LearnedTypedRevisionProposer:
    """Prompt a text model and parse its completion into typed actions."""

    policy_variant = "learned_lora"

    def __init__(self, generate_fn: GENERATE_FN) -> None:
        self._generate = generate_fn

    def propose(self, view: SharedCandidateView) -> ProposalOutput:
        prompt = build_proposer_prompt(view)
        raw_text = self._generate(prompt)
        return ProposalOutput(raw_text=raw_text, parse_result=parse_actions(raw_text))


class ScriptedProposer:
    """Replay a fixed list of actions (smoke test / oracle teacher forcing)."""

    policy_variant = "scripted"

    def __init__(self, actions: list[RevisionAction]) -> None:
        self._actions = list(actions)

    def propose(self, view: SharedCandidateView) -> ProposalOutput:
        raw_text = actions_to_json(self._actions)
        return ProposalOutput(raw_text=raw_text, parse_result=parse_actions(raw_text))
