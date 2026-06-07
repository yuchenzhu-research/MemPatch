"""MemPatch Revision Module end-to-end runner (Algorithm 1)."""
from __future__ import annotations

import json
from typing import Any

from benchmark.public_view import public_scenario_view

from retrace_learn.runtime.benchmark_projection import project_to_benchmark_response
from retrace_learn.runtime.dpa_runtime import run_from_text
from retrace_learn.runtime.learned_proposer import TypedRevisionProposer
from retrace_learn.runtime.scenario_revision import build_scenario_revision_view


def _noop_actions_text(new_evidence_id: str) -> str:
    return json.dumps(
        [
            {
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "evidence_ids": [new_evidence_id],
                "rationale": "noop policy cites latest visible evidence",
            }
        ],
        ensure_ascii=False,
    )


def run_revision_module_on_scenario(
    scenario: dict[str, Any],
    *,
    actions_text: str | None = None,
    proposer: TypedRevisionProposer | None = None,
    raw_response: dict[str, Any] | None = None,
    fallback_answer: str = "",
) -> dict[str, Any]:
    """Run View Builder → Policy (actions) → DPA projection → benchmark response."""
    view = build_scenario_revision_view(scenario)
    public_view = public_scenario_view(scenario)
    if proposer is not None:
        proposal = proposer.propose(view, metadata={"scenario_id": scenario["scenario_id"]})
        actions_text = proposal.raw_text
    elif actions_text is None:
        actions_text = _noop_actions_text(view.new_evidence.evidence_id)

    runtime_result = run_from_text(view, actions_text)
    response = project_to_benchmark_response(
        runtime_result=runtime_result,
        raw_response=raw_response,
        scenario_public_view=public_view,
        fallback_answer=fallback_answer,
    )
    return {"scenario_id": scenario["scenario_id"], "response": response}
