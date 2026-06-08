from __future__ import annotations

from mempatch_dpa.authorization import authorize

from mempatch_learn.runtime.dpa_runtime import run_actions
from mempatch_learn.runtime.views import actions_to_proposal_batches, build_view
from mempatch_learn.schemas import RevisionAction


def _base_evidence() -> list[dict[str, str]]:
    return [
        {"evidence_id": "e0", "timestamp": "2024-01-01T00:00:00Z", "text": "init"},
        {"evidence_id": "e1", "timestamp": "2024-01-02T00:00:00Z", "text": "update"},
    ]


def _build_view(
    *,
    new_evidence_id: str = "e1",
    with_replacement: bool = True,
    with_condition: bool = False,
) -> object:
    candidate_replacement_beliefs: list[dict[str, object]] = []
    if with_replacement:
        candidate_replacement_beliefs = [
            {
                "belief_id": "b1_new",
                "proposition": "replacement",
                "source_evidence_ids": ["e1"],
            }
        ]
    candidate_conditions_by_belief: dict[str, list[dict[str, str]]] = {}
    dependency_edges_by_belief: dict[str, list[dict[str, str]]] = {}
    if with_condition:
        candidate_conditions_by_belief = {"b1": [{"condition_id": "c1", "text": "hold"}]}
        dependency_edges_by_belief = {
            "b1": [{"edge_id": "dep1", "belief_id": "b1", "condition_id": "c1"}]
        }
    return build_view(
        instance_id="dpa_case",
        query_id="dpa_case",
        query="What is the current belief?",
        evidence_context=_base_evidence(),
        new_evidence_id=new_evidence_id,
        candidate_beliefs=[
            {
                "belief_id": "b1",
                "proposition": "old belief",
                "source_evidence_ids": ["e0"],
            }
        ],
        candidate_replacement_beliefs=candidate_replacement_beliefs,
        candidate_conditions_by_belief=candidate_conditions_by_belief,
        dependency_edges_by_belief=dependency_edges_by_belief,
    )


def test_authorize_no_proposals_keeps_candidate_beliefs_authorized() -> None:
    view = _build_view(with_replacement=False)
    result = authorize(view, ())
    assert result.authorized_belief_ids == ("b1",)
    assert result.excluded_belief_ids == ()
    assert result.trace["fine_grained_statuses"] == {"b1": "AUTHORIZED"}


def test_supersede_defeats_target_and_surfaces_replacement() -> None:
    view = _build_view()
    runtime = run_actions(
        view,
        [
            RevisionAction(
                action_type="SUPERSEDES",
                target_belief_id="b1",
                replacement_belief_id="b1_new",
                evidence_ids=("e1",),
            )
        ],
    )
    assert runtime.final_belief_statuses == {
        "b1": "SUPERSEDED",
        "b1_new": "AUTHORIZED",
    }
    assert runtime.authorized_belief_ids == ()
    assert runtime.excluded_belief_ids == ("b1",)


def test_prerequisite_block_and_release_snapshot() -> None:
    view = build_view(
        instance_id="release_case",
        query_id="release_case",
        query="Release blocked belief?",
        evidence_context=[
            {"evidence_id": "e0", "timestamp": "2024-01-01T00:00:00Z", "text": "init"},
            {"evidence_id": "e1", "timestamp": "2024-01-02T00:00:00Z", "text": "block"},
            {"evidence_id": "e2", "timestamp": "2024-01-03T00:00:00Z", "text": "release"},
        ],
        new_evidence_id="e2",
        candidate_beliefs=[
            {
                "belief_id": "b1",
                "proposition": "target",
                "source_evidence_ids": ["e0"],
            }
        ],
        candidate_replacement_beliefs=[],
        candidate_conditions_by_belief={"b1": [{"condition_id": "c1", "text": "hold"}]},
        dependency_edges_by_belief={
            "b1": [{"edge_id": "dep1", "belief_id": "b1", "condition_id": "c1"}]
        },
    )
    batches = actions_to_proposal_batches(
        [
            RevisionAction(
                action_type="BLOCKS",
                target_condition_id="c1",
                evidence_ids=("e1",),
            ),
            RevisionAction(
                action_type="RELEASES",
                target_condition_id="c1",
                evidence_ids=("e2",),
            ),
        ]
    )
    result = authorize(view, batches)
    assert result.trace["fine_grained_statuses"] == {"b1": "AUTHORIZED"}
    assert result.authorized_belief_ids == ("b1",)
    assert result.excluded_belief_ids == ()


def test_uncertain_marks_belief_unresolved() -> None:
    view = _build_view(with_replacement=False)
    runtime = run_actions(
        view,
        [
            RevisionAction(
                action_type="UNCERTAIN",
                target_belief_id="b1",
                evidence_ids=("e1",),
            )
        ],
    )
    assert runtime.final_belief_statuses == {"b1": "UNRESOLVED"}
    assert runtime.excluded_belief_ids == ("b1",)


def test_supersede_precedence_over_prerequisite_block() -> None:
    view = build_view(
        instance_id="precedence_case",
        query_id="precedence_case",
        query="Which defeat wins?",
        evidence_context=[
            {"evidence_id": "e0", "timestamp": "2024-01-01T00:00:00Z", "text": "init"},
            {"evidence_id": "e1", "timestamp": "2024-01-02T00:00:00Z", "text": "block"},
            {"evidence_id": "e2", "timestamp": "2024-01-03T00:00:00Z", "text": "supersede"},
        ],
        new_evidence_id="e2",
        candidate_beliefs=[
            {
                "belief_id": "b1",
                "proposition": "old belief",
                "source_evidence_ids": ["e0"],
            }
        ],
        candidate_replacement_beliefs=[
            {
                "belief_id": "b1_new",
                "proposition": "replacement",
                "source_evidence_ids": ["e2"],
            }
        ],
        candidate_conditions_by_belief={"b1": [{"condition_id": "c1", "text": "hold"}]},
        dependency_edges_by_belief={
            "b1": [{"edge_id": "dep1", "belief_id": "b1", "condition_id": "c1"}]
        },
    )
    runtime = run_actions(
        view,
        [
            RevisionAction(
                action_type="BLOCKS",
                target_condition_id="c1",
                evidence_ids=("e1",),
            ),
            RevisionAction(
                action_type="SUPERSEDES",
                target_belief_id="b1",
                replacement_belief_id="b1_new",
                evidence_ids=("e2",),
            ),
        ],
    )
    assert runtime.final_belief_statuses["b1"] == "SUPERSEDED"
    assert runtime.final_belief_statuses["b1_new"] == "AUTHORIZED"
