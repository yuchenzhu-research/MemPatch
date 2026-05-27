"""Deterministic gate-unit tests for `DefeatPathAuthorizationAlgorithm`.

Every test injects typed graph nodes and edges directly. No verifier is
ever instantiated. The required cases are enumerated in the Wave-1A spec
under "Required gate-unit tests"; this module covers all ten plus a
single supporting test for the documented legacy import alias.

The semantic convention these tests assert (refactor plan amendment A2 +
Wave-1A spec, section 4):

    AUTHORIZED means the belief is eligible to participate in the
    current authorized basis. It does not mean the system has newly
    verified that the belief is presently true. RELEASES clears a
    blocker; it never creates a new assertion of truth. A later
    SUPERSEDES still defeats the old belief regardless of any prior
    RELEASES.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Load the sibling `_helpers` module without relying on package-relative
# imports, so the same file works under pytest discovery and under the
# stdlib loader used in the refactor verification workflow.
_HERE = os.path.dirname(__file__)
_HELPERS_PATH = os.path.join(_HERE, "_helpers.py")
_helpers_spec = importlib.util.spec_from_file_location(
    "tests_gate_unit_helpers", _HELPERS_PATH
)
assert _helpers_spec is not None and _helpers_spec.loader is not None
_helpers = importlib.util.module_from_spec(_helpers_spec)
sys.modules.setdefault("tests_gate_unit_helpers", _helpers)
_helpers_spec.loader.exec_module(_helpers)

make_evidence = _helpers.make_evidence
make_belief = _helpers.make_belief
make_condition = _helpers.make_condition
make_dependency = _helpers.make_dependency
make_evidence_edge = _helpers.make_evidence_edge
build_world = _helpers.build_world

from retracemem.schemas import (
    AuthorizationStatus,
    DefeatPathType,
    EvidenceEdgeType,
)
from retracemem.tms.authorization import (
    AuthorizationEngine,
    DefeatPathAuthorizationAlgorithm,
)


# ---------------------------------------------------------------------------
# Required case 1: direct supersession surfaces the replacement belief
# ---------------------------------------------------------------------------


def test_direct_supersession_surfaces_replacement_belief() -> None:
    """A SUPERSEDES edge on bel_old must defeat bel_old AND expose bel_new
    via ``DefeatPath.replacement_belief_id`` (amendment A1)."""

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_old", "2026-01-01T00:00:00Z"),
            make_evidence("ev_new", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_old_address", "User lives at 12 Pine St.", ("ev_old",)),
            make_belief("bel_new_address", "User lives at 48 Cedar Ave.", ("ev_new",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_super",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_new",
                target_kind="belief",
                target_id="bel_old_address",
                replacement_belief_id="bel_new_address",
            ),
        ],
    )

    dpa = DefeatPathAuthorizationAlgorithm(store, ledger)
    trace = dpa.authorize("bel_old_address")

    assert trace.status == AuthorizationStatus.SUPERSEDED
    assert trace.accepted_defeat_path is not None
    assert trace.accepted_defeat_path.path_type == DefeatPathType.DIRECT_SUPERSEDE
    assert trace.accepted_defeat_path.replacement_belief_id == "bel_new_address"
    assert trace.accepted_defeat_path.supporting_evidence_edge_ids == ("ev_edge_super",)
    assert trace.accepted_defeat_path.supporting_dependency_edge_ids == ()

    new_trace = dpa.authorize("bel_new_address")
    assert new_trace.status == AuthorizationStatus.AUTHORIZED
    assert new_trace.accepted_defeat_path is None


# ---------------------------------------------------------------------------
# Required case 2: prerequisite block disables authorization
# ---------------------------------------------------------------------------


def test_prerequisite_block_disables_authorization() -> None:
    """A BLOCKS(c) edge defeats bel only when REQUIRES(bel, c) is present.

    The accepted defeat path must contain both the dependency edge id and
    the blocking evidence edge id so the trace is independently
    auditable.
    """

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_broken_leg", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_bike_commute", "User commutes by bicycle.", ("ev_seed",)),
        ],
        conditions=[make_condition("cond_cycling", "cycling ability")],
        dependency_edges=[
            make_dependency(
                edge_id="dep_bike_cycling",
                belief_id="bel_bike_commute",
                condition_id="cond_cycling",
                supporting_evidence_ids=("ev_seed",),
            ),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_blocks",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_broken_leg",
                target_kind="condition",
                target_id="cond_cycling",
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_bike_commute")

    assert trace.status == AuthorizationStatus.BLOCKED
    assert trace.accepted_defeat_path is not None
    assert trace.accepted_defeat_path.path_type == DefeatPathType.PREREQUISITE_BLOCK
    assert trace.accepted_defeat_path.supporting_dependency_edge_ids == ("dep_bike_cycling",)
    assert trace.accepted_defeat_path.supporting_evidence_edge_ids == ("ev_edge_blocks",)
    assert trace.accepted_defeat_path.replacement_belief_id is None


# ---------------------------------------------------------------------------
# Required case 3: an unrelated belief remains authorized
# ---------------------------------------------------------------------------


def test_unrelated_belief_remains_authorized() -> None:
    """A blocker that targets a condition unrelated to ``bel_food`` must
    leave ``bel_food`` AUTHORIZED. Defeat reasons must not leak across
    beliefs."""

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed_bike", "2026-01-01T00:00:00Z"),
            make_evidence("ev_seed_food", "2026-01-02T00:00:00Z"),
            make_evidence("ev_broken_leg", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_bike_commute", "User commutes by bicycle.", ("ev_seed_bike",)),
            make_belief("bel_food_preference", "User likes spicy noodles.", ("ev_seed_food",)),
        ],
        conditions=[make_condition("cond_cycling", "cycling ability")],
        dependency_edges=[
            make_dependency(
                edge_id="dep_bike_cycling",
                belief_id="bel_bike_commute",
                condition_id="cond_cycling",
                supporting_evidence_ids=("ev_seed_bike",),
            ),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_blocks",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_broken_leg",
                target_kind="condition",
                target_id="cond_cycling",
            ),
        ],
    )

    dpa = DefeatPathAuthorizationAlgorithm(store, ledger)
    food_trace = dpa.authorize("bel_food_preference")
    bike_trace = dpa.authorize("bel_bike_commute")

    assert food_trace.status == AuthorizationStatus.AUTHORIZED
    assert food_trace.accepted_defeat_path is None
    assert food_trace.supporting_evidence_ids == ("ev_seed_food",)

    # Sanity check: the blocker still defeats the related belief.
    assert bike_trace.status == AuthorizationStatus.BLOCKED


# ---------------------------------------------------------------------------
# Required case 4: BLOCKS without a REQUIRES anchor does not defeat
# ---------------------------------------------------------------------------


def test_blocks_without_requires_anchor_does_not_defeat_belief() -> None:
    """The structural rule: BLOCKS(c) is well-formed but defeats no belief
    unless an accepted REQUIRES(bel, c) anchor exists. The well-formed
    BLOCKS edge must still be retained in the store for later anchoring.
    """

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_broken_leg", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_bike_commute", "User commutes by bicycle.", ("ev_seed",)),
        ],
        conditions=[make_condition("cond_cycling", "cycling ability")],
        # No DependencyEdge linking bel_bike_commute to cond_cycling.
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_blocks_orphan",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_broken_leg",
                target_kind="condition",
                target_id="cond_cycling",
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_bike_commute")

    assert trace.status == AuthorizationStatus.AUTHORIZED
    assert trace.accepted_defeat_path is None
    assert trace.considered_defeat_paths == ()
    # The orphan BLOCKS edge is still present in the store, ready to
    # become active if a REQUIRES edge is later admitted.
    assert store.has_evidence_edge("ev_edge_blocks_orphan")


# ---------------------------------------------------------------------------
# Required case 5: RELEASES clears the active blocker without asserting truth
# ---------------------------------------------------------------------------


def test_release_clears_active_blocker_without_asserting_new_truth() -> None:
    """After RELEASES at T2 follows BLOCKS at T1, the belief is AUTHORIZED
    again because the latest evidence update for the condition is no
    longer a blocker.

    The trace's status reflects eligibility only, not a fresh truth claim.
    """

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_broken_leg", "2026-02-01T00:00:00Z"),
            make_evidence("ev_recovered", "2026-03-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_bike_commute", "User commutes by bicycle.", ("ev_seed",)),
        ],
        conditions=[make_condition("cond_cycling", "cycling ability")],
        dependency_edges=[
            make_dependency(
                edge_id="dep_bike_cycling",
                belief_id="bel_bike_commute",
                condition_id="cond_cycling",
            ),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_blocks",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_broken_leg",
                target_kind="condition",
                target_id="cond_cycling",
            ),
            make_evidence_edge(
                edge_id="ev_edge_releases",
                edge_type=EvidenceEdgeType.RELEASES,
                evidence_id="ev_recovered",
                target_kind="condition",
                target_id="cond_cycling",
            ),
        ],
    )

    dpa = DefeatPathAuthorizationAlgorithm(store, ledger)

    # At the broken-leg cutoff, the belief is blocked.
    blocked = dpa.authorize("bel_bike_commute", as_of_evidence_id="ev_broken_leg")
    assert blocked.status == AuthorizationStatus.BLOCKED

    # After recovery, the latest condition update is RELEASES; the belief
    # is eligible again.
    released = dpa.authorize("bel_bike_commute", as_of_evidence_id="ev_recovered")
    assert released.status == AuthorizationStatus.AUTHORIZED
    assert released.accepted_defeat_path is None
    # The supporting evidence is still the original seed, not the recovery.
    assert released.supporting_evidence_ids == ("ev_seed",)


# ---------------------------------------------------------------------------
# Required case 6: a later SUPERSEDES wins over a prior RELEASES
# ---------------------------------------------------------------------------


def test_release_does_not_restore_belief_after_later_supersession() -> None:
    """Sequence: BLOCKS, RELEASES, then SUPERSEDES. The belief must end
    SUPERSEDED, never re-admitted by the RELEASES that preceded the
    supersession.
    """

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_broken_leg", "2026-02-01T00:00:00Z"),
            make_evidence("ev_recovered", "2026-03-01T00:00:00Z"),
            make_evidence("ev_moved_house", "2026-04-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_old_address", "User lives at 12 Pine St.", ("ev_seed",)),
            make_belief("bel_new_address", "User lives at 48 Cedar Ave.", ("ev_moved_house",)),
        ],
        conditions=[make_condition("cond_at_pine_st", "currently at Pine St.")],
        dependency_edges=[
            make_dependency(
                edge_id="dep_old_residence",
                belief_id="bel_old_address",
                condition_id="cond_at_pine_st",
            ),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_blocks_pine",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_broken_leg",
                target_kind="condition",
                target_id="cond_at_pine_st",
            ),
            make_evidence_edge(
                edge_id="ev_edge_releases_pine",
                edge_type=EvidenceEdgeType.RELEASES,
                evidence_id="ev_recovered",
                target_kind="condition",
                target_id="cond_at_pine_st",
            ),
            make_evidence_edge(
                edge_id="ev_edge_super_address",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_moved_house",
                target_kind="belief",
                target_id="bel_old_address",
                replacement_belief_id="bel_new_address",
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_old_address")

    assert trace.status == AuthorizationStatus.SUPERSEDED
    assert trace.accepted_defeat_path is not None
    assert trace.accepted_defeat_path.path_type == DefeatPathType.DIRECT_SUPERSEDE
    assert trace.accepted_defeat_path.replacement_belief_id == "bel_new_address"
    assert trace.accepted_defeat_path.supporting_evidence_edge_ids == ("ev_edge_super_address",)


# ---------------------------------------------------------------------------
# Required case 7: UNCERTAIN yields UNRESOLVED
# ---------------------------------------------------------------------------


def test_uncertain_yields_unresolved() -> None:
    """An UNCERTAIN edge on a belief, with no later REAFFIRMS, yields the
    UNRESOLVED_UNCERTAIN defeat path."""

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_doubt", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_address", "User lives at 12 Pine St.", ("ev_seed",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_unc",
                edge_type=EvidenceEdgeType.UNCERTAIN,
                evidence_id="ev_doubt",
                target_kind="belief",
                target_id="bel_address",
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_address")

    assert trace.status == AuthorizationStatus.UNRESOLVED
    assert trace.accepted_defeat_path is not None
    assert trace.accepted_defeat_path.path_type == DefeatPathType.UNRESOLVED_UNCERTAIN
    assert trace.accepted_defeat_path.supporting_evidence_edge_ids == ("ev_edge_unc",)
    assert trace.accepted_defeat_path.supporting_dependency_edge_ids == ()


# ---------------------------------------------------------------------------
# Required case 8: REAFFIRMS clears prior UNCERTAIN
# ---------------------------------------------------------------------------


def test_reaffirms_clears_prior_uncertainty() -> None:
    """A REAFFIRMS edge strictly later than the latest UNCERTAIN restores
    the belief to AUTHORIZED."""

    store, ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_doubt", "2026-02-01T00:00:00Z"),
            make_evidence("ev_confirm", "2026-03-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_address", "User lives at 12 Pine St.", ("ev_seed",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_unc",
                edge_type=EvidenceEdgeType.UNCERTAIN,
                evidence_id="ev_doubt",
                target_kind="belief",
                target_id="bel_address",
            ),
            make_evidence_edge(
                edge_id="ev_edge_reaff",
                edge_type=EvidenceEdgeType.REAFFIRMS,
                evidence_id="ev_confirm",
                target_kind="belief",
                target_id="bel_address",
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_address")

    assert trace.status == AuthorizationStatus.AUTHORIZED
    assert trace.accepted_defeat_path is None
    assert trace.supporting_evidence_ids == ("ev_seed",)


# ---------------------------------------------------------------------------
# Required case 9: authorized trace has no defeat path
# ---------------------------------------------------------------------------


def test_authorized_trace_has_no_defeat_path() -> None:
    """Per amendment A5, an authorized trace must have
    ``accepted_defeat_path is None`` and an empty considered list.
    Supporting evidence ids come from the belief's own provenance."""

    store, ledger = build_world(
        evidences=[make_evidence("ev_simple", "2026-01-01T00:00:00Z")],
        beliefs=[
            make_belief(
                "bel_simple", "User enjoys hiking.", ("ev_simple",),
            ),
        ],
    )

    trace = DefeatPathAuthorizationAlgorithm(store, ledger).authorize("bel_simple")

    assert trace.status == AuthorizationStatus.AUTHORIZED
    assert trace.accepted_defeat_path is None
    assert trace.considered_defeat_paths == ()
    assert trace.supporting_evidence_ids == ("ev_simple",)


# ---------------------------------------------------------------------------
# Required case 10: temporal tie-break is deterministic
# ---------------------------------------------------------------------------


def test_temporal_tie_break_is_deterministic() -> None:
    """When two competing SUPERSEDES edges share the same timestamp,
    ``(timestamp, ledger_index, edge_id)`` must pick the same winner
    regardless of edge insertion order."""

    # Two SUPERSEDES edges from two evidence atoms recorded at the same
    # ISO-8601 timestamp; ledger insertion order alone breaks the tie.
    forward_store, forward_ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_tie_a", "2026-02-01T00:00:00Z"),
            make_evidence("ev_tie_b", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_address", "User lives at 12 Pine St.", ("ev_seed",)),
            make_belief("bel_address_alt_a", "User lives at 48 Cedar Ave.", ("ev_tie_a",)),
            make_belief("bel_address_alt_b", "User lives at 7 Birch Rd.", ("ev_tie_b",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_super_a",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie_a",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_a",
            ),
            make_evidence_edge(
                edge_id="ev_edge_super_b",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie_b",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_b",
            ),
        ],
    )

    forward_trace = DefeatPathAuthorizationAlgorithm(
        forward_store, forward_ledger
    ).authorize("bel_address")
    # ev_tie_b has the larger ledger index, so its SUPERSEDES wins.
    assert forward_trace.status == AuthorizationStatus.SUPERSEDED
    assert forward_trace.accepted_defeat_path is not None
    assert forward_trace.accepted_defeat_path.replacement_belief_id == "bel_address_alt_b"

    # Re-run with the two competing edges inserted in the opposite order.
    # Ledger positions are unchanged (both ev_tie_a / ev_tie_b sit at
    # index 1 / 2 respectively), so the result must be identical.
    reverse_store, reverse_ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_tie_a", "2026-02-01T00:00:00Z"),
            make_evidence("ev_tie_b", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_address", "User lives at 12 Pine St.", ("ev_seed",)),
            make_belief("bel_address_alt_a", "User lives at 48 Cedar Ave.", ("ev_tie_a",)),
            make_belief("bel_address_alt_b", "User lives at 7 Birch Rd.", ("ev_tie_b",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="ev_edge_super_b",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie_b",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_b",
            ),
            make_evidence_edge(
                edge_id="ev_edge_super_a",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie_a",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_a",
            ),
        ],
    )

    reverse_trace = DefeatPathAuthorizationAlgorithm(
        reverse_store, reverse_ledger
    ).authorize("bel_address")
    assert reverse_trace.status == AuthorizationStatus.SUPERSEDED
    assert reverse_trace.accepted_defeat_path is not None
    assert (
        reverse_trace.accepted_defeat_path.replacement_belief_id
        == forward_trace.accepted_defeat_path.replacement_belief_id
    )
    assert (
        reverse_trace.accepted_defeat_path.supporting_evidence_edge_ids
        == forward_trace.accepted_defeat_path.supporting_evidence_edge_ids
    )

    # And the same-evidence edge-id tiebreak: two SUPERSEDES edges sharing
    # the same evidence atom must be ranked lexicographically on edge_id.
    same_evidence_store, same_evidence_ledger = build_world(
        evidences=[
            make_evidence("ev_seed", "2026-01-01T00:00:00Z"),
            make_evidence("ev_tie", "2026-02-01T00:00:00Z"),
        ],
        beliefs=[
            make_belief("bel_address", "User lives at 12 Pine St.", ("ev_seed",)),
            make_belief("bel_address_alt_a", "User lives at 48 Cedar Ave.", ("ev_tie",)),
            make_belief("bel_address_alt_b", "User lives at 7 Birch Rd.", ("ev_tie",)),
        ],
        evidence_edges=[
            make_evidence_edge(
                edge_id="aaa_super_first",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_a",
            ),
            make_evidence_edge(
                edge_id="zzz_super_second",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id="ev_tie",
                target_kind="belief",
                target_id="bel_address",
                replacement_belief_id="bel_address_alt_b",
            ),
        ],
    )
    edge_tiebreak = DefeatPathAuthorizationAlgorithm(
        same_evidence_store, same_evidence_ledger
    ).authorize("bel_address")
    assert edge_tiebreak.accepted_defeat_path is not None
    # Lexicographically later edge_id wins the final tiebreak.
    assert edge_tiebreak.accepted_defeat_path.supporting_evidence_edge_ids == (
        "zzz_super_second",
    )
    assert edge_tiebreak.accepted_defeat_path.replacement_belief_id == "bel_address_alt_b"


# ---------------------------------------------------------------------------
# Supporting test: the legacy import alias points at the DPA class
# ---------------------------------------------------------------------------


def test_authorization_engine_alias_points_at_dpa() -> None:
    """``tms.AuthorizationEngine`` is a name alias for the new DPA so that
    downstream packages still importing it do not raise ImportError at
    module load. The underlying behavior is the typed-graph DPA, not the
    legacy decide(Belief, ...) method.
    """

    assert AuthorizationEngine is DefeatPathAuthorizationAlgorithm
