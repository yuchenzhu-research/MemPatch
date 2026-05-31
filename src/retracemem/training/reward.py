"""Reward evaluation for trainable Stage C policies.

Includes shaping terms for JSON validation, vocabulary constraint, grounding,
DPA-status accuracy, stale propagation, action minimality, and audit trace completeness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from retracemem.schemas import EvidenceEdge


@dataclass(frozen=True)
class RewardWeights:
    final_status: float = 1.0
    valid_json: float = 0.2
    valid_vocabulary: float = 0.2
    target_grounding: float = 0.3
    evidence_grounding: float = 0.3
    no_stale_propagation: float = 0.5
    minimal_sufficient_action_set: float = 0.2
    audit_completeness: float = 0.3
    parser_error: float = 1.0
    invalid_target: float = 0.5
    missing_evidence: float = 0.5
    over_update: float = 0.5
    under_update: float = 0.5
    spurious_uncertain: float = 0.3
    stale_propagation: float = 1.0


DEFAULT_WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class RewardBreakdown:
    # Positive items [0, 1]
    final_status_reward: float
    valid_json_reward: float
    valid_vocabulary_reward: float
    target_grounding_reward: float
    evidence_grounding_reward: float
    no_stale_propagation_reward: float
    minimal_sufficient_action_set_reward: float
    audit_completeness_reward: float

    # Penalties [0, 1]
    parser_error_penalty: float
    invalid_target_penalty: float
    missing_evidence_penalty: float
    over_update_penalty: float
    under_update_penalty: float
    spurious_uncertain_penalty: float
    stale_propagation_penalty: float

    total_reward: float
    failure_category: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reward_breakdown(self) -> dict[str, float]:
        d = asdict(self)
        d.pop("diagnostics", None)
        d.pop("failure_category", None)
        d.pop("total_reward", None)
        return d


def _fraction(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    if denominator == 0:
        return empty
    return numerator / denominator


def classify_failure(b: RewardBreakdown) -> str:
    if b.parser_error_penalty > 0:
        return "PARSER_ERROR"
    if b.invalid_target_penalty > 0:
        return "INVALID_TARGET"
    if b.missing_evidence_penalty > 0:
        return "MISSING_EVIDENCE"
    if b.stale_propagation_penalty > 0:
        return "STALE_PROPAGATION"
    if b.over_update_penalty > 0:
        return "OVER_UPDATE"
    if b.under_update_penalty > 0:
        return "UNDER_UPDATE"
    if b.spurious_uncertain_penalty > 0:
        return "SPURIOUS_UNCERTAIN"
    if b.final_status_reward < 1.0:
        return "WRONG_FINAL_STATUS"
    return "NONE"


def score_rollout(
    *,
    actions: list[dict[str, Any]],
    valid_json: bool,
    valid_vocabulary: bool,
    dpa_final_statuses: dict[str, str],
    gold_final_statuses: dict[str, str],
    valid_belief_ids: set[str],
    valid_condition_ids: set[str],
    valid_evidence_ids: set[str],
    defeat_paths: list[dict[str, Any]],
    gold_actions: list[dict[str, Any]] | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown:
    """Evaluate and score a proposer rollout trajectory."""
    valid_json_reward = 1.0 if valid_json else 0.0
    parser_error_penalty = 0.0 if valid_json else 1.0
    valid_vocabulary_reward = 1.0 if valid_vocabulary else 0.0

    revision_actions = [a for a in actions if a.get("action_type") != "NO_REVISION"]
    n_rev = len(revision_actions)

    # 1. Grounding checks
    grounded_targets = 0
    grounded_evidence = 0
    for a in revision_actions:
        target_ok = True
        if a.get("target_belief_id") is not None:
            target_ok = target_ok and a["target_belief_id"] in valid_belief_ids
        if a.get("replacement_belief_id") is not None:
            target_ok = target_ok and a["replacement_belief_id"] in valid_belief_ids
        if a.get("target_condition_id") is not None:
            target_ok = target_ok and a["target_condition_id"] in valid_condition_ids
        if target_ok:
            grounded_targets += 1
        if a.get("evidence_ids") and all(ev in valid_evidence_ids for ev in a["evidence_ids"]):
            grounded_evidence += 1

    target_grounding_reward = _fraction(grounded_targets, n_rev)
    evidence_grounding_reward = _fraction(grounded_evidence, n_rev)
    invalid_target_penalty = 1.0 - target_grounding_reward
    missing_evidence_penalty = 1.0 - evidence_grounding_reward

    # 2. Final Status Correctness
    gold_keys = list(gold_final_statuses.keys())
    correct = sum(
        1
        for bid in gold_keys
        if dpa_final_statuses.get(bid) == gold_final_statuses[bid]
    )
    final_status_reward = _fraction(correct, len(gold_keys), empty=1.0)

    # 3. Stale Propagation / Over-update / Under-update
    over = 0
    under = 0
    stale = 0
    for bid in gold_keys:
        gold = gold_final_statuses[bid]
        pred = dpa_final_statuses.get(bid)
        if pred is None:
            continue
        if gold == "AUTHORIZED" and pred != "AUTHORIZED":
            over += 1
        if gold != "AUTHORIZED" and pred == "AUTHORIZED":
            under += 1
            if gold in ("SUPERSEDED", "BLOCKED"):
                stale += 1

    n_gold = len(gold_keys)
    over_update_penalty = _fraction(over, n_gold, empty=0.0)
    under_update_penalty = _fraction(under, n_gold, empty=0.0)
    stale_propagation_penalty = _fraction(stale, n_gold, empty=0.0)
    no_stale_propagation_reward = 1.0 - stale_propagation_penalty

    # 4. Spurious Uncertainty
    spurious = 0.0
    n_uncertain = sum(1 for a in revision_actions if a.get("action_type") == "UNCERTAIN")
    if n_uncertain:
        gold_uncertain_targets = set()
        for ga in gold_actions or []:
            if ga.get("action_type") == "UNCERTAIN" and ga.get("target_belief_id"):
                gold_uncertain_targets.add(ga["target_belief_id"])
        spurious_count = sum(
            1
            for a in revision_actions
            if a.get("action_type") == "UNCERTAIN"
            and a.get("target_belief_id") not in gold_uncertain_targets
        )
        spurious = _fraction(spurious_count, n_uncertain, empty=0.0)
    spurious_uncertain_penalty = spurious

    # 5. Minimal Sufficient Action Set
    # Reward is 1.0 if generated action count matches or is less than gold, otherwise scaled down
    gold_count = len(gold_actions) if gold_actions is not None else 0
    if n_rev <= gold_count:
        minimal_sufficient_action_set_reward = 1.0
    else:
        minimal_sufficient_action_set_reward = _fraction(gold_count, n_rev, empty=1.0)

    # 6. Audit Completeness
    # Every excluded belief should have a corresponding defeat path record
    excluded_beliefs = [bid for bid, status in dpa_final_statuses.items() if status != "AUTHORIZED"]
    audit_documented = 0
    for bid in excluded_beliefs:
        # Check if this bid exists in defeat_paths list
        if any(dp.get("belief_id") == bid for dp in defeat_paths):
            audit_documented += 1
    audit_completeness_reward = _fraction(audit_documented, len(excluded_beliefs), empty=1.0)

    total = (
        weights.final_status * final_status_reward
        + weights.valid_json * valid_json_reward
        + weights.valid_vocabulary * valid_vocabulary_reward
        + weights.target_grounding * target_grounding_reward
        + weights.evidence_grounding * evidence_grounding_reward
        + weights.no_stale_propagation * no_stale_propagation_reward
        + weights.minimal_sufficient_action_set * minimal_sufficient_action_set_reward
        + weights.audit_completeness * audit_completeness_reward
        - weights.parser_error * parser_error_penalty
        - weights.invalid_target * invalid_target_penalty
        - weights.missing_evidence * missing_evidence_penalty
        - weights.over_update * over_update_penalty
        - weights.under_update * under_update_penalty
        - weights.spurious_uncertain * spurious_uncertain_penalty
        - weights.stale_propagation * stale_propagation_penalty
    )

    breakdown = RewardBreakdown(
        final_status_reward=final_status_reward,
        valid_json_reward=valid_json_reward,
        valid_vocabulary_reward=valid_vocabulary_reward,
        target_grounding_reward=target_grounding_reward,
        evidence_grounding_reward=evidence_grounding_reward,
        no_stale_propagation_reward=no_stale_propagation_reward,
        minimal_sufficient_action_set_reward=minimal_sufficient_action_set_reward,
        audit_completeness_reward=audit_completeness_reward,
        parser_error_penalty=parser_error_penalty,
        invalid_target_penalty=invalid_target_penalty,
        missing_evidence_penalty=missing_evidence_penalty,
        over_update_penalty=over_update_penalty,
        under_update_penalty=under_update_penalty,
        spurious_uncertain_penalty=spurious_uncertain_penalty,
        stale_propagation_penalty=stale_propagation_penalty,
        total_reward=round(total, 6),
        failure_category="NONE",
        diagnostics={
            "n_revision_actions": n_rev,
            "n_gold_beliefs": n_gold,
            "correct_final_status": correct,
            "over_count": over,
            "under_count": under,
            "stale_count": stale,
            "audit_documented": audit_documented,
        },
    )
    return RewardBreakdown(
        **{**breakdown.to_dict(), "failure_category": classify_failure(breakdown)}
    )
