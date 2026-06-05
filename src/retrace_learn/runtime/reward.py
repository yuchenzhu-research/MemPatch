"""Benchmark-grounded Feedback — MemPatch Revision Module training signal.

After DPA-Consistent Projection, scores ``r_final`` against ``hidden_gold``
and decomposes benchmark metrics into a training signal. Supports supervised
``L = L_state + L_evidence + L_decision + L_diagnosis`` and preference-style
improvement when training scripts are configured (SFT / RSFT / DPO).

Benchmark-aligned reward terms:

* ``final_status_reward`` → ``memory_state_accuracy`` (``L_state``)
* ``evidence_grounding_reward`` → ``evidence_f1`` (``L_evidence``)
* ``stale_propagation_penalty`` → ``stale_reuse_rate`` / ``stale_memory_reuse``
* ``over_update_penalty`` / ``under_update_penalty`` → failure modes

Aggregate (conceptual):

    R ≈ memory_state_accuracy + evidence_f1 + joint_revision_success
        - stale_reuse_rate - over_update_penalty - under_update_penalty

Decomposed implementation:

    R = + w_final  * final_status_reward
        + w_json   * valid_json_reward
        + w_tgrd   * target_grounding_reward
        + w_egrd   * evidence_grounding_reward
        + w_nostale* no_stale_propagation_reward
        - w_parse  * parser_error_penalty
        - w_inv    * invalid_target_penalty
        - w_miss   * missing_evidence_penalty
        - w_over   * over_update_penalty
        - w_under  * under_update_penalty
        - w_spur   * spurious_uncertain_penalty
        - w_stale  * stale_propagation_penalty
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from retracemem.methods.contracts import SharedCandidateView

from retrace_learn.schemas import RevisionAction
from retrace_learn.runtime.dpa_runtime import ParseResult, RuntimeResult
from retrace_learn.runtime.engine_errors import EngineError, EngineStage

# Statuses that mean "the belief was defeated and must not stay usable".
_DEFEATED_STATUSES = frozenset({"SUPERSEDED", "BLOCKED"})


@dataclass(frozen=True)
class RewardWeights:
    final_status: float = 1.0
    valid_json: float = 0.2
    target_grounding: float = 0.3
    evidence_grounding: float = 0.3
    no_stale_propagation: float = 0.5
    parser_error: float = 1.0
    invalid_target: float = 0.5
    missing_evidence: float = 0.5
    over_update: float = 0.5
    under_update: float = 0.5
    spurious_uncertain: float = 0.3
    stale_propagation: float = 1.0
    gate_rejection: float = 0.4
    no_revision_overuse: float = 0.2


DEFAULT_WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class LearnRewardBreakdown:
    # positive components, each in [0, 1]
    final_status_reward: float
    valid_json_reward: float
    target_grounding_reward: float
    evidence_grounding_reward: float
    no_stale_propagation_reward: float
    # penalty components, each in [0, 1]
    parser_error_penalty: float
    invalid_target_penalty: float
    missing_evidence_penalty: float
    over_update_penalty: float
    under_update_penalty: float
    spurious_uncertain_penalty: float
    stale_propagation_penalty: float
    # aggregate
    total_reward: float
    failure_category: str
    diagnostics: dict = field(default_factory=dict)
    gate_rejection_penalty: float = 0.0
    no_revision_overuse_penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def reward_breakdown(self) -> dict[str, float]:
        """The component dict written to ``dpa_rl_rollouts.jsonl``."""
        d = asdict(self)
        d.pop("diagnostics", None)
        d.pop("failure_category", None)
        d.pop("total_reward", None)
        return d


def _fraction(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    if denominator == 0:
        return empty
    return numerator / denominator


def classify_failure(b: "LearnRewardBreakdown") -> str:
    """Pick the single dominant failure category for analysis/curriculum."""
    if b.parser_error_penalty > 0:
        return "PARSER_ERROR"
    if b.invalid_target_penalty > 0:
        return "INVALID_TARGET"
    if b.missing_evidence_penalty > 0:
        return "MISSING_EVIDENCE"
    if b.gate_rejection_penalty > 0:
        return "GATE_REJECTION"
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


def compute_reward(
    *,
    actions: list[RevisionAction],
    parse_result: ParseResult,
    dpa_final_statuses: dict[str, str],
    gold_final_statuses: dict[str, str],
    valid_belief_ids: set[str],
    valid_condition_ids: set[str],
    valid_evidence_ids: set[str],
    gold_actions: list[RevisionAction] | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    engine_errors: tuple[EngineError, ...] = (),
) -> LearnRewardBreakdown:
    """Compute the decomposed reward for one rollout."""
    parser_ok = parse_result.valid_json and parse_result.schema_valid
    valid_json_reward = 1.0 if parser_ok else 0.0
    parser_error_penalty = 0.0 if parser_ok else 1.0

    revision_actions = [a for a in actions if a.action_type != "NO_REVISION"]

    # --- grounding (target + evidence) over revision actions ---
    grounded_targets = 0
    grounded_evidence = 0
    for a in revision_actions:
        target_ok = True
        if a.target_belief_id is not None:
            target_ok = target_ok and a.target_belief_id in valid_belief_ids
        if a.replacement_belief_id is not None:
            target_ok = target_ok and a.replacement_belief_id in valid_belief_ids
        if a.target_condition_id is not None:
            target_ok = target_ok and a.target_condition_id in valid_condition_ids
        if target_ok:
            grounded_targets += 1
        if a.evidence_ids and all(ev in valid_evidence_ids for ev in a.evidence_ids):
            grounded_evidence += 1

    n_rev = len(revision_actions)
    target_grounding_reward = _fraction(grounded_targets, n_rev)
    evidence_grounding_reward = _fraction(grounded_evidence, n_rev)
    invalid_target_penalty = 1.0 - target_grounding_reward
    missing_evidence_penalty = 1.0 - evidence_grounding_reward

    # --- DPA final-status agreement ---
    gold_keys = list(gold_final_statuses.keys())
    correct = sum(
        1
        for bid in gold_keys
        if dpa_final_statuses.get(bid) == gold_final_statuses[bid]
    )
    final_status_reward = _fraction(correct, len(gold_keys), empty=1.0)

    # --- memory-safety: over / under / stale ---
    over = 0  # model defeated a belief gold left AUTHORIZED
    under = 0  # model left AUTHORIZED a belief gold defeated/unresolved
    stale = 0  # specifically: gold SUPERSEDED/BLOCKED but model AUTHORIZED
    for bid in gold_keys:
        gold = gold_final_statuses[bid]
        pred = dpa_final_statuses.get(bid)
        if pred is None:
            continue
        if gold == "AUTHORIZED" and pred != "AUTHORIZED":
            over += 1
        if gold != "AUTHORIZED" and pred == "AUTHORIZED":
            under += 1
            if gold in _DEFEATED_STATUSES:
                stale += 1
    n_gold = len(gold_keys)
    over_update_penalty = _fraction(over, n_gold, empty=0.0)
    under_update_penalty = _fraction(under, n_gold, empty=0.0)
    stale_propagation_penalty = _fraction(stale, n_gold, empty=0.0)
    no_stale_propagation_reward = 1.0 - stale_propagation_penalty

    # --- spurious uncertainty: UNCERTAIN where gold had none for that belief ---
    spurious = 0.0
    n_uncertain = sum(1 for a in revision_actions if a.action_type == "UNCERTAIN")
    if n_uncertain:
        gold_uncertain_targets = set()
        for ga in gold_actions or []:
            if ga.action_type == "UNCERTAIN" and ga.target_belief_id:
                gold_uncertain_targets.add(ga.target_belief_id)
        spurious_count = sum(
            1
            for a in revision_actions
            if a.action_type == "UNCERTAIN"
            and a.target_belief_id not in gold_uncertain_targets
        )
        spurious = _fraction(spurious_count, n_uncertain, empty=0.0)
    spurious_uncertain_penalty = spurious

    # --- gate rejection penalty (from structured engine errors) ---
    gate_rejections = sum(1 for e in engine_errors if e.stage == EngineStage.REVISION_GATE)
    gate_rejection_penalty = _fraction(gate_rejections, max(n_rev, 1), empty=0.0)

    # --- NO_REVISION overuse penalty ---
    n_no_revision = sum(1 for a in actions if a.action_type == "NO_REVISION")
    gold_revision_count = sum(1 for ga in (gold_actions or []) if ga.action_type != "NO_REVISION")
    no_revision_overuse = 0.0
    if gold_revision_count > 0 and len(actions) > 0:
        no_revision_ratio = n_no_revision / len(actions)
        if no_revision_ratio > 0.5:
            no_revision_overuse = min(1.0, (no_revision_ratio - 0.5) * 2.0)
    no_revision_overuse_penalty = no_revision_overuse

    total = (
        weights.final_status * final_status_reward
        + weights.valid_json * valid_json_reward
        + weights.target_grounding * target_grounding_reward
        + weights.evidence_grounding * evidence_grounding_reward
        + weights.no_stale_propagation * no_stale_propagation_reward
        - weights.parser_error * parser_error_penalty
        - weights.invalid_target * invalid_target_penalty
        - weights.missing_evidence * missing_evidence_penalty
        - weights.over_update * over_update_penalty
        - weights.under_update * under_update_penalty
        - weights.spurious_uncertain * spurious_uncertain_penalty
        - weights.stale_propagation * stale_propagation_penalty
        - weights.gate_rejection * gate_rejection_penalty
        - weights.no_revision_overuse * no_revision_overuse_penalty
    )

    breakdown = LearnRewardBreakdown(
        final_status_reward=final_status_reward,
        valid_json_reward=valid_json_reward,
        target_grounding_reward=target_grounding_reward,
        evidence_grounding_reward=evidence_grounding_reward,
        no_stale_propagation_reward=no_stale_propagation_reward,
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
            "gate_rejections": gate_rejections,
            "n_no_revision": n_no_revision,
        },
        gate_rejection_penalty=gate_rejection_penalty,
        no_revision_overuse_penalty=no_revision_overuse_penalty,
    )
    return LearnRewardBreakdown(
        **{**breakdown.to_dict(), "failure_category": classify_failure(breakdown)}
    )


def compute_reward_for_view(
    view: SharedCandidateView,
    runtime_result: RuntimeResult,
    gold_final_statuses: dict[str, str],
    *,
    gold_actions: list[RevisionAction] | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> LearnRewardBreakdown:
    """Convenience wrapper deriving valid-id sets from the view + runtime result."""
    valid_belief_ids = {b.belief_id for b in view.candidate_beliefs}
    valid_belief_ids |= {b.belief_id for b in view.candidate_replacement_beliefs}
    valid_condition_ids = {
        c.condition_id for _bid, conds in view.candidate_conditions_by_belief for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in view.evidence_context}
    return compute_reward(
        actions=list(runtime_result.parse_result.actions),
        parse_result=runtime_result.parse_result,
        dpa_final_statuses=runtime_result.final_belief_statuses,
        gold_final_statuses=gold_final_statuses,
        valid_belief_ids=valid_belief_ids,
        valid_condition_ids=valid_condition_ids,
        valid_evidence_ids=valid_evidence_ids,
        gold_actions=gold_actions,
        weights=weights,
        engine_errors=runtime_result.engine_errors,
    )
