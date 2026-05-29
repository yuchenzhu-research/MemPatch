from __future__ import annotations

DOMAINS = ("software_engineering", "research_workflow")

FAILURE_TYPES = (
    "direct_supersession",
    "stale_propagation",
    "scope_expansion",
    "cross_agent_conflict",
    "temporary_blocker_recovery",
    "duplicate_evidence",
    "ambiguous_update",
)

POLICY_VARIANTS = ("oracle_replay", "prompt", "sft", "reward_refined")

# Scale Target Specifications:
# Current seed: 14 development-only E1 episodes
# Reviewed dev target: 2 domains x 7 failure types x 5 variants = 70 episodes
# Train target: 2 domains x 7 failure types x 30 variants = 420 episodes minimum
# Frozen primary test target: 2 domains x 7 failure types x 20 variants = 280 episodes
# Stress axes:
#   number_of_subagents in {2, 4, 8}
#   conflict_density in {low, medium, high}
#   delay_depth in {1, 3, 6}
