"""Evaluation case loading and namespace renaming (shared A/B/C)."""
from __future__ import annotations

from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    TypedRevisionTarget,
)
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes
from retracemem.evaluation.multiagent.data.paper1_balanced import (
    generate_paper1_balanced_episodes,
)
from retracemem.multiagent.utils import rename_submission

DATASET_DEV_EXPANSION = "dev_expansion"
DATASET_PAPER1_BALANCED = "paper1_balanced"
AVAILABLE_DATASETS = (DATASET_DEV_EXPANSION, DATASET_PAPER1_BALANCED)


def rename_string(s: str | None, old_ns: str, new_ns: str) -> str | None:
    if s is None:
        return None
    return s.replace(old_ns, new_ns)



def rename_episode_and_gold(
    episode: FixedCandidateInputEpisode,
    gold: FixedCandidateGoldRecord,
) -> tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]:
    """Rename namespace of an episode and its gold records."""
    old_ns = episode.episode_id
    new_ns = f"{old_ns}__heldout_base"

    # Rename submissions
    renamed_subs = tuple(rename_submission(s, old_ns, new_ns) for s in episode.submissions)

    # Rename downstream tasks
    renamed_tasks = []
    for t in episode.downstream_tasks:
        renamed_tasks.append(
            t.__class__(
                task_id=rename_string(t.task_id, old_ns, new_ns),
                query=rename_string(t.query, old_ns, new_ns),
                expected_answer_or_action=rename_string(t.expected_answer_or_action, old_ns, new_ns),
                relevant_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in t.relevant_belief_ids),
                protected_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in t.protected_belief_ids),
                metadata=t.metadata,
            )
        )

    # Rename gold record
    renamed_belief_statuses = {}
    for bid, status in gold.gold_snapshot.belief_statuses.items():
        renamed_belief_statuses[rename_string(bid, old_ns, new_ns)] = status

    renamed_gold_snapshot = gold.gold_snapshot.__class__(
        belief_statuses=renamed_belief_statuses,
        required_authorized_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in gold.gold_snapshot.required_authorized_belief_ids),
        forbidden_authorized_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in gold.gold_snapshot.forbidden_authorized_belief_ids),
        rationale=gold.gold_snapshot.rationale,
    )

    renamed_targets = []
    for target in gold.gold_typed_targets:
        renamed_targets.append(
            TypedRevisionTarget(
                submission_id=rename_string(target.submission_id, old_ns, new_ns),
                action_type=target.action_type,
                target_belief_id=rename_string(target.target_belief_id, old_ns, new_ns),
                target_condition_id=rename_string(target.target_condition_id, old_ns, new_ns),
                replacement_belief_id=rename_string(target.replacement_belief_id, old_ns, new_ns),
                rationale=target.rationale,
                evidence_ids=tuple(rename_string(eid, old_ns, new_ns) for eid in target.evidence_ids),
            )
        )

    renamed_episode = episode.__class__(
        episode_id=new_ns,
        domain=episode.domain,
        failure_type_public_or_controlled=episode.failure_type_public_or_controlled,
        subagent_roles=episode.subagent_roles,
        submissions=renamed_subs,
        downstream_tasks=tuple(renamed_tasks),
        stress_factors=episode.stress_factors,
        split=episode.split,
        protocol_mode=episode.protocol_mode,
        proposal_source=episode.proposal_source,
        metadata=episode.metadata,
    )

    renamed_gold = gold.__class__(
        episode_id=new_ns,
        gold_snapshot=renamed_gold_snapshot,
        gold_typed_targets=tuple(renamed_targets),
        failure_type=gold.failure_type,
        metadata=gold.metadata,
    )

    return renamed_episode, renamed_gold




def load_eval_cases(
    max_cases: int | None = None,
    dataset: str = DATASET_DEV_EXPANSION,
) -> list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    """Loads evaluation episodes for the selected dataset.

    ``dev_expansion`` (default) is the 70-case development diagnostic set and
    keeps the legacy ``_v5`` -> ``__heldout_base`` namespace rename.
    ``paper1_balanced`` is the internal balanced synthetic validation set
    (420 cases); it is loaded verbatim from its deterministic generator.
    """
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(AVAILABLE_DATASETS)}."
        )

    if dataset == DATASET_PAPER1_BALANCED:
        ep_gold_pairs = generate_paper1_balanced_episodes()
        print(f"Successfully loaded {len(ep_gold_pairs)} episodes from paper1_balanced.")
        processed_cases = list(ep_gold_pairs)
    else:
        ep_gold_pairs = generate_expanded_episodes()
        print(f"Successfully loaded {len(ep_gold_pairs)} episodes from dev_expansion.")
        processed_cases = []
        for ep, gold in ep_gold_pairs:
            if ep.episode_id.endswith("_v5"):
                ep_renamed, gold_renamed = rename_episode_and_gold(ep, gold)
                processed_cases.append((ep_renamed, gold_renamed))
            else:
                processed_cases.append((ep, gold))

    if max_cases is not None:
        processed_cases = processed_cases[:max_cases]
        print(f"Restricted to first {len(processed_cases)} cases via max_cases.")
    return processed_cases
