from __future__ import annotations

from typing import List, Tuple, Dict, Any
from retracemem.evaluation.multiagent.data.episodes_fc_dev import get_fc_dev_episodes
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    StageCTrainingExample,
    TypedRevisionTarget,
)

CANONICAL_ACTIONS = {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}


def validate_training_example(example: StageCTrainingExample) -> None:
    """Enforce fair validation constraints for a training example."""
    # 1. Verify split is development_only
    if example.split != "development_only":
        raise ValueError(f"Example {example.example_id} must have split='development_only', got {example.split}")
    
    # 2. Verify scientific_status
    status = example.metadata.get("scientific_status")
    if status != "pipeline_validation_only":
        raise ValueError(f"Example {example.example_id} must be tagged with scientific_status='pipeline_validation_only'")
        
    # 3. Verify method_visible_input contains no edges or gold statuses
    sub = example.method_visible_input
    if hasattr(sub, "candidate_edges") or "candidate_edges" in sub.metadata:
         raise ValueError(f"Submission {sub.submission_id} contains candidate edges in visible inputs.")
    if "gold_snapshot" in sub.metadata:
         raise ValueError(f"Submission {sub.submission_id} contains gold snapshot in visible inputs.")
         
    # 4. Verify target actions are canonical and targets are grounded
    for t in example.targets:
        if t.action_type not in CANONICAL_ACTIONS:
            raise ValueError(f"Target action {t.action_type} is not canonical in example {example.example_id}")
        
        # Ground check: targets must refer to something meaningful
        if not t.target_belief_id and not t.target_condition_id and t.action_type != "NO_REVISION":
            raise ValueError(f"Target in example {example.example_id} must ground to a belief_id or condition_id for action {t.action_type}")
            
        # If SUPERSEDES, it must have a replacement belief
        if t.action_type == "SUPERSEDES" and not t.replacement_belief_id:
            raise ValueError(f"SUPERSEDES target in example {example.example_id} requires replacement_belief_id")


def build_stagec_dataset() -> List[StageCTrainingExample]:
    """Convert the 14 E1 development seeds into StageCTrainingExamples."""
    episodes = get_fc_dev_episodes()
    examples = []
    
    for ep, gold, _ in episodes:
        # Map targets by submission_id
        targets_by_sub: Dict[str, List[TypedRevisionTarget]] = {}
        for t in gold.gold_typed_targets:
            targets_by_sub.setdefault(t.submission_id, []).append(t)
            
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            
            # Fetch targets or default to NO_REVISION if none
            sub_targets = targets_by_sub.get(sub.submission_id, [])
            if not sub_targets:
                # Add default NO_REVISION target for submissions with no active labels
                sub_targets = [
                    TypedRevisionTarget(
                        submission_id=sub.submission_id,
                        action_type="NO_REVISION",
                        rationale="Default no-revision target.",
                    )
                ]
                
            metadata = {
                "scientific_status": "pipeline_validation_only",
                "contains_gold_in_user_input": False,
                "training_eligible": False,
            }
            
            ex = StageCTrainingExample(
                example_id=example_id,
                episode_id=ep.episode_id,
                submission_id=sub.submission_id,
                method_visible_input=sub,
                targets=tuple(sub_targets),
                split="development_only",
                domain=ep.domain,
                failure_type=gold.failure_type or ep.failure_type_public_or_controlled,
                label_source="human_authored_seed",
                metadata=metadata,
            )
            
            validate_training_example(ex)
            examples.append(ex)
            
    return examples
