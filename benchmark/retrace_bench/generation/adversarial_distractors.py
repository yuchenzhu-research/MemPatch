import random
from typing import Any, Dict, List


def generate_version_distractor(rng: random.Random, scenario_id: str, case_id: str, other_scope: str, frame: Dict[str, Any], topic: str) -> Dict[str, Any]:
    artifact = frame["artifact"]
    # Generates a distractor event that is true for another scope/version but not the current stable one.
    return {
        "event_id": f"e-{scenario_id}-distractor-ver",
        "actor_role": "user",
        "trust_level": "trusted",
        "visibility_scope": other_scope,
        "event_type": "comment",
        "text": f"User notes that on nightly build in {other_scope}, the {topic} update has been applied to {case_id} and the new {artifact} is successfully active.",
        "related_memory_ids": []
    }


def generate_authority_distractor(rng: random.Random, scenario_id: str, case_id: str, scope: str, frame: Dict[str, Any], topic: str) -> Dict[str, Any]:
    # Generates an untrusted claim that conflicts with verified sources
    return {
        "event_id": f"e-{scenario_id}-distractor-auth",
        "actor_role": "user",
        "trust_level": "untrusted",
        "visibility_scope": scope,
        "event_type": "comment",
        "text": f"An unverified user posted on the chat channel that the {topic} conflict for {case_id} was resolved and the old setting is obsolete.",
        "related_memory_ids": []
    }


def generate_rollback_distractor(rng: random.Random, scenario_id: str, case_id: str, scope: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    artifact = frame["artifact"]
    return {
        "event_id": f"e-{scenario_id}-distractor-rollback",
        "actor_role": "reviewer",
        "trust_level": "trusted",
        "visibility_scope": scope,
        "event_type": "review",
        "text": f"Reviewer proposed to revert the latest patch on {case_id} to restore the previous {artifact} configurations temporarily.",
        "related_memory_ids": []
    }


def generate_ci_distractor(rng: random.Random, scenario_id: str, case_id: str, scope: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": f"e-{scenario_id}-distractor-ci",
        "actor_role": "ci",
        "trust_level": "verified",
        "visibility_scope": scope,
        "event_type": "ci",
        "text": f"Continuous integration checks failed for the performance optimization branch on {case_id}.",
        "related_memory_ids": []
    }
