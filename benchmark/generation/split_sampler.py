"""Stratified split sampling for MemPatch v1.3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

from benchmark.generation.blueprints import (
    V13BlueprintInstance,
    V13DecisionVariant,
    V13PatternFamily,
    all_variants,
    pattern_families_for_decision,
    variants_for_decision,
)
from benchmark.general_taxonomy import DECISIONS

SPLIT_SEED_NAMESPACE = "mempatch_v13"

# Standard ML splits: train (SFT) / validation (dev eval) / test (held-out eval).
SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (200_001, 202_700),
    "validation": (202_701, 203_500),
    "test": (203_501, 204_000),
}

PILOT_QUOTAS: dict[str, dict[str, int]] = {
    "train": {
        "use_current_memory": 100,
        "mark_unresolved": 100,
        "ask_clarification": 100,
        "escalate": 100,
        "refuse_due_to_policy": 100,
    },
    "validation": {
        "use_current_memory": 20,
        "mark_unresolved": 20,
        "ask_clarification": 20,
        "escalate": 20,
        "refuse_due_to_policy": 20,
    },
    "test": {
        "use_current_memory": 20,
        "mark_unresolved": 20,
        "ask_clarification": 20,
        "escalate": 20,
        "refuse_due_to_policy": 20,
    },
}

FULL_QUOTAS: dict[str, dict[str, int]] = {
    "train": {
        "use_current_memory": 600,
        "mark_unresolved": 600,
        "ask_clarification": 600,
        "escalate": 600,
        "refuse_due_to_policy": 300,
    },
    "validation": {
        "use_current_memory": 400,
        "mark_unresolved": 150,
        "ask_clarification": 100,
        "escalate": 75,
        "refuse_due_to_policy": 75,
    },
    "test": {
        "use_current_memory": 150,
        "mark_unresolved": 100,
        "ask_clarification": 100,
        "escalate": 75,
        "refuse_due_to_policy": 75,
    },
}


@dataclass(frozen=True)
class SampledBlueprint:
    blueprint: V13BlueprintInstance
    variant: V13DecisionVariant
    family: V13PatternFamily


def _variant_cycle(decision: str, index: int) -> tuple[V13PatternFamily, V13DecisionVariant]:
    pool = variants_for_decision(decision)
    if not pool:
        raise ValueError(f"no variants registered for {decision}")
    families_by_variant = {v.variant_id: f for f, v in all_variants()}
    variant = pool[index % len(pool)]
    family = families_by_variant[variant.variant_id]
    return family, variant


def _scenario_num(split: str, offset: int) -> int:
    start, _end = SPLIT_RANGES[split]
    return start + offset


def _blueprint_id(split: str, scenario_num: int, variant_id: str) -> str:
    payload = f"{SPLIT_SEED_NAMESPACE}:{split}:{scenario_num}:{variant_id}"
    return "blueprint-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def sample_split(
    split: str,
    quotas: dict[str, int],
    *,
    difficulty: str = "L3",
) -> Iterator[SampledBlueprint]:
    """Yield blueprint instances to hit per-decision quotas with pattern rotation."""
    offset = 0
    for decision in DECISIONS:
        count = quotas.get(decision, 0)
        patterns = pattern_families_for_decision(decision)
        if count <= 0:
            continue
        for i in range(count):
            family, variant = _variant_cycle(decision, i)
            scenario_num = _scenario_num(split, offset)
            offset += 1
            blueprint = V13BlueprintInstance(
                blueprint_id=_blueprint_id(split, scenario_num, variant.variant_id),
                pattern=family.pattern,
                pattern_trap_type=variant.pattern_trap_type,
                decision_variant=variant.variant_id,
                decision_triggers=list(variant.triggers),
                split=split,
                split_seed_namespace=SPLIT_SEED_NAMESPACE,
                scenario_num=scenario_num,
                difficulty=difficulty,
                params={"pattern_family_index": patterns.index(family.pattern) if family.pattern in patterns else 0},
            )
            yield SampledBlueprint(blueprint=blueprint, variant=variant, family=family)


def pilot_blueprints() -> list[SampledBlueprint]:
    out: list[SampledBlueprint] = []
    for split, quotas in PILOT_QUOTAS.items():
        out.extend(sample_split(split, quotas))
    return out
