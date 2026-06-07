"""Tests for MemPatch v1.3 blueprint registry and sampling."""

from __future__ import annotations

from benchmark.generation.blueprints import validate_registry
from benchmark.generation.split_sampler import PILOT_QUOTAS, pilot_blueprints


def test_v13_registry_pilot_ready() -> None:
    errors = validate_registry()
    assert errors == [], errors


def test_v13_pilot_quotas() -> None:
    samples = pilot_blueprints()
    assert len(samples) == sum(sum(q.values()) for q in PILOT_QUOTAS.values())
    by_split: dict[str, int] = {}
    for s in samples:
        by_split[s.blueprint.split] = by_split.get(s.blueprint.split, 0) + 1
    assert by_split["train"] == 500
    assert by_split["validation"] == 100
    assert by_split["test"] == 100
