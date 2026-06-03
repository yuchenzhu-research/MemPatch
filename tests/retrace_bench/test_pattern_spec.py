from benchmark.retrace_bench.generation.hard_plus_blueprints import build_deterministic_scenario
from benchmark.retrace_bench.generation.pattern_spec import (
    PATTERN_SPEC,
    resolve_pattern_binding,
    validate_pattern_semantics,
)
from scripts.validate_retrace_bench_dataset import validate_one


def test_failure_mode_derived_from_pattern_not_global_index():
    # Index 1 -> closed_as_duplicate_not_fixed, not FAILURE_MODES[1].
    sc = build_deterministic_scenario(1, "hard", 2027)
    assert sc["pattern"] == "closed_as_duplicate_not_fixed"
    allowed = set(PATTERN_SPEC["closed_as_duplicate_not_fixed"]["allowed_failure_modes"])
    assert sc["primary_failure_mode"] in allowed
    assert sc["hidden_gold"]["expected_failure_diagnosis"] == sc["primary_failure_mode"]
    assert sc["hidden_gold"]["expected_failure_diagnosis"] != "under_update"


def test_pattern_binding_variants_are_semantically_valid():
    for index in range(30):
        sc = build_deterministic_scenario(index, "hard", 2027)
        errors = validate_pattern_semantics(sc, sc["hidden_gold"])
        assert errors == [], errors


def test_validator_accepts_bound_scenario():
    sc = build_deterministic_scenario(7, "hard", 2027)
    errors, warnings = validate_one(sc)
    assert errors == []


def test_scope_leakage_pattern_forces_scope_diagnosis():
    binding = resolve_pattern_binding("version_scope_leakage", 0)
    assert binding.failure_mode == "scope_leakage"


def test_security_policy_pattern_forces_policy_violation():
    binding = resolve_pattern_binding("security_policy_override", 0)
    assert binding.failure_mode == "policy_violation"
    assert binding.expected_decision == "refuse_due_to_policy"
