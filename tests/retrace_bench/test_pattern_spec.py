from benchmark.retrace_bench.generation.hard_plus_blueprints import build_deterministic_scenario
from benchmark.retrace_bench.generation.pattern_spec import (
    PATTERN_SPEC,
    resolve_pattern_binding,
    validate_pattern_semantics,
)
from scripts.validate_retrace_bench_dataset import validate_one


def test_failure_mode_derived_from_pattern_not_global_index():
    sc = build_deterministic_scenario(1, "hard", 2027, split_count=150)
    allowed = set(PATTERN_SPEC[sc["pattern"]]["allowed_failure_modes"])
    assert sc["primary_failure_mode"] in allowed
    assert sc["hidden_gold"]["expected_failure_diagnosis"] == sc["primary_failure_mode"]
    assert sc["hidden_gold"]["expected_failure_diagnosis"] != "under_update" or sc["pattern"] == "ci_failed_after_claim"


def test_pattern_binding_variants_are_semantically_valid():
    for index in range(30):
        sc = build_deterministic_scenario(index, "hard", 2027, split_count=30)
        errors = validate_pattern_semantics(sc, sc["hidden_gold"])
        assert errors == [], errors


def test_validator_accepts_bound_scenario():
    sc = build_deterministic_scenario(7, "hard", 2027, split_count=30)
    errors, warnings = validate_one(sc)
    assert errors == []


def test_scope_leakage_pattern_forces_scope_diagnosis():
    binding = resolve_pattern_binding("version_scope_leakage", 0)
    assert binding.failure_mode == "scope_leakage"


def test_security_policy_pattern_forces_policy_violation():
    binding = resolve_pattern_binding("security_policy_override", 0)
    assert binding.failure_mode == "policy_violation"
    assert binding.expected_decision == "refuse_due_to_policy"


def test_expected_answer_does_not_leak_pattern_or_failure_labels():
    for index in range(30):
        sc = build_deterministic_scenario(index, "hard", 2027, split_count=30)
        answer = sc["hidden_gold"]["expected_answer"].lower()
        pattern = sc["pattern"]
        failure = sc["hidden_gold"]["expected_failure_diagnosis"]
        assert pattern.replace("_", " ") not in answer.replace("_", " ")
        assert failure.replace("_", " ") not in answer.replace("_", " ")
