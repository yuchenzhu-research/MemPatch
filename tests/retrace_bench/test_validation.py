from benchmark.retrace_bench.generation.expand_scenarios import expand_templates
from benchmark.retrace_bench.generation.validate_generated import validate_scenarios
from benchmark.retrace_bench.schemas import Scenario
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily


def test_validation_success():
    scenarios = expand_templates(num_scenarios=5, seed=7)
    report, accepted, rejected = validate_scenarios(scenarios)
    assert report.is_valid
    assert len(accepted) == 5
    assert len(rejected) == 0


def test_validation_failure():
    scenarios = expand_templates(num_scenarios=1, seed=7)
    # Intentionally corrupt a scenario to trigger validation failure
    scen = scenarios[0]
    corrupted_scen = Scenario(
        scenario_id=scen.scenario_id,
        domain=scen.domain,
        revision_family=scen.revision_family,
        conflict_type=scen.conflict_type,
        memory_topology=scen.memory_topology,
        dialogue_history=scen.dialogue_history,
        memory_snapshot=scen.memory_snapshot,
        gold_final_statuses=scen.gold_final_statuses,
        gold_revision_actions=scen.gold_revision_actions,
        probe_queries=[],  # Erase queries to trigger failure
        metadata=scen.metadata
    )
    
    report, accepted, rejected = validate_scenarios([corrupted_scen])
    assert not report.is_valid
    assert len(accepted) == 0
    assert len(rejected) == 1
    assert any("has 0 queries" in err for err in report.errors)
