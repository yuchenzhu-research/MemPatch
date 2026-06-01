from benchmark.retrace_bench.generation.expand_scenarios import expand_templates


def test_generation_smoke():
    scenarios = expand_templates(num_scenarios=10, seed=7)
    assert len(scenarios) == 10
    
    # Check that each scenario has 4 probe queries
    for scen in scenarios:
        assert len(scen.probe_queries) == 4
        assert scen.scenario_id.startswith("scen_")
        
        # Verify topology exists
        assert "requires" in scen.memory_topology
