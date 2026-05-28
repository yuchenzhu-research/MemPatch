from retracemem.adapters.memora_oracle_diagnostic import analyze_stage_a_failure_modes


def test_stage_a_failure_analysis_counts_provenance_patterns():
    report = {
        "questions_executed": 1,
        "errors": [],
        "rows": [
            {
                "question_id": "q1",
                "persona": "p1",
                "candidate_roles": {
                    "m1": "memory_presence",
                    "f1": "forgetting_absence",
                    "f2": "forgetting_absence",
                    "f3": "forgetting_absence",
                },
                "stage_a": {
                    "authorized_belief_ids": ["m1", "f1", "f2"],
                    "excluded_belief_ids": ["f3"],
                    "provenance": {
                        "fine_grained_statuses": {
                            "m1": "AUTHORIZED",
                            "f1": "AUTHORIZED",
                            "f2": "AUTHORIZED",
                            "f3": "UNRESOLVED",
                        },
                        "edge_proposals": [
                            {
                                "belief_id": "f1",
                                "target_id": "f1",
                                "edge_type": "REAFFIRMS",
                                "admitted": True,
                                "gate_reason": "ok",
                            },
                            {
                                "belief_id": "f2",
                                "target_id": "f2",
                                "edge_type": "UNCERTAIN",
                                "admitted": False,
                                "gate_reason": "bad",
                            },
                        ],
                        "defeat_paths": [],
                    },
                },
            }
        ],
    }

    analysis = analyze_stage_a_failure_modes(report)
    aggregate = analysis["aggregate"]
    assert aggregate["forgetting_false_positive_total"] == 2
    assert aggregate["reaffirms_proposed"] == 1
    assert aggregate["uncertain_proposed"] == 1
    assert aggregate["rejected_edge"] == 1
    assert aggregate["admitted_edge_but_final_authorized"] == 1
    assert aggregate["no_edge_proposed"] == 0
    assert len(analysis["false_positives"]) == 2
    assert len(analysis["correctly_excluded_forgetting"]) == 1
    assert analysis["manual_annotation_schema"]["classification"]


def test_stage_a_failure_analysis_counts_no_edge_false_positive():
    report = {
        "questions_executed": 1,
        "errors": [],
        "rows": [
            {
                "question_id": "q1",
                "persona": "p1",
                "candidate_roles": {"f1": "forgetting_absence"},
                "stage_a": {
                    "authorized_belief_ids": ["f1"],
                    "excluded_belief_ids": [],
                    "provenance": {
                        "fine_grained_statuses": {"f1": "AUTHORIZED"},
                        "edge_proposals": [],
                        "defeat_paths": [],
                    },
                },
            }
        ],
    }
    analysis = analyze_stage_a_failure_modes(report)
    assert analysis["aggregate"]["forgetting_false_positive_total"] == 1
    assert analysis["aggregate"]["no_edge_proposed"] == 1
    assert analysis["false_positives"][0]["belief_id"] == "f1"
