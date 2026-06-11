from scripts.linux.response_schema_projection import project_response_schema


PUBLIC_VIEW = {
    "public_input": {
        "initial_memory": [
            {"memory_id": "m-target", "text": "Prior state"},
            {"memory_id": "m-noise", "text": "Distractor info: beta only"},
        ],
        "event_trace": [{"event_id": "e-valid"}],
    }
}


def test_projection_repairs_missing_memory_and_invented_evidence() -> None:
    response, repairs = project_response_schema(
        {
            "answer": "Use it.",
            "decision": "use_current_memory",
            "evidence_event_ids": ["e-valid", "e-invented"],
            "failure_diagnosis": "none",
        },
        PUBLIC_VIEW,
    )

    assert response["memory_state"] == {
        "m-target": "current",
        "m-noise": "out_of_scope",
    }
    assert response["evidence_event_ids"] == ["e-valid"]
    assert response["failure_diagnosis"] == "memory_hallucination"
    assert repairs


def test_projection_uses_conservative_default_for_invalid_decision() -> None:
    response, _ = project_response_schema({}, PUBLIC_VIEW)
    assert response["decision"] == "ask_clarification"
    assert response["memory_state"]["m-target"] == "unresolved"
