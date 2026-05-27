import json

from retracemem.adapters.memora_adapter import MemoraAdapter
from retracemem.adapters.stale_adapter import StaleAdapter


def test_stale_discovers_and_loads_main_records(tmp_path) -> None:
    main_path = tmp_path / "STALE" / "outputs" / "demo_T1_MAIN.json"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        json.dumps(
            [
                {
                    "uid": "sample-1",
                    "M_old": "I bike to work.",
                    "M_new": "My bike was stolen.",
                    "query_time": "2025-01-01 09:00",
                    "probing_queries": {
                        "dim1_query": "Can I still bike?",
                        "dim2_query": "Since I bike to work, what route?",
                        "dim3_query": "How should I commute?",
                    },
                    "haystack_session": [["old"], ["new"]],
                    "timestamps": ["2024-12-31 08:00", "2025-01-01 08:00"],
                    "explanation": "The new fact invalidates the old commute.",
                }
            ]
        ),
        encoding="utf-8",
    )

    adapter = StaleAdapter(tmp_path / "STALE")

    assert adapter.exists()
    assert adapter.discover_main_files() == [main_path]

    samples = adapter.load_records()

    assert len(samples) == 1
    assert samples[0]["sample_id"] == "sample-1"
    assert samples[0]["uid"] == "sample-1"
    assert samples[0]["M_old"] == "I bike to work."
    assert samples[0]["M_new"] == "My bike was stolen."
    assert samples[0]["query_time"] == "2025-01-01 09:00"
    assert samples[0]["haystack_session"] == [["old"], ["new"]]
    assert samples[0]["sessions"] == [["old"], ["new"]]
    assert samples[0]["timestamps"] == ["2024-12-31 08:00", "2025-01-01 08:00"]
    assert samples[0]["probing_queries"]["dim1_query"] == "Can I still bike?"


def test_stale_degrades_without_data_files(tmp_path) -> None:
    adapter = StaleAdapter(tmp_path / "missing")

    assert not adapter.exists()
    assert adapter.discover() == []
    assert adapter.load_records() == []


def test_stale_degrades_on_invalid_json(tmp_path) -> None:
    main_path = tmp_path / "STALE" / "outputs" / "broken_MAIN.json"
    main_path.parent.mkdir(parents=True)
    main_path.write_text("{", encoding="utf-8")

    assert StaleAdapter(tmp_path / "STALE").load_records(main_path) == []


def test_memora_discovers_roots_and_loads_chronological_sessions(tmp_path) -> None:
    persona_root = tmp_path / "Memora" / "data" / "weekly" / "software_engineer"
    conversations = persona_root / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "session_0002.json").write_text(
        json.dumps(
            {
                "session_id": 2,
                "session_type": "memory_update",
                "operation": "update",
                "operation_details": {"field": "todo"},
                "date": "2025-06-02",
                "persona": "software_engineer",
                "conversation": [{"turn": 1, "speaker": "user_agent", "message": "Updated", "share_memory": True}],
            }
        ),
        encoding="utf-8",
    )
    (conversations / "session_0001.json").write_text(
        json.dumps(
            {
                "session_id": 1,
                "session_type": "activity",
                "operation": "add",
                "operation_details": {"field": "todo"},
                "date": "2025-06-01",
                "persona": "software_engineer",
                "conversation": [{"turn": 1, "speaker": "user_agent", "message": "Added", "share_memory": True}],
            }
        ),
        encoding="utf-8",
    )
    questions_path = persona_root / "evaluation_questions_software_engineer.json"
    questions_path.write_text(
        json.dumps(
            {
                "persona": "software_engineer",
                "date_range": {"start_date": "2025-06-01", "end_date": "2025-06-07"},
                "questions": {
                    "remembering": [
                        {
                            "question_id": "q1",
                            "question": "What remains?",
                            "question_date": "2025-06-07",
                            "memory_evidence": {"session_id": 1},
                            "forgetting_evidence": {},
                            "evaluation": {
                                "evaluation_questions": [
                                    {
                                        "evaluation_question_id": "q1_eval",
                                        "evaluation_question": "Does it mention the task?",
                                        "expected_answer": "yes",
                                        "evaluation_type": "memory_presence",
                                    }
                                ],
                                "total_evaluation_questions": 1,
                            },
                        }
                    ],
                    "reasoning": [],
                    "recommending": [],
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = MemoraAdapter(tmp_path / "Memora")

    roots = adapter.discover_data_roots()
    sessions = adapter.load_sessions("weekly", "software_engineer")
    questions = adapter.load_evaluation_questions("weekly", "software_engineer")

    assert len(roots) == 1
    assert roots[0]["period"] == "weekly"
    assert roots[0]["persona_id"] == "software_engineer"
    assert roots[0]["evaluation_questions_path"] == questions_path
    assert [session["session_id"] for session in sessions] == [1, 2]
    assert sessions[0]["conversation"][0]["message"] == "Added"
    assert len(questions) == 1
    assert questions[0]["persona_id"] == "software_engineer"
    assert questions[0]["period"] == "weekly"
    assert questions[0]["task_bucket"] == "remembering"
    assert questions[0]["question"] == "What remains?"
    assert questions[0]["evaluation"]["total_evaluation_questions"] == 1


def test_memora_degrades_without_data_files(tmp_path) -> None:
    adapter = MemoraAdapter(tmp_path / "missing")

    assert not adapter.exists()
    assert adapter.discover() == []
    assert adapter.load_sessions("weekly", "software_engineer") == []
    assert adapter.load_evaluation_questions("weekly", "software_engineer") == []


def test_stale_load_as_records(tmp_path) -> None:
    main_path = tmp_path / "STALE" / "outputs" / "demo_T1_MAIN.json"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        json.dumps(
            [
                {
                    "uid": "sample-1",
                    "M_old": "I bike to work.",
                    "M_new": "My bike was stolen.",
                    "query_time": "2025-01-01 09:00",
                    "probing_queries": {
                        "dim1_query": "Can I still bike?",
                        "dim2_query": "Since I bike to work, what route?",
                    },
                    "haystack_session": [["old"], ["new"]],
                    "timestamps": ["2024-12-31 08:00", "2025-01-01 08:00"],
                    "explanation": "The new fact invalidates the old commute.",
                }
            ]
        ),
        encoding="utf-8",
    )

    adapter = StaleAdapter(tmp_path / "STALE")
    ev_records, q_records = adapter.load_as_records()

    assert len(ev_records) == 2
    assert ev_records[0].evidence_id == "sample-1_evidence_0"
    assert ev_records[0].session_id == "sample-1_session_0"
    assert ev_records[0].text == "old"
    assert ev_records[0].timestamp == "2024-12-31 08:00"
    assert ev_records[1].text == "new"

    assert len(q_records) == 2
    assert q_records[0].query_id == "sample-1_dim1_query"
    assert q_records[0].query_text == "Can I still bike?"
    assert q_records[0].timestamp == "2025-01-01 09:00"
    assert q_records[0].metadata["M_old"] == "I bike to work."
    assert q_records[0].metadata["M_new"] == "My bike was stolen."


def test_memora_load_as_records(tmp_path) -> None:
    persona_root = tmp_path / "Memora" / "data" / "weekly" / "software_engineer"
    conversations = persona_root / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "session_0001.json").write_text(
        json.dumps(
            {
                "session_id": 1,
                "session_type": "activity",
                "operation": "add",
                "date": "2025-06-01",
                "persona": "software_engineer",
                "conversation": [{"turn": 1, "speaker": "user_agent", "message": "Added", "share_memory": True}],
            }
        ),
        encoding="utf-8",
    )
    questions_path = persona_root / "evaluation_questions_software_engineer.json"
    questions_path.write_text(
        json.dumps(
            {
                "persona": "software_engineer",
                "date_range": {"start_date": "2025-06-01", "end_date": "2025-06-07"},
                "questions": {
                    "remembering": [
                        {
                            "question_id": "q1",
                            "question": "What remains?",
                            "question_date": "2025-06-07",
                            "memory_evidence": {"session_id": 1},
                            "forgetting_evidence": {},
                            "evaluation": {
                                "total_evaluation_questions": 1,
                            },
                        }
                    ],
                    "reasoning": [],
                    "recommending": [],
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = MemoraAdapter(tmp_path / "Memora")
    ev_records, q_records = adapter.load_as_records("weekly", "software_engineer")

    assert len(ev_records) == 1
    assert ev_records[0].evidence_id == "software_engineer_weekly_session_1"
    assert ev_records[0].text == "user_agent: Added"
    assert ev_records[0].timestamp == "2025-06-01"

    assert len(q_records) == 1
    assert q_records[0].query_id == "software_engineer_weekly_q1"
    assert q_records[0].query_text == "What remains?"
    assert q_records[0].timestamp == "2025-06-07"
    assert q_records[0].metadata["task_bucket"] == "remembering"


def test_cupmem_adapter_subclass(tmp_path) -> None:
    from retracemem.adapters.cupmem_adapter import CUPMemAdapter
    adapter = CUPMemAdapter(tmp_path / "STALE")
    assert not adapter.exists()

