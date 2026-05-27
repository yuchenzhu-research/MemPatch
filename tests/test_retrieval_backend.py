from retracemem.backends import RetrievalBaselineBackend


def test_retrieval_baseline_stores_and_searches_raw_session_text() -> None:
    backend = RetrievalBaselineBackend()
    backend.ingest_session(
        "u1",
        {
            "id": "s1",
            "messages": [
                {"role": "user", "content": "I commute by bicycle on weekdays."},
                {"role": "assistant", "content": "Noted."},
            ],
        },
    )
    backend.ingest_session("u1", {"id": "s2", "text": "The user now drives to work."})

    results = backend.search("u1", "bicycle commute", limit=1)

    assert len(results) == 1
    assert results[0]["id"] == "s1_evidence_0001"
    assert "user: I commute by bicycle" in results[0]["text"]
    assert results[0]["match_terms"] == ["bicycle", "commute"]


def test_retrieval_baseline_answer_is_deterministic_shell() -> None:
    backend = RetrievalBaselineBackend()
    retrieved = [{"id": "e1", "text": "The user likes tea."}]

    answer = backend.answer("u1", "What does the user like?", retrieved)

    assert answer == (
        "method: retrieval_baseline\n"
        "query: What does the user like?\n"
        "retrieved_evidence_ids: e1\n"
        "answer: deterministic retrieval baseline; inspect retrieved evidence.\n"
        "retrieved_evidence:\n"
        "1. [e1] The user likes tea."
    )


def test_retrieval_baseline_reset_user_clears_state() -> None:
    backend = RetrievalBaselineBackend()
    backend.ingest_session("u1", {"id": "s1", "text": "Persistent note about hiking."})

    assert backend.search("u1", "hiking")

    backend.reset_user("u1")

    assert backend.search("u1", "hiking") == []
