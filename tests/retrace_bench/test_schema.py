from benchmark.retrace_bench.schemas import DialogueTurn, MemoryEntry, ProbeQuery, Scenario, Prediction
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, FinalStatus, ProbeType


def test_dialogue_turn():
    turn = DialogueTurn(speaker="agent_A", text="test text")
    assert turn.speaker == "agent_A"
    assert turn.text == "test text"
    assert turn.timestamp is None


def test_memory_entry():
    entry = MemoryEntry(entry_id="b1", content="Db leak", entry_type="belief")
    assert entry.entry_id == "b1"
    assert entry.entry_type == "belief"


def test_probe_query():
    query = ProbeQuery(
        query_id="q1",
        probe_type=ProbeType.STATE_RESOLUTION,
        question="What status?",
        options={"A": "AUTHORIZED", "B": "BLOCKED"},
        gold_answer="A"
    )
    assert query.query_id == "q1"
    assert query.gold_answer == "A"
