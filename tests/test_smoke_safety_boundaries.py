from __future__ import annotations

from pathlib import Path

from retracemem.adapters.memora_wrapper import ReTraceMemorySystem


ROOT = Path(__file__).resolve().parents[1]


def test_stale_runner_labels_smoke_and_avoids_reference_fallback_writes() -> None:
    text = (ROOT / "scripts" / "run_stale_official_eval.py").read_text(encoding="utf-8")

    assert "STALE ADAPTER SMOKE/DRY-RUN" in text
    assert "not an official STALE result" in text
    assert "--allow-official-live" in text
    assert "outputs" in text and "stale_smoke" in text
    assert 'reference" / "STALE" / "outputs"' not in text


def test_memora_runner_labels_smoke_and_requires_live_opt_in() -> None:
    text = (ROOT / "scripts" / "run_memora_official_eval.py").read_text(encoding="utf-8")

    assert "MEMORA ADAPTER SMOKE/DRY-RUN" in text
    assert "not an official Memora result" in text
    assert "--allow-official-live" in text
    assert "Refusing live Memora execution" in text


def test_memora_wrapper_reports_development_fixture_mode() -> None:
    wrapper = ReTraceMemorySystem("smoke_user")
    result = wrapper.add_conversation_to_memory(
        {
            "session_id": "1",
            "date": "2026-05-28",
            "conversation": [{"speaker": "user", "message": "I prefer tea."}],
        }
    )

    assert result["status"] == "success"
    assert result["pipeline_mode"] == "development_fixture"
