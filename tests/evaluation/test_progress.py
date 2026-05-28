from io import StringIO

from retracemem.evaluation.progress import ProgressReporter, ProgressSnapshot


def test_progress_line_mode_reports_safe_counters() -> None:
    stream = StringIO()
    reporter = ProgressReporter(mode="line", stream=stream)
    reporter.plan(
        "STALE development-only run | scenarios=3 | provider=gemini | model=gemini-3.5-flash"
    )
    reporter.update(
        ProgressSnapshot(
            phase="generating",
            stage="Stage A",
            scenarios_done=1,
            scenarios_total=3,
            queries_done=3,
            queries_total=9,
            semantic_invocations=3,
            outbound_network_calls=3,
            max_calls=1000,
            cache_hits=0,
            cache_misses=3,
            tokens_from_outbound_calls=1200,
            max_tokens=2000000,
            current_id="sample-1 unsafe text should not appear",
        )
    )
    reporter.done("Report written to outputs/example/report.json")
    reporter.close()

    text = stream.getvalue()
    assert "[PLAN] STALE development-only run" in text
    assert "[Stage A] scenarios 1/3" in text
    assert "queries 3/9" in text
    assert "outbound=3/1000" in text
    assert "tokens=1200/2000000" in text
    assert "unsafe text should not appear" not in text
    assert "GEMINI_API_KEY" not in text
    assert "Bearer " not in text
