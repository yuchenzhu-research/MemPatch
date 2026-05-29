import json
from pathlib import Path
from unittest.mock import patch

import pytest

from retracemem.adapters.stale_official_runner import (
    StaleLiveRunConfig,
    StaleOfflineRunConfig,
    _NoEdgeBatchedVerifier,
    answer_probing_queries,
    build_chronological_evidence_chunks,
    build_live_stage_a_pipeline,
    build_live_stage_b_pipeline,
    build_offline_pipeline,
    estimate_live_calls,
    export_official_target_responses,
    ingest_method_visible_sessions,
    run_offline_wiring_demo,
    select_subset,
)
from retracemem.adapters.stale_official_adapter import StaleOfficialAdapter
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.verifier.proposal_strategy import BatchedEvidenceEdgeProposalStrategy


def _record(uid: str, rtype: str = "T1") -> dict:
    return {
        "uid": uid,
        "M_old": "evaluator-only old fact",
        "M_new": "evaluator-only new fact",
        "explanation": "evaluator-only why",
        "probing_queries": {
            "dim1_query": f"q1 for {uid}?",
            "dim2_query": f"q2 for {uid}?",
            "dim3_query": f"q3 for {uid}?",
        },
        "relevant_session_index": [0],
        "timestamps": ["2025-01-01 09:00", "2025-01-02 09:00"],
        "haystack_session": [["session 0 turn"], ["session 1 turn"]],
        "type": rtype,
    }


def _record_with_sessions(uid: str, count: int, rtype: str = "T1") -> dict:
    record = _record(uid, rtype)
    record["timestamps"] = [f"2025-01-{index + 1:02d} 09:00" for index in range(count)]
    record["haystack_session"] = [[f"session {index} turn"] for index in range(count)]
    return record


def _write_dataset(tmp_path: Path, payload: list[dict]) -> Path:
    path = tmp_path / "T1_T2_400_FULL.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_client(tmp_path: Path, name: str) -> CachedLLMClient:
    cache = JSONLCache(str(tmp_path / f"{name}.jsonl"))
    provider = MockLLMProvider(default_response=json.dumps({"verdicts": [], "edges": []}))
    return CachedLLMClient(cache=cache, provider_client=provider)


def test_pipeline_uses_bounded_batched_proposal_strategy(tmp_path: Path) -> None:
    client = _make_client(tmp_path, "c")
    pipeline = build_offline_pipeline(client)
    assert isinstance(pipeline.backend.edge_proposal_strategy, BatchedEvidenceEdgeProposalStrategy)
    assert isinstance(pipeline.backend.edge_proposal_strategy.verifier, _NoEdgeBatchedVerifier)


def test_session_ingestion_runs_once_per_scenario(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    record = StaleOfficialAdapter(path).load()[0]
    client = _make_client(tmp_path, "c")
    pipeline = build_offline_pipeline(client)
    user_id = "stale:retrace:u1"
    ids = ingest_method_visible_sessions(pipeline, user_id, record.method_visible)
    ledger = pipeline.backend.ledgers[user_id]
    assert ids == ["u1:chunk:0", "u1:chunk:1"]
    assert len(ledger) == 2


def test_chunked_ingestion_groups_fifty_sessions_in_order(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record_with_sessions("u1", 50)])
    record = StaleOfficialAdapter(path).load()[0]
    chunks = build_chronological_evidence_chunks(record.method_visible, chunk_size=10)
    assert len(chunks) == 5
    assert chunks[0].metadata["raw_session_indices"] == tuple(range(10))
    assert chunks[-1].metadata["raw_session_indices"] == tuple(range(40, 50))
    assert chunks[0].metadata["timestamps"][0] == "2025-01-01 09:00"
    assert chunks[0].metadata["raw_session_count"] == 10
    assert chunks[0].metadata["chunk_index"] == 0
    assert "session 0 turn" in chunks[0].text
    assert "session 10 turn" not in chunks[0].text


def test_three_queries_reuse_persistent_state(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    record = StaleOfficialAdapter(path).load()[0]
    client = _make_client(tmp_path, "c")
    pipeline = build_offline_pipeline(client)
    user_id = "stale:retrace:u1"
    ingest_method_visible_sessions(pipeline, user_id, record.method_visible)
    before = len(pipeline.backend.ledgers[user_id])
    answers, meta = answer_probing_queries(pipeline, user_id, record.method_visible, method="retrace")
    after = len(pipeline.backend.ledgers[user_id])
    assert after == before  # answering must not append new evidence
    assert set(answers) == {"dim1_response", "dim2_response", "dim3_response"}
    assert set(meta) == {"dim1_meta", "dim2_meta", "dim3_meta"}


def test_method_visible_state_does_not_contain_gold_fields(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    record = StaleOfficialAdapter(path).load()[0]
    client = _make_client(tmp_path, "c")
    pipeline = build_offline_pipeline(client)
    user_id = "stale:retrace:u1"
    ingest_method_visible_sessions(pipeline, user_id, record.method_visible)
    ledger_text = " ".join(ev.text for ev in pipeline.backend.ledgers[user_id].all())
    assert "evaluator-only old fact" not in ledger_text
    assert "evaluator-only new fact" not in ledger_text
    assert "evaluator-only why" not in ledger_text


def test_chunk_text_does_not_inject_gold_fields(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record_with_sessions("u1", 50)])
    record = StaleOfficialAdapter(path).load()[0]
    chunks = build_chronological_evidence_chunks(record.method_visible, chunk_size=10)
    all_text = "\n".join(chunk.text for chunk in chunks)
    assert "evaluator-only old fact" not in all_text
    assert "evaluator-only new fact" not in all_text
    assert "evaluator-only why" not in all_text


def test_stage_b_is_not_raw_haystack_direct_answer_baseline(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    record = StaleOfficialAdapter(path).load()[0]
    client_a = _make_client(tmp_path, "a")
    client_b = _make_client(tmp_path, "b")
    pipeline_a = build_offline_pipeline(client_a)
    pipeline_b = build_offline_pipeline(client_b)
    ingest_method_visible_sessions(pipeline_a, "ua", record.method_visible)
    ingest_method_visible_sessions(pipeline_b, "ub", record.method_visible)
    answer_probing_queries(pipeline_a, "ua", record.method_visible, method="retrace")
    answer_probing_queries(pipeline_b, "ub", record.method_visible, method="directjudge")
    # Stage B must have ingested the same haystack into a persistent backend
    # rather than direct-prompting raw context per query.
    assert len(pipeline_b.backend.ledgers["ub"]) == len(record.method_visible.haystack_sessions)


def test_export_uses_exact_official_schema(tmp_path: Path) -> None:
    rows = [
        {"uid": "u1", "target_model_responses": {
            "dim1_response": "a", "dim2_response": "b", "dim3_response": "c",
        }},
    ]
    out = tmp_path / "x.json"
    export_official_target_responses(rows, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == [
        {"uid": "u1", "target_model_responses": {
            "dim1_response": "a", "dim2_response": "b", "dim3_response": "c",
        }},
    ]


def test_no_network_calls_in_replay(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    config = StaleOfflineRunConfig(
        dataset_path=str(path),
        output_dir=str(tmp_path / "out"),
        limit_t1=1,
        limit_t2=0,
    )
    client_a = _make_client(tmp_path, "a")
    client_b = _make_client(tmp_path, "b")
    result = run_offline_wiring_demo(config, client_a, client_b)
    manifest = result["manifest"]
    assert manifest["live_provider_calls"] is False
    assert manifest["official_model_result"] is False
    assert manifest["official_judge_evaluation_executed"] is False
    assert manifest["dataset_source"] == "STALEproj/STALE"
    assert manifest["dataset_artifact"] == "T1_T2_400_FULL.json"


def test_live_builders_can_share_extraction_client_and_keep_state_isolated(tmp_path: Path) -> None:
    shared_extract = _make_client(tmp_path, "shared_extract")
    stage_a_edges = _make_client(tmp_path, "stage_a_edges")
    stage_a_answer = _make_client(tmp_path, "stage_a_answer")
    stage_b_answer_judge = _make_client(tmp_path, "stage_b_answer_judge")
    config = StaleLiveRunConfig(dataset_path=str(tmp_path / "unused.json"), ingest_chunk_size=10)
    pipeline_a = build_live_stage_a_pipeline(shared_extract, stage_a_edges, stage_a_answer, config)
    pipeline_b = build_live_stage_b_pipeline(shared_extract, stage_b_answer_judge, config)
    assert pipeline_a.backend is not pipeline_b.backend
    assert pipeline_a.backend.extractor.client is shared_extract
    assert pipeline_b.backend.extractor.client is shared_extract
    assert pipeline_a.backend.stores is not pipeline_b.backend.stores


def test_estimate_mode_counts_chunked_shared_extraction_without_network(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record_with_sessions("u1", 50)])
    config = StaleLiveRunConfig(
        dataset_path=str(path),
        limit_t1=1,
        limit_t2=0,
        ingest_chunk_size=10,
    )
    estimate = estimate_live_calls(config)
    assert estimate["zero_api_calls"] is True
    assert estimate["expected_shared_extraction_network_calls"] == 5
    assert estimate["stage_a_edge_call_upper_bound"] == 5
    assert estimate["stage_a_answer_calls"] == 3
    assert estimate["stage_b_directjudge_calls"] == 3
    assert estimate["stage_b_answer_calls"] == 3
    assert estimate["approx_target_method_total_call_upper_bound_excluding_evaluator"] == 19


def test_outputs_under_outputs_directory(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1")])
    out_dir = tmp_path / "outputs" / "demo"
    config = StaleOfflineRunConfig(
        dataset_path=str(path),
        output_dir=str(out_dir),
        limit_t1=1,
        limit_t2=0,
    )
    run_offline_wiring_demo(config, _make_client(tmp_path, "a"), _make_client(tmp_path, "b"))
    assert (out_dir / "stage_a_target_responses.json").is_file()
    assert (out_dir / "stage_b_target_responses.json").is_file()
    assert (out_dir / "wiring_demo_manifest.json").is_file()


def test_partial_failure_writes_truthful_manifest(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [_record("u1"), _record("u2", "T2")])
    config = StaleOfflineRunConfig(
        dataset_path=str(path),
        output_dir=str(tmp_path / "out"),
        limit_t1=1,
        limit_t2=1,
    )
    client_a = _make_client(tmp_path, "a")
    client_b = _make_client(tmp_path, "b")
    # Force Stage B answer call to fail by using a provider that returns invalid
    # JSON only for the directjudge prompt path.
    class FailingDirectJudgeProvider(MockLLMProvider):
        def generate(self, prompt: str, **kwargs):  # type: ignore[override]
            if "verdicts" in prompt.lower():
                self.default_response = "not json"
            else:
                self.default_response = json.dumps({"verdicts": [], "edges": []})
            return super().generate(prompt, **kwargs)

    cache = JSONLCache(str(tmp_path / "fail.jsonl"))
    client_b = CachedLLMClient(cache=cache, provider_client=FailingDirectJudgeProvider(default_response="{}"))

    result = run_offline_wiring_demo(config, client_a, client_b)
    manifest = result["manifest"]
    assert isinstance(manifest["errors"], list)
    assert (Path(config.output_dir) / "wiring_demo_manifest.json").is_file()


def test_select_subset_respects_t1_t2_limits(tmp_path: Path) -> None:
    path = _write_dataset(
        tmp_path,
        [_record("a", "T1"), _record("b", "T2"), _record("c", "T1"), _record("d", "T2")],
    )
    records = StaleOfficialAdapter(path).load()
    selected = select_subset(records, limit_t1=1, limit_t2=2)
    assert [r.method_visible.uid for r in selected] == ["a", "b", "d"]


def test_loads_real_official_dataset_when_available() -> None:
    real_path = Path("data_external/stale_official_frozen/T1_T2_400_FULL.json")
    if not real_path.is_file():
        pytest.skip("Official dataset not present in this environment")
    records = StaleOfficialAdapter(real_path).load()
    assert len(records) == 400
    types = {r.evaluator_only.type for r in records}
    assert types == {"T1", "T2"}


def test_stale_runner_modes_and_ingestion_gating(tmp_path: Path) -> None:
    # 1. Test StaleLiveRunConfig default chunk size is 1
    config = StaleLiveRunConfig()
    assert config.ingest_chunk_size == 1

    # 2. Test run_stale_official_frozen_eval argparse defaults and parser
    import sys
    from unittest.mock import patch
    import scripts.run_stale_official_frozen_eval as run_script

    # Helper to run main with mocked args
    def run_main_with_args(args_list):
        with patch.object(sys, "argv", ["run_stale_official_frozen_eval.py"] + args_list):
            run_script.main()

    # Test estimate mode works and prints JSON without network
    with patch("builtins.print") as mock_print:
        run_main_with_args(["--mode", "estimate", "--dataset-path", str(_write_dataset(tmp_path, [_record("u1")]))])
        assert mock_print.called

    # Test official-eval fails without confirm flag
    with pytest.raises(SystemExit) as exc:
        run_main_with_args(["--mode", "official-eval"])
    assert "requires the explicit opt-in confirmation flag" in str(exc.value)

    # Test official-eval fails if chunk_size > 1 and not allowed
    with pytest.raises(ValueError) as exc_val:
        run_main_with_args(["--mode", "official-eval", "--i-confirm-official-evaluation", "--ingest-chunk-size", "2"])
    assert "Official evaluation must use canonical ingest_chunk_size = 1" in str(exc_val.value)


def test_manifest_canonical_ingestion_flag(tmp_path: Path) -> None:
    # Test canonical_ingestion value in live generation logic
    from retracemem.adapters.stale_official_runner import run_live_stageab_generation
    
    path = _write_dataset(tmp_path, [_record("u1")])
    config_canonical = StaleLiveRunConfig(
        dataset_path=str(path),
        output_dir=str(tmp_path / "out_canonical"),
        limit_t1=1,
        limit_t2=0,
        ingest_chunk_size=1,
    )
    
    # We mock the make_live_client and pipeline builders to avoid network
    with patch("retracemem.adapters.stale_official_runner.make_live_client"), \
         patch("retracemem.adapters.stale_official_runner.build_live_stage_a_pipeline"), \
         patch("retracemem.adapters.stale_official_runner.build_live_stage_b_pipeline"), \
         patch("retracemem.adapters.stale_official_runner.run_scenario", return_value=({}, {}, ["ev1"])):
        
        res = run_live_stageab_generation(config_canonical)
        manifest = res["manifest"]
        assert manifest["canonical_ingestion"] is True
        assert manifest["approximate_chunked_ingestion"] is False

    config_non_canonical = StaleLiveRunConfig(
        dataset_path=str(path),
        output_dir=str(tmp_path / "out_non_canonical"),
        limit_t1=1,
        limit_t2=0,
        ingest_chunk_size=2,
    )

    with patch("retracemem.adapters.stale_official_runner.make_live_client"), \
         patch("retracemem.adapters.stale_official_runner.build_live_stage_a_pipeline"), \
         patch("retracemem.adapters.stale_official_runner.build_live_stage_b_pipeline"), \
         patch("retracemem.adapters.stale_official_runner.run_scenario", return_value=({}, {}, ["ev1"])):
        
        res = run_live_stageab_generation(config_non_canonical)
        manifest = res["manifest"]
        assert manifest["canonical_ingestion"] is False
        assert manifest["approximate_chunked_ingestion"] is True

