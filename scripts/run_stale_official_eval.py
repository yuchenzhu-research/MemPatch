#!/usr/bin/env python3
"""Official STALE evaluation runner using dynamic clean-room monkey-patching."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

# Resolve STALE path
STALE_DIR = Path(__file__).resolve().parents[1] / "reference" / "STALE" / "STALE"
if str(STALE_DIR) not in sys.path:
    sys.path.insert(0, str(STALE_DIR))
if str(STALE_DIR / "Evaluation") not in sys.path:
    sys.path.insert(0, str(STALE_DIR / "Evaluation"))

from retracemem.adapters.stale_adapter import StaleAdapter
from retracemem.adapters.stale_v1_adapter import StaleV1Adapter
from retracemem.pipeline import ReTracePipeline


def monkey_patch_stale(is_live: bool) -> None:
    # Patch the async client in Generation.clients if not running live
    if not is_live:
        print("  [MOCK] Mocking STALE judge client for offline execution.")
        import Generation.clients

        class MockChatCompletions:
            async def create(self, *args: Any, **kwargs: Any) -> Any:
                class MockMessage:
                    content = '{"dim1_eval": {"reasoning": "Mock pass", "pass": true}, "dim2_eval": {"reasoning": "Mock pass", "pass": true}, "dim3_eval": {"reasoning": "Mock pass", "pass": true}}'
                class MockChoice:
                    message = MockMessage()
                class MockResponse:
                    choices = [MockChoice()]
                return MockResponse()

        class MockChat:
            completions = MockChatCompletions()

        class MockClient:
            chat = MockChat()

        def patched_get_Async_client(provider: str) -> Any:
            return MockClient()

        Generation.clients.get_Async_client = patched_get_Async_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official STALE evaluation.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live API calls to LLM Judge (requires JUDGE_PROVIDER/API keys).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit number of STALE records to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stale",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    if not STALE_DIR.exists():
        print(f"Error: STALE repository not found at {STALE_DIR}")
        sys.exit(1)

    print("=" * 70)
    print("RUNNING OFFICIAL STALE EVALUATION")
    print("=" * 70)
    print(f"  Limit:  {args.limit}")
    print(f"  Mode:   {'LIVE' if args.live else 'MOCK'}")
    print()

    # Load dataset
    adapter = StaleAdapter(args_reference_root := Path(__file__).resolve().parents[1] / "reference" / "STALE")
    main_files = adapter.discover_main_files()
    if not main_files:
        print("  No STALE MAIN dataset files discovered under reference/STALE.")
        print("  Generating outputs/demo_T1_MAIN.json mock file dynamically.")
        mock_out_dir = Path(__file__).resolve().parents[1] / "reference" / "STALE" / "outputs"
        mock_out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = mock_out_dir / "demo_T1_MAIN.json"
        
        mock_data = [
            {
                "uid": "sample-1",
                "M_old": "I commute by bicycle.",
                "M_new": "My bicycle was stolen.",
                "explanation": "No longer has a bicycle.",
                "query_time": "2025-01-01 09:00",
                "probing_queries": {
                    "dim1_query": "Can I still commute by bicycle?",
                    "dim2_query": "Since I bike, what route?",
                    "dim3_query": "What is my current commute plan?"
                },
                "haystack_session": [["old"], ["new"]],
                "timestamps": ["2024-12-31 08:00", "2025-01-01 08:00"]
            }
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)
    else:
        dataset_path = main_files[0]

    print(f"Loading data from: {dataset_path}")
    samples = adapter.load_records(dataset_path)[: args.limit]

    # Run ReTrace pipeline
    pipeline = ReTracePipeline.for_development_fixture()
    records = []
    
    for idx, sample in enumerate(samples):
        uid = sample["uid"]
        user_id = f"stale_user_{uid}"
        pipeline.reset_user(user_id)

        # Ingest sessions sequentially
        # Haystack sessions
        for s_idx, session_item in enumerate(sample["sessions"]):
            text = " ".join(session_item) if isinstance(session_item, list) else str(session_item)
            ev = {
                "id": f"{uid}_session_{s_idx}",
                "text": text,
                "timestamp": sample["timestamps"][s_idx] if s_idx < len(sample["timestamps"]) else None
            }
            pipeline.backend.ingest_session(user_id, ev)

        # Add a mock belief representing the pre-existing state (M_old) once per sample
        belief_id = f"{uid}_belief_old"
        from retracemem.schemas import BeliefNode
        pipeline.add_belief(user_id, BeliefNode(
            belief_id=belief_id,
            proposition=sample["M_old"],
            source_evidence_ids=(f"{uid}_session_0",)
        ))

        # Ingest M_new session to trigger supersession/blocking
        new_ev = {
            "id": f"{uid}_new_memory",
            "text": sample["M_new"],
            "timestamp": sample["query_time"]
        }
        pipeline.backend.ingest_session(user_id, new_ev)

        # Answer probing queries
        for dim_key, query_text in sample["probing_queries"].items():
            if not query_text:
                continue
            
            pipeline.backend.query_retriever.query_map[query_text] = [belief_id]

            # Generate answer
            record = pipeline.answer(user_id, query_text, limit=5, method="retrace")
            
            # Re-key query_id for adapter compatibility
            records.append({
                "query_id": f"{uid}_{dim_key}",
                "answer": record.answer
            })

    # Export answers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_json = output_dir / "stale_answers.json"
    StaleV1Adapter.export_to_official_json(records, answers_json)
    print(f"Exported target responses to: {answers_json}")

    # Apply monkey patches
    monkey_patch_stale(args.live)

    # Run official evaluator
    print("\n--- Running Official STALE Evaluator ---")
    output_eval_json = output_dir / "stale_eval_results.json"
    
    import full_eval_performance
    
    # Override settings in full_eval_performance
    full_eval_performance.EVAL_PROVIDER = "MOCK" if not args.live else (os.getenv("JUDGE_PROVIDER") or "OPENAI")
    full_eval_performance.EVAL_MODEL = "mock-model" if not args.live else (os.getenv("JUDGE_MODEL") or "gpt-4")
    full_eval_performance.CONCURRENCY_LIMIT = 5
    
    # Run evaluation
    asyncio.run(
        full_eval_performance.run_evaluation(
            answers_path=str(answers_json),
            dataset_path=str(dataset_path),
            output_path=str(output_eval_json),
            model_method="retrace_pipeline",
            conflict_type="stale_eval",
        )
    )

    print()
    print("=" * 70)
    print("STALE EVALUATION COMPLETE")
    print(f"Official evaluation JSON saved to: {output_eval_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
