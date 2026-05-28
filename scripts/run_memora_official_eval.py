#!/usr/bin/env python3
"""Memora adapter smoke/dry-run runner using dynamic clean-room monkey-patching."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

# Resolve Memora evals path
MEMORA_DIR = Path(__file__).resolve().parents[1] / "reference" / "Memora"
MEMORA_EVALS_DIR = MEMORA_DIR / "evals" / "agent_eval"

if str(MEMORA_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORA_EVALS_DIR))


def monkey_patch_memora(is_live: bool) -> None:
    # 1. Patch argparse to allow 'retrace' as a system choice
    import argparse
    original_add_argument = argparse.ArgumentParser.add_argument

    def patched_add_argument(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 0 and args[0] == "--system" and "choices" in kwargs:
            kwargs["choices"] = list(kwargs["choices"]) + ["retrace"]
        return original_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = patched_add_argument

    # 2. Patch get_memory_system to dynamically register retrace
    import base_evaluator
    from retracemem.adapters.memora_wrapper import ReTraceMemorySystem

    original_get_memory_system = base_evaluator.get_memory_system

    def patched_get_memory_system(system_name: str, user_id: str, **kwargs: Any) -> Any:
        if system_name.lower() == "retrace":
            return ReTraceMemorySystem(user_id, **kwargs)
        return original_get_memory_system(system_name, user_id, **kwargs)

    base_evaluator.get_memory_system = patched_get_memory_system

    # 2. Patch OpenAI client if not running live
    if not is_live:
        print("  [MOCK] Mocking OpenAI chat completion calls for offline execution.")
        try:
            import openai
        except ImportError:
            import types
            mock_openai = types.ModuleType("openai")
            sys.modules["openai"] = mock_openai
            import openai

        class MockChatCompletions:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                class MockMessage:
                    content = "yes\nMocked Memora OpenAI Response."
                class MockChoice:
                    message = MockMessage()
                class MockResponse:
                    choices = [MockChoice()]
                return MockResponse()

        class MockChat:
            completions = MockChatCompletions()

        class MockOpenAI:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            chat = MockChat()

        openai.OpenAI = MockOpenAI  # type: ignore[misc,assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memora adapter smoke/dry-run.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live official-style calls only with --allow-official-live.",
    )
    parser.add_argument(
        "--allow-official-live",
        action="store_true",
        help="Explicit opt-in for future official-style live execution; not used in this task.",
    )
    parser.add_argument(
        "--persona",
        default="software_engineer",
        help="Memora persona to evaluate (e.g. software_engineer, academic_researcher).",
    )
    parser.add_argument(
        "--timeline",
        default="weekly",
        help="Memora timeline (weekly, monthly, quarterly).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit number of evaluation questions to run.",
    )
    args = parser.parse_args()

    if args.live and not args.allow_official_live:
        print("Refusing live Memora execution without --allow-official-live.")
        print("This task permits adapter smoke/dry-runs only, not official benchmark evaluation.")
        sys.exit(2)

    if not MEMORA_DIR.exists():
        print(f"Error: Memora repository not found at {MEMORA_DIR}")
        sys.exit(1)

    print("=" * 70)
    print("RUNNING MEMORA ADAPTER SMOKE/DRY-RUN")
    print("=" * 70)
    print("  Disclaimer: not an official Memora result and not Stage A/B evidence.")
    print(f"  Persona:  {args.persona}")
    print(f"  Timeline: {args.timeline}")
    print(f"  Limit:    {args.limit}")
    print(f"  Mode:     {'LIVE OFFICIAL-STYLE' if args.live else 'MOCK SMOKE'}")
    print()

    # Apply patching
    monkey_patch_memora(args.live)

    # Set up arguments for conversation_to_memory.py
    user_id = f"{args.persona}_{args.timeline}"
    conv_dir = MEMORA_DIR / "data" / args.timeline / args.persona / "conversations"

    # Step 1: Bulk process conversations to memory
    print("--- STEP 1: conversation_to_memory ---")
    sys.argv = [
        "conversation_to_memory.py",
        "--system", "retrace",
        "--user-id", user_id,
        "--conversation-directory", str(conv_dir),
        "--num-sessions", str(args.limit),
    ]

    import conversation_to_memory
    try:
        conversation_to_memory.main()
    except SystemExit as exc:
        if exc.code != 0:
            print(f"conversation_to_memory exited with code {exc.code}")
            sys.exit(exc.code)

    # Step 2: Answer questions using memories
    print("\n--- STEP 2: memory_to_answer ---")
    questions_file = MEMORA_DIR / "data" / args.timeline / args.persona / f"evaluation_questions_{args.persona}.json"
    dest_dir = Path("outputs/memora")
    dest_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "memory_to_answer.py",
        str(questions_file),
        "--system", "retrace",
        "--user-id", user_id,
        "--limit", str(args.limit),
    ]

    # Ensure OPENAI_API_KEY env variable is mock-set if not live to avoid validation crash
    if not args.live and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "mock-openai-key"

    import shutil
    import glob
    
    # Store initial files in output directory to find new ones
    memora_eval_results_dir = questions_file.parent / "eval_results" / "retrace"
    initial_files = set(glob.glob(str(memora_eval_results_dir / "*.json"))) if memora_eval_results_dir.exists() else set()

    import memory_to_answer
    try:
        memory_to_answer.main()
    except SystemExit as exc:
        if exc.code != 0:
            print(f"memory_to_answer exited with code {exc.code}")
            sys.exit(exc.code)

    # Copy newly created JSON files to outputs/memora
    copied_files = []
    if memora_eval_results_dir.exists():
        post_files = set(glob.glob(str(memora_eval_results_dir / "*.json")))
        new_files = post_files - initial_files
        for f in new_files:
            file_path = Path(f)
            dest_file = dest_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            copied_files.append(dest_file)

    print()
    print("=" * 70)
    print("MEMORA ADAPTER SMOKE/DRY-RUN COMPLETE")
    print("This output is not an official Memora benchmark result.")
    if copied_files:
        print("Adapter output JSON files copied to:")
        for f in copied_files:
            print(f"  - {f}")
    else:
        print(f"Check output under: {memora_eval_results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
