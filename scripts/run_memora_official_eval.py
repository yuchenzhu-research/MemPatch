from __future__ import annotations

import sys
from pathlib import Path

# Add project root and Memora agent eval to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

memora_agent_eval_dir = project_root / "reference" / "Memora" / "evals" / "agent_eval"
sys.path.insert(0, str(memora_agent_eval_dir))

try:
    from base_evaluator import BaseMemorySystem, BaseEvaluator
    from memory_to_answer import fama_score
    MEMORA_IMPORT_OK = True
except Exception as e:
    print(f"Error importing Memora official evaluator components: {e}")
    MEMORA_IMPORT_OK = False


def main() -> None:
    if not MEMORA_IMPORT_OK:
        print("Could not import official Memora evaluation modules. Skipping verification.")
        sys.exit(1)

    print("Successfully imported Memora BaseMemorySystem, BaseEvaluator, and fama_score.")

    # 1. Verify fama_score correctness
    print("\n--- Verifying FAMA Score Logic ---")
    test_cases = [
        # (presence_correct, presence_total, absence_correct, absence_total, expected_score)
        (1, 1, 1, 1, 1.0),   # 100% presence, 100% absence
        (1, 1, 0, 1, 0.5),   # 100% presence, 0% absence. lambda = 0.5. FAMA = 1.0 - 0.5*(1-0) = 0.5
        (0, 1, 1, 1, 0.0),   # 0% presence, 100% absence. lambda = 0.5. FAMA = 0 - 0.5*(0) = 0.0
        (2, 2, 1, 2, 0.75),  # 100% presence, 50% absence. lambda = 2/4 = 0.5. FAMA = 1.0 - 0.5*(0.5) = 0.75
    ]

    all_passed = True
    for idx, (pc, pt, ac, at, expected) in enumerate(test_cases):
        score = fama_score(pc, pt, ac, at)
        passed = abs(score - expected) < 1e-6
        print(f"Case {idx}: presence={pc}/{pt}, absence={ac}/{at} => FAMA={score:.4f} (Expected={expected:.4f}) -> {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    # 2. Instantiate MockMemorySystem to verify BaseMemorySystem interface
    print("\n--- Verifying BaseMemorySystem Subclass Interface ---")
    
    class ReTraceMemorySystem(BaseMemorySystem):
        def get_system_name(self) -> str:
            return "retrace_mock"
        
        def initialize_client(self) -> bool:
            return True
            
        def add_conversation_to_memory(self, conversation_data: dict) -> dict:
            return {"status": "success"}
            
        def search_memories(self, query: str, limit: int = 50, session_date = None, date_range = None) -> list:
            return [{"memory": "Mocked memory support", "score": 1.0}]
            
        def get_required_env_vars(self) -> list:
            return []

    try:
        sys_instance = ReTraceMemorySystem(user_id="test_user_id")
        evaluator = BaseEvaluator(sys_instance)
        print("Successfully instantiated ReTraceMemorySystem and BaseEvaluator.")
        print(f"System name: {sys_instance.get_system_name()}")
        print(f"Config info: {sys_instance.get_config_info()}")
    except Exception as e:
        print(f"Failed to instantiate or verify evaluator interface: {e}")
        all_passed = False

    if all_passed:
        print("\nAll minimal scoring fixture verifications PASSED.")
        sys.exit(0)
    else:
        print("\nSome verifications FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
