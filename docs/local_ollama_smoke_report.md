# Local Ollama Smoke Report

- Timestamp: 2026-06-26T16:35:47Z
- Machine note: MacBook Pro M5 Pro, 48GB unified memory
- Ollama base URL: http://localhost:11434
- Repository SHA: 5577919b9046d06dfd4459234bc228894b00dc81
- Pytest preflight: run separately; see console output
- Benchmark split sizes: {"challenge_test_hard": 300, "dev_calibration": 200, "main_test_synthetic": 1000}
- Methods run: direct_json, mempatch
- Models requested: deepseek-r1:32b, qwen3-coder:30b, qwen3-coder-next:latest
- Models available: qwen3-coder-next:latest, deepseek-r1:32b, qwen3-coder:30b
- Models skipped: none
- Cases per split: {"challenge_test_hard": 5, "main_test_synthetic": 5}
- Prediction count: 60
- Model failures: 0
- Overall parse failure rate: 0.100

## Aggregate Metrics

| model | method | split | n | schema_valid_rate | exact_state_map | contract_valid_state_success | decision_macro_f1 | evidence_f1 | diagnosis_accuracy | strict_joint | unsafe_reuse_rate | parse_failure_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1:32b | direct_json | main_test_synthetic | 5 | 0.400 | 0.400 | 0.400 | 0.167 | 0.233 | 0.200 | 0.000 | 0.000 | 0.600 |
| deepseek-r1:32b | direct_json | challenge_test_hard | 5 | 0.800 | 0.000 | 0.000 | 0.667 | 0.673 | 0.800 | 0.000 | 0.000 | 0.000 |
| deepseek-r1:32b | mempatch | main_test_synthetic | 5 | 1.000 | 0.200 | 0.200 | 0.500 | 0.233 | 0.600 | 0.000 | 0.000 | 0.600 |
| deepseek-r1:32b | mempatch | challenge_test_hard | 5 | 1.000 | 0.000 | 0.000 | 0.167 | 0.267 | 0.800 | 0.000 | 0.000 | 0.000 |
| qwen3-coder:30b | direct_json | main_test_synthetic | 5 | 1.000 | 0.400 | 0.400 | 0.333 | 0.707 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder:30b | direct_json | challenge_test_hard | 5 | 1.000 | 0.000 | 0.000 | 0.667 | 0.720 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder:30b | mempatch | main_test_synthetic | 5 | 1.000 | 0.200 | 0.200 | 0.500 | 0.307 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder:30b | mempatch | challenge_test_hard | 5 | 1.000 | 0.000 | 0.000 | 0.333 | 0.222 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder-next:latest | direct_json | main_test_synthetic | 5 | 1.000 | 0.400 | 0.400 | 0.667 | 0.733 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder-next:latest | direct_json | challenge_test_hard | 5 | 1.000 | 0.400 | 0.400 | 0.667 | 0.700 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder-next:latest | mempatch | main_test_synthetic | 5 | 1.000 | 0.200 | 0.200 | 0.333 | 0.133 | 0.200 | 0.000 | 0.000 | 0.000 |
| qwen3-coder-next:latest | mempatch | challenge_test_hard | 5 | 1.000 | 0.200 | 0.200 | 0.333 | 0.133 | 0.200 | 0.000 | 0.000 | 0.000 |

`decision_macro_f1` is the smoke-run class-balanced decision correctness available from deterministic score rows.

## Parse Failure Examples

```json
[
  {
    "scenario_id": "mp_syn_00000",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "main_test_synthetic",
    "parse_error": "JSONDecodeError: Expecting ',' delimiter: line 11 column 4 (char 378)",
    "response": {
      "answer": "",
      "decision": null,
      "memory_operation": null,
      "memory_state": {},
      "evidence_event_ids": [],
      "failure_diagnosis": null,
      "followup_answer": ""
    }
  },
  {
    "scenario_id": "mp_syn_00003",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "main_test_synthetic",
    "parse_error": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
    "response": {
      "answer": "",
      "decision": null,
      "memory_operation": null,
      "memory_state": {},
      "evidence_event_ids": [],
      "failure_diagnosis": null,
      "followup_answer": ""
    }
  },
  {
    "scenario_id": "mp_syn_00004",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "main_test_synthetic",
    "parse_error": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
    "response": {
      "answer": "",
      "decision": null,
      "memory_operation": null,
      "memory_state": {},
      "evidence_event_ids": [],
      "failure_diagnosis": null,
      "followup_answer": ""
    }
  },
  {
    "scenario_id": "mp_syn_00000",
    "model": "deepseek-r1:32b",
    "method": "mempatch",
    "split": "main_test_synthetic",
    "parse_error": "JSONDecodeError: Expecting ',' delimiter: line 11 column 4 (char 378)",
    "response": {
      "answer": "",
      "decision": "use_current_memory",
      "memory_operation": "REVISE",
      "memory_state": {
        "mp_syn_00000_m01": "outdated",
        "mp_syn_00000_m02": "current",
        "mp_syn_00000_m03": "current",
        "mp_syn_00000_m04": "current",
        "mp_syn_00000_m05": "current"
      },
      "evidence_event_ids": [
        "mp_syn_00000_e1"
      ],
      "failure_diagnosis": "stale_memory_reuse",
      "followup_answer": ""
    }
  },
  {
    "scenario_id": "mp_syn_00003",
    "model": "deepseek-r1:32b",
    "method": "mempatch",
    "split": "main_test_synthetic",
    "parse_error": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
    "response": {
      "answer": "",
      "decision": "mark_unresolved",
      "memory_operation": "MARK_UNRESOLVED",
      "memory_state": {
        "mp_syn_00003_m01": "current",
        "mp_syn_00003_m02": "unresolved",
        "mp_syn_00003_m03": "current",
        "mp_syn_00003_m04": "current",
        "mp_syn_00003_m05": "current",
        "mp_syn_00003_m06": "current",
        "mp_syn_00003_m07": "current",
        "mp_syn_00003_m08": "current"
      },
      "evidence_event_ids": [
        "mp_syn_00003_e1",
        "mp_syn_00003_e3"
      ],
      "failure_diagnosis": "conflict_collapse",
      "followup_answer": ""
    }
  }
]
```

## Parse-Successful Prediction Examples

```json
[
  {
    "scenario_id": "mp_syn_00001",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "main_test_synthetic",
    "parse_error": null,
    "response": {
      "answer": "The durable memory for CASE-10001 should remain preserved with its current state.",
      "decision": "use_current_memory",
      "memory_operation": "PRESERVE",
      "memory_state": {
        "mp_syn_00001_m01": "current",
        "mp_syn_00001_m02": "current",
        "mp_syn_00001_m03": "out_of_scope"
      },
      "evidence_event_ids": [
        "mp_syn_00001_e0",
        "mp_syn_00001_e1"
      ],
      "failure_diagnosis": "stale_memory_reuse",
      "followup_answer": "CASE-10001's support entitlement remains unchanged as per the current durable memory."
    }
  },
  {
    "scenario_id": "mp_syn_00002",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "main_test_synthetic",
    "parse_error": null,
    "response": {
      "answer": "The durable memory for CASE-10002 should be marked unresolved due to conflicting updates from different workspaces.",
      "decision": "escalate",
      "memory_operation": "MARK_UNRESOLVED",
      "memory_state": {
        "mp_syn_00002_m01": "unresolved",
        "mp_syn_00002_m02": "current",
        "mp_syn_00002_m03": "out_of_scope"
      },
      "evidence_event_ids": [
        "mp_syn_00002_e1"
      ],
      "failure_diagnosis": "scope_leakage",
      "followup_answer": "The room booking for CASE-10002 is currently unresolved and requires verification from the most authoritative source."
    }
  },
  {
    "scenario_id": "mp_hard_00000",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "challenge_test_hard",
    "parse_error": null,
    "response": {
      "answer": "The durable memory for CASE-10000 should be revised to reflect the newer verified record.",
      "decision": "use_current_memory",
      "memory_operation": "REVISE",
      "memory_state": {
        "mp_hard_00000_m01": "outdated",
        "mp_hard_00000_m02": "current",
        "mp_hard_00000_m03": "current"
      },
      "evidence_event_ids": [
        "mp_hard_00000_e1"
      ],
      "failure_diagnosis": "stale_memory_reuse",
      "followup_answer": "CASE-10000's release gate state has been updated to the newer verified record."
    }
  },
  {
    "scenario_id": "mp_hard_00001",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "challenge_test_hard",
    "parse_error": null,
    "response": {
      "answer": "Preserve the current memory for CASE-10001.",
      "decision": "use_current_memory",
      "memory_operation": "PRESERVE",
      "memory_state": {
        "mp_hard_00001_m01": "current",
        "mp_hard_00001_m02": "current",
        "mp_hard_00001_m03": "out_of_scope",
        "mp_hard_00001_m04": "out_of_scope",
        "mp_hard_00001_m05": "out_of_scope",
        "mp_hard_00001_m06": "out_of_scope",
        "mp_hard_00001_m07": "out_of_scope",
        "mp_hard_00001_m08": "out_of_scope"
      },
      "evidence_event_ids": [
        "mp_hard_00001_e0",
        "mp_hard_00001_e1",
        "mp_hard_00001_e3"
      ],
      "failure_diagnosis": "under_update",
      "followup_answer": "CASE-10001's support entitlement remains as previously recorded."
    }
  },
  {
    "scenario_id": "mp_hard_00002",
    "model": "deepseek-r1:32b",
    "method": "direct_json",
    "split": "challenge_test_hard",
    "parse_error": null,
    "response": {
      "answer": "Preserve the current memory for CASE-10002 as it remains authoritative within the main workspace.",
      "decision": "use_current_memory",
      "memory_operation": "PRESERVE",
      "memory_state": {
        "mp_hard_00002_m01": "current",
        "mp_hard_00002_m02": "current",
        "mp_hard_00002_m03": "background",
        "mp_hard_00002_m04": "background",
        "mp_hard_00002_m05": "background"
      },
      "evidence_event_ids": [
        "mp_hard_00002_e1",
        "mp_hard_00002_e3"
      ],
      "failure_diagnosis": "scope_leakage",
      "followup_answer": "The room booking state for CASE-10002 remains as per the current durable memory."
    }
  }
]
```

## Recommendation

Use this local run only for adapter, prompt, parsing, and scorer validation. Run full benchmark campaigns on a server after local parse/schema stability is acceptable.
