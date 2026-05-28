"""Legacy test_pipeline tests retired in Wave 2 closure.

All behaviors previously tested here are now covered by the canonical typed
test suites:

- Block / supersede / uncertain flows:
    tests/backend_contract/test_retrace_backend_typed_pipeline.py
- JSONL round-trip:
    test_pipeline_answer_record_is_jsonl_compatible in same file
- Query-conditioned answer:
    test_pipeline_answer_blocked_beliefs_are_query_conditioned in same file
"""
