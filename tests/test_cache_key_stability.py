from __future__ import annotations
from retracemem.cache.jsonl_cache import calculate_cache_key


def test_cache_key_stability_and_uniqueness() -> None:
    # Baseline parameters
    params = {
        "provider": "openai",
        "model_id": "gpt-4o",
        "model_revision_or_api_version": "2024-05-13",
        "prompt_template_hash": "p_hash",
        "response_schema_version": "s_v1",
        "parser_version": "par_v1",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100,
        "seed": 42,
        "input_hash": "inp_hash",
        "condition_context_hash": "cond_hash",
        "temporal_context_hash": "temp_hash",
    }

    # Generate baseline key
    key1 = calculate_cache_key(**params)
    key2 = calculate_cache_key(**params)

    # Identical parameters must yield identical key
    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # SHA256 length in hex

    # Modify each of the 13 parameters one by one and ensure key changes
    for key in params.keys():
        modified_params = params.copy()
        
        # Apply a modification depending on type
        val = modified_params[key]
        if isinstance(val, str):
            modified_params[key] = val + "_diff"
        elif isinstance(val, float):
            modified_params[key] = val + 1.0
        elif isinstance(val, int):
            modified_params[key] = val + 1
        elif val is None:
            modified_params[key] = "not_none"

        key_mod = calculate_cache_key(**modified_params)
        assert key_mod != key1, f"Modifying parameter '{key}' did not change cache key!"
