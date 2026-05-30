from __future__ import annotations

import json
import re
import difflib
from typing import Any

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their contents from LLM outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (like ```json ... ```) from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def extract_first_json_array(text: str) -> list:
    """Extract and parse the first valid JSON array from the output."""
    cleaned = strip_think_tags(text)
    cleaned = strip_code_fences(cleaned)
    # Find the start of the array
    start_idx = cleaned.find("[")
    if start_idx == -1:
        raise ValueError("No JSON array start '[' found in text.")
    # Attempt to locate the matched closing bracket or try parsing substrings
    # A simple but robust way: try parsing from start_idx to the end, trimming character by character if needed,
    # or finding the last ']'
    end_idx = cleaned.rfind("]")
    if end_idx == -1 or end_idx < start_idx:
        raise ValueError("No JSON array end ']' found in text.")
    
    array_str = cleaned[start_idx:end_idx + 1]
    try:
        data = json.loads(array_str)
        if isinstance(data, list):
            return data
    except Exception as e:
        pass

    # Fallback to scanning for first valid array
    for i in range(len(cleaned)):
        if cleaned[i] == "[":
            for j in range(len(cleaned), i, -1):
                if cleaned[j-1] == "]":
                    try:
                        data = json.loads(cleaned[i:j])
                        if isinstance(data, list):
                            return data
                    except Exception:
                        pass
    raise ValueError("Could not parse a valid JSON array from LLM response.")

def extract_json_object(text: str) -> dict:
    """Extract and parse a valid JSON object from the output."""
    cleaned = strip_think_tags(text)
    cleaned = strip_code_fences(cleaned)
    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object start '{' found in text.")
    end_idx = cleaned.rfind("}")
    if end_idx == -1 or end_idx < start_idx:
        raise ValueError("No JSON object end '}' found in text.")
    
    obj_str = cleaned[start_idx:end_idx + 1]
    try:
        data = json.loads(obj_str)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback scanning
    for i in range(len(cleaned)):
        if cleaned[i] == "{":
            for j in range(len(cleaned), i, -1):
                if cleaned[j-1] == "}":
                    try:
                        data = json.loads(cleaned[i:j])
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        pass
    raise ValueError("Could not parse a valid JSON object from LLM response.")

def canonicalize_belief_id_with_type(
    returned_id: str,
    valid_belief_ids: set[str],
) -> tuple[str, bool, str]:
    """Canonicalize a returned belief ID against valid belief IDs with ambiguity rejection.
    
    Returns:
        (canonical_id, applied, match_type)
        where match_type is one of: "exact", "prefix", "suffix", "fuzzy", "failed"
    """
    if returned_id in valid_belief_ids:
        return returned_id, False, "exact"
        
    # Attempt Prefix Match
    prefix_matches = [
        v_id for v_id in valid_belief_ids
        if returned_id.startswith(v_id) or v_id.startswith(returned_id)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0], True, "prefix"
    elif len(prefix_matches) > 1:
        # Ambiguity rejection
        return returned_id, False, "failed"
        
    # Attempt Suffix Match
    suffix_matches = [
        v_id for v_id in valid_belief_ids
        if returned_id.endswith(v_id) or v_id.endswith(returned_id)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0], True, "suffix"
    elif len(suffix_matches) > 1:
        # Ambiguity rejection
        return returned_id, False, "failed"
        
    # Attempt Fuzzy Match (using difflib)
    # Query top 2 matches to inspect ambiguity
    fuzzy_matches = difflib.get_close_matches(returned_id, list(valid_belief_ids), n=2, cutoff=0.5)
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], True, "fuzzy"
    elif len(fuzzy_matches) > 1:
        # Ambiguity rejection (multiple matches found with similarity >= 0.5)
        return returned_id, False, "failed"
        
    return returned_id, False, "failed"
