from __future__ import annotations

import json
import re
import difflib
from enum import Enum
from dataclasses import dataclass
from typing import Any

class ParseErrorCode(str, Enum):
    JSON_DECODE_FAILED = "JSON_DECODE_FAILED"
    SCHEMA_CONSTRAINTS_VIOLATED = "SCHEMA_CONSTRAINTS_VIOLATED"
    GROUNDING_FAILED = "GROUNDING_FAILED"
    ACTION_VOCAB_VIOLATED = "ACTION_VOCAB_VIOLATED"

@dataclass(frozen=True)
class StructuredParseError(Exception):
    code: ParseErrorCode
    message: str
    context: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

@dataclass(frozen=True)
class NormalizedResult:
    value: str
    sensitivity_applied: bool
    sensitivity_type: str  # "exact" | "prefix" | "suffix" | "fuzzy" | "failed"

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their contents from LLM outputs, supporting unclosed tags."""
    # Remove closed think tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle unclosed think tags
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()

def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Match ```json ... ``` or just ``` ... ```
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def extract_json_array(text: str) -> list:
    """Extract and parse the first valid JSON array from the output, fail-closed."""
    cleaned = strip_think_tags(text)
    cleaned = strip_code_fences(cleaned)
    
    start_idx = cleaned.find("[")
    if start_idx == -1:
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            "No JSON array start '[' found in text.",
            {"text_preview": cleaned[:200]}
        )
        
    end_idx = cleaned.rfind("]")
    if end_idx == -1 or end_idx < start_idx:
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            "No JSON array end ']' found in text.",
            {"text_preview": cleaned[:200]}
        )
        
    array_str = cleaned[start_idx:end_idx + 1]
    try:
        data = json.loads(array_str)
        if isinstance(data, list):
            return data
    except Exception as e:
        # Retry by scanning
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
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            f"Could not parse valid JSON array: {e}",
            {"json_str_preview": array_str[:200]}
        )

def extract_json_object(text: str) -> dict:
    """Extract and parse a valid JSON object from the output, fail-closed."""
    cleaned = strip_think_tags(text)
    cleaned = strip_code_fences(cleaned)
    
    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            "No JSON object start '{' found in text.",
            {"text_preview": cleaned[:200]}
        )
        
    end_idx = cleaned.rfind("}")
    if end_idx == -1 or end_idx < start_idx:
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            "No JSON object end '}' found in text.",
            {"text_preview": cleaned[:200]}
        )
        
    obj_str = cleaned[start_idx:end_idx + 1]
    try:
        data = json.loads(obj_str)
        if isinstance(data, dict):
            return data
    except Exception as e:
        # Retry by scanning
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
        raise StructuredParseError(
            ParseErrorCode.JSON_DECODE_FAILED,
            f"Could not parse valid JSON object: {e}",
            {"json_str_preview": obj_str[:200]}
        )

def canonicalize_id(
    returned_id: str,
    valid_ids: set[str],
) -> NormalizedResult:
    """Canonicalize a returned ID (belief or condition) with tracking and ambiguity rejection."""
    if returned_id in valid_ids:
        return NormalizedResult(returned_id, False, "exact")
        
    # Attempt Prefix Match
    prefix_matches = [
        v_id for v_id in valid_ids
        if returned_id.startswith(v_id) or v_id.startswith(returned_id)
    ]
    if len(prefix_matches) == 1:
        return NormalizedResult(prefix_matches[0], True, "prefix")
    elif len(prefix_matches) > 1:
        return NormalizedResult(returned_id, False, "failed")
        
    # Attempt Suffix Match
    suffix_matches = [
        v_id for v_id in valid_ids
        if returned_id.endswith(v_id) or v_id.endswith(returned_id)
    ]
    if len(suffix_matches) == 1:
        return NormalizedResult(suffix_matches[0], True, "suffix")
    elif len(suffix_matches) > 1:
        return NormalizedResult(returned_id, False, "failed")
        
    # Attempt Fuzzy Match
    fuzzy_matches = difflib.get_close_matches(returned_id, list(valid_ids), n=2, cutoff=0.5)
    if len(fuzzy_matches) == 1:
        return NormalizedResult(fuzzy_matches[0], True, "fuzzy")
    elif len(fuzzy_matches) > 1:
        return NormalizedResult(returned_id, False, "failed")
        
    return NormalizedResult(returned_id, False, "failed")
