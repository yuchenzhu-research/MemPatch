"""Inject persistent statistical unit fields into scenarios datasets."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mempatch.audit.rebuild_gold import extract_case_ref_and_topic

def get_variant_id_from_answer(expected_answer: str, case_ref: str, topic: str) -> str:
    # Look up matching template to find variant_id
    from mempatch.audit.rebuild_gold import VARIANT_ANSWER_TEMPLATES
    for var_id, template in VARIANT_ANSWER_TEMPLATES.items():
        rendered = template.format(case_ref=case_ref, topic=topic)
        if re.sub(r"\s+", " ", rendered.strip().lower()) == re.sub(r"\s+", " ", expected_answer.strip().lower()):
            return var_id
    return "unknown_variant"

def inject_fields(scenario_path: Path) -> None:
    if not scenario_path.exists():
        print(f"Warning: file {scenario_path} not found.")
        return
        
    rows = []
    with scenario_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                
    modified_rows = []
    template_families = set()
    for s in rows:
        public_input = s.get("public_input", {})
        gold = s.get("hidden_gold", {})
        
        case_ref, topic = extract_case_ref_and_topic(public_input)
        variant_id = get_variant_id_from_answer(gold.get("expected_answer", ""), case_ref, topic)
        pattern = s.get("pattern", "unknown_pattern")
        
        # Determine persistent fields
        tf_id = f"tf_{pattern}_{variant_id}"
        ti_id = s.get("scenario_id", "unknown_instance")
        fm = gold.get("expected_failure_diagnosis", s.get("primary_failure_mode", "unknown_failure"))
        dom = s.get("domain", "unknown_domain")
        gen_ver = "v1.3"
        
        # Inject to top level of the scenario record
        s["template_family_id"] = tf_id
        s["template_instance_id"] = ti_id
        s["failure_mode"] = fm
        s["domain"] = dom
        s["generator_version"] = gen_ver
        
        template_families.add(tf_id)
        modified_rows.append(s)
        
    # Write back in-place
    with scenario_path.open("w", encoding="utf-8") as f:
        for r in modified_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Successfully injected statistical fields into {scenario_path.name}:")
    print(f"  - Total records: {len(modified_rows)}")
    print(f"  - Unique Template Families detected: {len(template_families)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Inject persistent statistical unit fields.")
    parser.add_argument("--test-input", default="local/data/mempatch/test/scenarios.jsonl")
    parser.add_argument("--train-input", default="local/data/mempatch/train/scenarios.jsonl")
    args = parser.parse_args()
    
    inject_fields(Path(args.test_input))
    inject_fields(Path(args.train_input))

if __name__ == "__main__":
    main()
