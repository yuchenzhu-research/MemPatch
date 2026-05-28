#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retracemem.adapters.memora_oracle_diagnostic import analyze_stage_a_failure_modes


def render_markdown(analysis: dict[str, Any], report_path: Path) -> str:
    aggregate = analysis["aggregate"]
    lines = [
        "# Memora Oracle Diagnostic Stage A Failure Analysis",
        "",
        "Development-only provenance analysis. No provider calls were made.",
        "",
        f"- **Source report**: `{report_path}`",
        f"- **Questions executed**: {analysis.get('source_questions_executed')}",
        f"- **Source errors**: {analysis.get('source_errors')}",
        "",
        "## Aggregate provenance pattern",
        "",
    ]
    for key in sorted(aggregate):
        lines.append(f"- **{key}**: {aggregate[key]}")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- This file summarizes recorded provenance only.",
        "- It does not automatically infer semantic failure classes from private text.",
        "- Human semantic labels may be attached separately using the JSON schema field `manual_annotation_schema`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Memora oracle diagnostic Stage A failure analyzer.",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = analyze_stage_a_failure_modes(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = report_path.parent.name or report_path.stem
    json_path = out_dir / f"{stem}_stage_a_failure_analysis.json"
    md_path = out_dir / f"{stem}_stage_a_failure_analysis.md"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(analysis, report_path), encoding="utf-8")
    print(f"[DONE] analysis_json={json_path} analysis_md={md_path}", flush=True)


if __name__ == "__main__":
    main()
