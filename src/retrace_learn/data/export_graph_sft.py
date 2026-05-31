"""Export ``graph_extraction_sft.jsonl`` (Module 1 SFT data).

raw multi-subagent dialogue -> structured memory graph. Each row is validated
against :class:`retrace_learn.schemas.GraphExtractionExample` before writing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.data.jsonl_io import write_jsonl

DEFAULT_OUT = "outputs/retrace_learn/graph_extraction_sft.jsonl"


def build_rows() -> list[dict]:
    rows = []
    for ep in build_synthetic_episodes():
        ex = ep.to_graph_extraction_example()
        ex.validate()
        rows.append(ex.to_dict())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = build_rows()
    n = write_jsonl(Path(args.out), rows)
    print(f"wrote {n} graph_extraction_sft rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
