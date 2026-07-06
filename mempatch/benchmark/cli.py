"""Command line entry point for the MemPatch benchmark kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mempatch.benchmark.generate import DEFAULT_QUOTAS, generate_raw_files, validate_generated_row
from mempatch.benchmark.leakage import audit_public_rows
from mempatch.benchmark.release import export_release, read_jsonl, write_jsonl
from mempatch.benchmark.score import aggregate_scores, score_row


def _split_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=PATH")
    split, path = value.split("=", 1)
    if not split:
        raise argparse.ArgumentTypeError("split name cannot be empty")
    return split, Path(path)


def _quota(value: str) -> tuple[str, int]:
    split, path = _split_path(value)
    try:
        count = int(str(path))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SPLIT=COUNT") from exc
    return split, count


def _load_config_quotas(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    import yaml

    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    splits = payload.get("splits") or {}
    quotas: dict[str, int] = {}
    for split, config in splits.items():
        if isinstance(config, dict) and "count" in config:
            quotas[str(split)] = int(config["count"])
    return quotas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="MemPatch")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-synthetic")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--config", type=Path, default=None)
    generate.add_argument("--quota", action="append", type=_quota, default=[])
    generate.add_argument("--seed-namespace", default="mempatch_final")

    export = sub.add_parser("export-release")
    export.add_argument("--input", action="append", type=_split_path, required=True, help="SPLIT=PATH")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--release-version", default="final")

    audit = sub.add_parser("audit-public")
    audit.add_argument("--public-jsonl", type=Path, required=True)

    score = sub.add_parser("score")
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--scores", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--group-by",
        action="append",
        choices=("split", "domain", "difficulty", "failure_mode", "pattern"),
        default=[],
    )

    args = parser.parse_args(argv)
    if args.command == "generate-synthetic":
        config_quotas = _load_config_quotas(args.config)
        quotas = dict(args.quota) if args.quota else (config_quotas or DEFAULT_QUOTAS)
        paths = generate_raw_files(args.output, quotas, seed_namespace=args.seed_namespace)
        validation_errors = []
        for path in paths.values():
            for row in read_jsonl(path):
                validation_errors.extend(validate_generated_row(row))
        payload = {
            "output": str(args.output),
            "splits": {split: str(path) for split, path in paths.items()},
            "validation_errors": validation_errors,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1 if validation_errors else 0
    if args.command == "export-release":
        manifest = export_release(dict(args.input), args.output, version=args.release_version)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == "audit-public":
        violations = audit_public_rows(read_jsonl(args.public_jsonl))
        print(json.dumps({"violations": violations}, indent=2, ensure_ascii=False))
        return 1 if violations else 0
    if args.command == "score":
        labels = {row["scenario_id"]: row for row in read_jsonl(args.labels)}
        score_rows = []
        for prediction in read_jsonl(args.predictions):
            scenario_id = prediction.get("scenario_id")
            if scenario_id not in labels:
                raise ValueError(f"prediction has no label row: {scenario_id}")
            score_rows.append(score_row(labels[scenario_id], prediction))
        write_jsonl(args.output, score_rows)
        return 0
    if args.command == "aggregate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(aggregate_scores(read_jsonl(args.scores), group_by=args.group_by), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
