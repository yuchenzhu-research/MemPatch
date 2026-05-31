"""Stage SFT-1 eval: graph extraction quality.

Metrics: valid_json (graph well-formed), evidence_node_f1, belief_node_f1,
condition_node_f1, dependency_edge_f1. Node identity is the node id; dependency
edges are keyed by (belief_id, condition_id). The extractor is model-agnostic:
pass any object with ``.extract(raw_dialogue, roles) -> graph``.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from retrace_learn.schemas import GraphExtractionExample, SchemaValidationError, validate_memory_graph
from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.runtime.graph_extractor import RuleBasedGraphExtractor
from retrace_learn.eval.metrics import PRF, mean, set_prf


def _node_ids(nodes: list[dict[str, Any]], key: str) -> set[str]:
    return {n[key] for n in nodes}


def _dep_keys(edges: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(e["belief_id"], e["condition_id"]) for e in edges}


def evaluate_example(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    valid = True
    try:
        validate_memory_graph(pred)
    except SchemaValidationError:
        valid = False
    return {
        "valid_json": 1.0 if valid else 0.0,
        "evidence_node": set_prf(
            _node_ids(pred["evidence_nodes"], "evidence_id"),
            _node_ids(gold["evidence_nodes"], "evidence_id"),
        ),
        "belief_node": set_prf(
            _node_ids(pred["belief_nodes"], "belief_id"),
            _node_ids(gold["belief_nodes"], "belief_id"),
        ),
        "condition_node": set_prf(
            _node_ids(pred["condition_nodes"], "condition_id"),
            _node_ids(gold["condition_nodes"], "condition_id"),
        ),
        "dependency_edge": set_prf(
            _dep_keys(pred["dependency_edges"]), _dep_keys(gold["dependency_edges"])
        ),
    }


def evaluate(extractor, examples: list[GraphExtractionExample]) -> dict[str, float]:
    per: dict[str, list[float]] = {
        "valid_json": [],
        "evidence_node_f1": [],
        "belief_node_f1": [],
        "condition_node_f1": [],
        "dependency_edge_f1": [],
    }
    for ex in examples:
        pred = extractor.extract(ex.raw_dialogue, ex.subagent_roles)
        res = evaluate_example(pred, ex.output_graph)
        per["valid_json"].append(res["valid_json"])
        per["evidence_node_f1"].append(res["evidence_node"].f1)
        per["belief_node_f1"].append(res["belief_node"].f1)
        per["condition_node_f1"].append(res["condition_node"].f1)
        per["dependency_edge_f1"].append(res["dependency_edge"].f1)
    return {k: mean(v) for k, v in per.items()}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    examples = [ep.to_graph_extraction_example() for ep in build_synthetic_episodes()]
    report = evaluate(RuleBasedGraphExtractor(), examples)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
