import json
import os
import subprocess
import sys

import pytest


def test_sft_exporters_execution(tmp_path):
    # Setup temporary paths for output files
    graph_sft_out = tmp_path / "graph_sft.jsonl"
    revision_sft_out = tmp_path / "revision_sft.jsonl"

    # Use python executable to run the exporter scripts
    python_bin = sys.executable

    # Run graph extractor exporter
    res_graph = subprocess.run(
        [
            python_bin,
            "scripts/export_graph_extractor_sft.py",
            "--in-file",
            "outputs/raw_dialogue_synth.jsonl",
            "--out-file",
            str(graph_sft_out),
        ],
        capture_output=True,
        text=True,
    )
    assert res_graph.returncode == 0, f"Graph exporter failed: {res_graph.stderr}"

    # Run revision proposer exporter
    res_revision = subprocess.run(
        [
            python_bin,
            "scripts/export_revision_proposer_sft.py",
            "--in-file",
            "outputs/raw_dialogue_synth.jsonl",
            "--out-file",
            str(revision_sft_out),
        ],
        capture_output=True,
        text=True,
    )
    assert res_revision.returncode == 0, f"Revision exporter failed: {res_revision.stderr}"

    # Validate output file existence
    assert os.path.exists(graph_sft_out)
    assert os.path.exists(revision_sft_out)

    # Validate graph SFT contents
    with open(graph_sft_out, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            msgs = data["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

            # Assistant content must be a valid JSON representation of graph
            graph_data = json.loads(msgs[2]["content"])
            assert "evidence_nodes" in graph_data
            assert "belief_nodes" in graph_data

    # Validate revision SFT contents
    with open(revision_sft_out, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            msgs = data["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

            # Assistant content must be a valid JSON array of actions
            actions = json.loads(msgs[2]["content"])
            assert isinstance(actions, list)
            for a in actions:
                assert "action_type" in a
