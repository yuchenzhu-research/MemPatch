import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_sft_exporters_execution(tmp_path):
    # Self-contained: generate the synthetic dataset into the temp dir first, so
    # the exporters do not depend on any pre-existing outputs/ artifact.
    python_bin = sys.executable
    synth_in = tmp_path / "raw_dialogue_synth.jsonl"
    graph_sft_out = tmp_path / "graph_sft.jsonl"
    revision_sft_out = tmp_path / "revision_sft.jsonl"

    res_synth = subprocess.run(
        [
            python_bin,
            "scripts/build_raw_dialogue_synth.py",
            "--out",
            str(synth_in),
            "--n",
            "26",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res_synth.returncode == 0, f"Synth builder failed: {res_synth.stderr}"

    # Run graph extractor exporter
    res_graph = subprocess.run(
        [
            python_bin,
            "scripts/export_graph_extractor_sft.py",
            "--in-file",
            str(synth_in),
            "--out-file",
            str(graph_sft_out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res_graph.returncode == 0, f"Graph exporter failed: {res_graph.stderr}"

    # Run revision proposer exporter
    res_revision = subprocess.run(
        [
            python_bin,
            "scripts/export_revision_proposer_sft.py",
            "--in-file",
            str(synth_in),
            "--out-file",
            str(revision_sft_out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
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
