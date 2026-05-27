from __future__ import annotations
import json
import dataclasses
from retracemem.schemas import RunManifest


def test_run_manifest_serialization() -> None:
    manifest = RunManifest(
        run_manifest_id="manifest_12345",
        method_name="retrace_pipeline",
        model_config_id="config-gpt-4o",
        timestamp="2026-05-27T10:00:00Z",
        upstream_commit="abcde12345",
        output_path="/Users/yuchenzhu/Desktop/ReTrace/outputs/test.jsonl",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        metadata={"run_type": "smoke_test", "limit": 10},
    )

    # Convert to dict
    manifest_dict = dataclasses.asdict(manifest)
    assert manifest_dict["run_manifest_id"] == "manifest_12345"
    assert manifest_dict["method_name"] == "retrace_pipeline"
    assert manifest_dict["output_path"] == "/Users/yuchenzhu/Desktop/ReTrace/outputs/test.jsonl"
    assert manifest_dict["metadata"]["run_type"] == "smoke_test"

    # Serialize to JSON
    json_str = json.dumps(manifest_dict)
    
    # Deserialize back
    loaded_dict = json.loads(json_str)
    
    # Reconstruct
    fields = {f.name for f in dataclasses.fields(RunManifest)}
    kwargs = {k: v for k, v in loaded_dict.items() if k in fields}
    restored = RunManifest(**kwargs)

    assert restored.run_manifest_id == manifest.run_manifest_id
    assert restored.method_name == manifest.method_name
    assert restored.model_config_id == manifest.model_config_id
    assert restored.timestamp == manifest.timestamp
    assert restored.upstream_commit == manifest.upstream_commit
    assert restored.output_path == manifest.output_path
    assert restored.checksum == manifest.checksum
    assert restored.metadata == manifest.metadata
