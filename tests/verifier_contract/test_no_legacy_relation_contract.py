from __future__ import annotations

from pathlib import Path


CANONICAL_TYPED_FILES = (
    Path("src/retracemem/verifier/contracts.py"),
    Path("src/retracemem/verifier/requirement_inducer.py"),
    Path("src/retracemem/verifier/evidence_edge_verifier.py"),
)


def test_new_typed_verifier_files_do_not_import_or_return_legacy_relation_schema() -> None:
    for path in CANONICAL_TYPED_FILES:
        text = path.read_text(encoding="utf-8")
        assert "RelationPrediction" not in text
        assert "RelationType" not in text

