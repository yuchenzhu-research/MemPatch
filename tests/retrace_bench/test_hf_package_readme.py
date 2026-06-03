"""Guard the v1.0 HF packaging template against weakened split semantics.

The Hugging Face card must advertise only the public split names
``main`` / ``hard`` / ``realistic`` / ``calibration`` (never train / dev /
validation / test), keep the calibration smoke-only warning, keep the realistic
annotation-pending statement, and link the repo + official evaluator.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SCRIPT = REPO_ROOT / "scripts" / "package_hf_retrace_bench.py"
GENERATED_README = REPO_ROOT / "release" / "huggingface" / "ReTrace-Bench" / "README.md"

PUBLIC_SPLITS = ("main", "hard", "realistic", "calibration")
FORBIDDEN_PUBLIC_SPLITS = ("train", "dev", "validation", "test")
CALIBRATION_WARNING = (
    "it is not a model-selection / checkpoint-selection validation set "
    "and must not be used to tune or select systems"
)
REALISTIC_PENDING = "annotation_status = pending"


def test_package_script_declares_only_public_splits():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    # The config block is built from the SPLITS table at runtime, so assert the
    # table advertises the public names and the script never hard-codes a
    # forbidden public split label.
    for split in PUBLIC_SPLITS:
        assert f'"{split}"' in text, f"SPLITS table missing public split {split}"
    for bad in FORBIDDEN_PUBLIC_SPLITS:
        assert f"split: {bad}" not in text, f"forbidden public split name: {bad}"


def test_package_script_template_has_calibration_and_realistic_semantics():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert CALIBRATION_WARNING in text
    assert REALISTIC_PENDING in text


def test_package_script_template_links_repo_and_evaluator():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert "github.com/yuchenzhu-research/ReTrace" in text
    assert "evaluate_retrace_bench_predictions.py" in text
    assert "decision_macro_f1" in text


def test_package_script_does_not_reference_legacy_splits():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    for legacy in ("test_800_templateheldout_en", "sample_80_hard_en", "train_3000_en", "dev_400_en"):
        assert legacy not in text, f"package script still references legacy split {legacy}"


def test_generated_readme_is_consistent_when_present():
    if not GENERATED_README.exists():
        return  # generated artifact may not be present in a fresh checkout
    text = GENERATED_README.read_text(encoding="utf-8")
    assert CALIBRATION_WARNING in text
    assert "github.com/yuchenzhu-research/ReTrace" in text
    for split in PUBLIC_SPLITS:
        assert f"split: {split}" in text
    for bad in FORBIDDEN_PUBLIC_SPLITS:
        assert f"split: {bad}" not in text
