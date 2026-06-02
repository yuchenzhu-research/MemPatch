"""Guard the HF packaging template against weakened split semantics."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SCRIPT = REPO_ROOT / "scripts" / "package_hf_retrace_bench.py"
GENERATED_README = REPO_ROOT / "release" / "huggingface" / "ReTrace-Bench" / "README.md"

STRONG_VALIDATION_WARNING = (
    "it is **not** a model-selection / checkpoint-selection validation set "
    "and must not be used to tune or select systems"
)


def test_package_script_template_has_strong_validation_warning():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert STRONG_VALIDATION_WARNING in text
    assert "for viewer compatibility only" in text


def test_package_script_template_links_repo_and_evaluator():
    text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert "github.com/yuchenzhu-research/ReTrace" in text
    assert "evaluate_retrace_bench_predictions.py" in text
    assert "decision_macro_f1" in text


def test_generated_readme_has_strong_validation_warning():
    if not GENERATED_README.exists():
        return  # generated artifact may not be present in a fresh checkout
    text = GENERATED_README.read_text(encoding="utf-8")
    assert STRONG_VALIDATION_WARNING in text
    assert "github.com/yuchenzhu-research/ReTrace" in text
