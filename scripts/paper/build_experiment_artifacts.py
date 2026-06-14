#!/usr/bin/env python3
"""Build smoke and formal CSV/LaTeX/PDF artifacts from evaluator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SMOKE_MODELS = (
    ("qwen3_14b", "Qwen3-14B"),
    ("gemma3_12b", "Gemma-3-12B"),
    ("phi4", "Phi-4"),
)
FORMAL_MODELS = (
    ("qwen3_14b", "Qwen3-14B"),
    ("gemma3_12b", "Gemma-3-12B"),
    ("phi4", "Phi-4"),
    ("mistral_nemo_12b", "Mistral-Nemo-12B"),
)
SMOKE_SYSTEMS = (
    ("frozen_direct_prompting", "Frozen Direct Prompting"),
    ("full_context", "Full Context"),
    ("mempatch_zero_shot", "MemPatch Zero-Shot"),
)
FROZEN_SYSTEMS = (
    ("structured_direct", "Frozen Direct Prompting"),
    ("full_context", "Full Context"),
    ("vanilla_rag", "Lexical RAG"),
    ("time_aware_rag", "Time-Aware RAG"),
    ("summary_memory", "Summary Memory"),
)
METRICS = (
    ("decision_macro_f1", "Decision $F_1$"),
    ("memory_state_accuracy", "MemState"),
    ("evidence_f1", "Evidence $F_1$"),
    ("failure_diagnosis_accuracy", "Diagnosis"),
    ("joint_revision_success", "Joint"),
    ("stale_reuse_rate", "Stale reuse"),
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def headline(path: Path) -> dict[str, float]:
    payload = load(path)
    return payload.get("headline_metrics") or payload


def pct(metrics: dict[str, float], key: str) -> float:
    return 100.0 * float(metrics[key])


def tex_table(lines: list[str], caption: str, label: str) -> str:
    return "\n".join(
        [r"\begin{table*}[t]", r"\centering", r"\small", *lines, f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table*}", ""]
    )


def build_smoke(bundle_root: Path, out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for system_id, display in SMOKE_SYSTEMS:
        model_metrics = []
        for model_id, _ in SMOKE_MODELS:
            metrics = headline(bundle_root / model_id / f"{system_id}_metrics.json")
            model_metrics.append(metrics)
        rows.append(
            {
                "system": display,
                "joints": [pct(metrics, "joint_revision_success") for metrics in model_metrics],
                "avg_joint": sum(pct(metrics, "joint_revision_success") for metrics in model_metrics) / 3,
                "avg_memstate": sum(pct(metrics, "memory_state_accuracy") for metrics in model_metrics) / 3,
            }
        )

    csv_path = out_root / "artifacts/smoke/aggregate_smoke_no_lora.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "qwen3_14b_joint", "gemma3_12b_joint", "phi4_joint", "avg_joint", "avg_memstate"])
        for row in rows:
            writer.writerow([row["system"], *[f"{v:.4f}" for v in row["joints"]], f"{row['avg_joint']:.4f}", f"{row['avg_memstate']:.4f}"])

    body = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Qwen3-14B Joint & Gemma-3-12B Joint & Phi-4 Joint & Avg. Joint & Avg. MemState \\",
        r"\midrule",
    ]
    for row in rows:
        values = [*row["joints"], row["avg_joint"], row["avg_memstate"]]
        body.append(f"{row['system']} & " + " & ".join(f"{value:.1f}" for value in values) + r" \\")
    body.extend((r"\bottomrule", r"\end{tabular}"))
    table_path = out_root / "paper/tables/table_smoke_no_lora.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        tex_table(
            body,
            "This is a no-LoRA 30-case smoke test for pipeline validation. It is not used as the main paper comparison.",
            "tab:smoke-no-lora",
        ),
        encoding="utf-8",
    )


def formal_metric_path(results_root: Path, slug: str, system_id: str) -> Path:
    if system_id == "final_state_control":
        return results_root / slug / "test500_final_state_control_lora_best_metrics.json"
    if system_id == "mempatch":
        return results_root / slug / "test500_mempatch_lora_best_metrics.json"
    return results_root / slug / f"baseline_{system_id}_metrics.json"


def build_frozen_table(results_root: Path, out_root: Path) -> None:
    body = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Frozen system & Qwen & Gemma & Phi-4 & Mistral & Avg. Joint & Avg. MemState \\",
        r"\midrule",
    ]
    for system_id, display in FROZEN_SYSTEMS:
        metrics = [headline(formal_metric_path(results_root, slug, system_id)) for slug, _ in FORMAL_MODELS]
        joints = [pct(row, "joint_revision_success") for row in metrics]
        avg_joint = sum(joints) / len(joints)
        avg_state = sum(pct(row, "memory_state_accuracy") for row in metrics) / len(metrics)
        values = [*joints, avg_joint, avg_state]
        body.append(f"{display} & " + " & ".join(f"{value:.1f}" for value in values) + r" \\")
    body.extend((r"\bottomrule", r"\end{tabular}"))
    path = out_root / "paper/tables/table_frozen_external.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tex_table(
            body,
            "Frozen external baselines on test500. All rows use frozen base backbones without benchmark-specific adaptation.",
            "tab:frozen-external",
        ),
        encoding="utf-8",
    )


def selected_checkpoint(results_root: Path, slug: str) -> tuple[int, float, str]:
    payload = load(results_root / slug / "checkpoint_selection.json")
    return int(payload["best_step"]), float(payload["best_val_loss"]), str(payload["checkpoint_dir"])


def assert_shared_checkpoint(results_root: Path, slug: str) -> str:
    selected = selected_checkpoint(results_root, slug)[2]
    direct_meta = load(results_root / slug / "test500_final_state_control_lora_best_manifest.json").get("run_meta") or {}
    patch_meta = load(results_root / slug / "test500_mempatch_lora_best_manifest.json").get("run_meta") or {}
    if direct_meta.get("adapter_path") != selected or patch_meta.get("adapter_path") != selected:
        raise ValueError(f"{slug}: Final-State Control and MemPatch do not share selected checkpoint {selected}")
    return selected


def build_full_metrics(results_root: Path, out_root: Path) -> None:
    body = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Backbone & System & Decision $F_1$ & MemState & Evidence $F_1$ & Diagnosis & Joint & Stale reuse & Validity \\",
        r"\midrule",
    ]
    for model_index, (slug, display) in enumerate(FORMAL_MODELS):
        assert_shared_checkpoint(results_root, slug)
        for system_id, system_display in (("final_state_control", "Final-State Control"), ("mempatch", "MemPatch")):
            metrics = headline(formal_metric_path(results_root, slug, system_id))
            manifest_name = "test500_final_state_control_lora_best_manifest.json" if system_id == "final_state_control" else "test500_mempatch_lora_best_manifest.json"
            meta = load(results_root / slug / manifest_name).get("run_meta") or {}
            validity = (
                100.0 * float(meta.get("raw_response_schema_compliance_rate", metrics.get("response_schema_compliance_rate", 0.0)))
                if system_id == "final_state_control"
                else 100.0 * float(meta.get("action_parse_valid_rate", 0.0))
            )
            values = [pct(metrics, key) for key, _ in METRICS]
            prefix = display if system_id == "final_state_control" else ""
            body.append(f"{prefix} & {system_display} & " + " & ".join(f"{value:.1f}" for value in values) + f" & {validity:.1f} " + r"\\")
        if model_index != len(FORMAL_MODELS) - 1:
            body.append(r"\addlinespace")
    body.extend((r"\bottomrule", r"\end{tabular}"))
    path = out_root / "supplement/tables/table_full_metrics.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tex_table(
            body,
            "Full test500 metrics for the matched adapted systems. Validity is raw five-field schema validity for Final-State Control and typed-action parse validity for MemPatch; invalid outputs are scored without silent repair.",
            "tab:full-metrics",
        ),
        encoding="utf-8",
    )


def load_curve(results_root: Path, log_root: Path, slug: str) -> list[tuple[int, float]]:
    checkpoint_dir = Path(selected_checkpoint(results_root, slug)[2])
    run_name = checkpoint_dir.parent.name
    payload = load(log_root / f"{slug}_split0" / run_name / "trainer_metrics.json")
    return [(int(row["step"]), float(row["val_loss"])) for row in payload.get("checkpoints") or []]


def build_checkpoint_artifacts(results_root: Path, log_root: Path, out_root: Path) -> None:
    body = [r"\begin{tabular}{lrr}", r"\toprule", r"Backbone & Selected step & Validation loss \\", r"\midrule"]
    curves = {}
    for slug, display in FORMAL_MODELS:
        step, loss, _ = selected_checkpoint(results_root, slug)
        body.append(f"{display} & {step} & {loss:.4f} " + r"\\")
        curves[display] = load_curve(results_root, log_root, slug)
    body.extend((r"\bottomrule", r"\end{tabular}"))
    path = out_root / "supplement/tables/table_checkpoint_selection.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tex_table(
            body,
            "Checkpoint selection uses minimum validation loss only. Test500 is not used for checkpoint selection.",
            "tab:checkpoint-selection",
        ),
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for display, rows in curves.items():
        steps = [row[0] for row in rows]
        losses = [row[1] for row in rows]
        ax.plot(steps, losses, marker="o", linewidth=1.5, label=display)
        best = min(range(len(rows)), key=lambda index: losses[index])
        ax.scatter(steps[best], losses[best], marker="*", s=85, zorder=3)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig_path = out_root / "supplement/figures/fig_validation_loss_curves.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def build_adapted_figure(results_root: Path, out_root: Path) -> None:
    import matplotlib.pyplot as plt

    labels, direct, patch = [], [], []
    for slug, display in FORMAL_MODELS:
        assert_shared_checkpoint(results_root, slug)
        labels.append(display)
        direct.append(pct(headline(formal_metric_path(results_root, slug, "final_state_control")), "joint_revision_success"))
        patch.append(pct(headline(formal_metric_path(results_root, slug, "mempatch")), "joint_revision_success"))
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for index, (left, right) in enumerate(zip(direct, patch)):
        ax.plot([left, right], [index, index], color="0.65", linewidth=1.5)
        ax.text(max(left, right) + 1.0, index, f"{right-left:+.1f}", va="center", fontsize=8)
    ax.scatter(direct, y, marker="o", s=34, label="Final-State Control", zorder=3)
    ax.scatter(patch, y, marker="s", s=34, label="MemPatch", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Joint revision success (%)")
    ax.set_xlim(0, max(direct + patch) * 1.28 if direct + patch else 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = out_root / "paper/figures/fig_adapted_pair_joint.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_formal(results_root: Path, log_root: Path, out_root: Path) -> None:
    build_frozen_table(results_root, out_root)
    build_full_metrics(results_root, out_root)
    build_adapted_figure(results_root, out_root)
    build_checkpoint_artifacts(results_root, log_root, out_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--bundle-root", type=Path, required=True)
    smoke.add_argument("--out-root", type=Path, required=True)
    formal = subparsers.add_parser("formal")
    formal.add_argument("--results-root", type=Path, required=True)
    formal.add_argument("--log-root", type=Path, required=True)
    formal.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "smoke":
            build_smoke(args.bundle_root, args.out_root)
        else:
            build_formal(args.results_root, args.log_root, args.out_root)
    except (FileNotFoundError, KeyError, TypeError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
