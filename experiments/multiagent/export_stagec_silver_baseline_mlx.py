#!/usr/bin/env python3
"""
Build a larger Stage C silver SFT dataset for MLX-LM QLoRA training.

Purpose:
- Train an exploratory Stage C adapter tonight without using STALE external-test records.
- Use the repository's executable synthetic episode generator.
- Include only submissions with explicit evaluator-authored typed revision targets.
- Expand training-only templates with controlled surface variants.
- Keep v5 templates out of training as held-out validation/test families.

Scientific status:
    silver_synthetic_training_only
    not_for_paper_main_results = true
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.multiagent.contracts import StageCTrainingExample, TypedRevisionTarget
from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.export_stagec_sft import (
    SYSTEM_PROMPT,
    format_assistant_response,
    format_user_prompt,
)


OUT_DIR = Path("outputs/local_training/stagec_qwen3_4b_silver/data")
MANIFEST_PATH = Path("outputs/local_training/stagec_qwen3_4b_silver/manifest.json")
RNG_SEED = 42

# Controlled surface transformations for training rows only.
# They preserve the typed revision relation while changing nouns/environment wording.
TRAIN_AUGMENTATIONS = [
    ("base", {}),
    ("surface_qa", {
        "Staging": "QA",
        "staging": "QA",
        "Production": "Live",
        "production": "live",
        "Module": "Service",
        "module": "service",
        "Project": "Workspace",
        "project": "workspace",
    }),
    ("surface_preprod", {
        "Staging": "Pre-production",
        "staging": "pre-production",
        "Production": "Production",
        "production": "production",
        "Module": "Component",
        "module": "component",
        "Experiment": "Evaluation run",
        "experiment": "evaluation run",
    }),
    ("surface_canary", {
        "Staging": "Canary",
        "staging": "canary",
        "Production": "Primary",
        "production": "primary",
        "build": "release",
        "Build": "Release",
        "study": "analysis",
        "Study": "Analysis",
    }),
    ("surface_sandbox", {
        "Staging": "Sandbox",
        "staging": "sandbox",
        "Production": "Serving",
        "production": "serving",
        "pipeline": "workflow",
        "Pipeline": "Workflow",
        "dataset": "corpus",
        "Dataset": "Corpus",
    }),
    ("surface_validation", {
        "Staging": "Validation",
        "staging": "validation",
        "Production": "Deployed",
        "production": "deployed",
        "server": "endpoint",
        "Server": "Endpoint",
        "parser": "processor",
        "Parser": "Processor",
    }),
    ("surface_testbed", {
        "Staging": "Testbed",
        "staging": "testbed",
        "Production": "Customer-facing",
        "production": "customer-facing",
        "database": "storage",
        "Database": "Storage",
        "report": "record",
        "Report": "Record",
    }),
    ("surface_shadow", {
        "Staging": "Shadow",
        "staging": "shadow",
        "Production": "Mainline",
        "production": "mainline",
        "configuration": "setting",
        "Configuration": "Setting",
        "release": "snapshot",
        "Release": "Snapshot",
    }),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_replacements(text: str, mapping: dict[str, str]) -> str:
    out = text
    # Longer phrases first protects against partial collisions.
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(old, new)
    return out


def make_example(ep: Any, sub: Any, targets: tuple[TypedRevisionTarget, ...]) -> StageCTrainingExample:
    return StageCTrainingExample(
        example_id=f"silver_{ep.episode_id}_{sub.submission_id}",
        episode_id=ep.episode_id,
        submission_id=sub.submission_id,
        method_visible_input=sub,
        targets=targets,
        split="silver_synthetic",
        domain=ep.domain,
        failure_type=ep.failure_type_public_or_controlled,
        label_source="template_authored_executable_silver",
        metadata={
            "scientific_status": "silver_synthetic_training_only",
            "not_for_paper_main_results": True,
            "contains_gold_in_user_input": False,
        },
    )


def to_chat_row(ex: StageCTrainingExample, augmentation_name: str, mapping: dict[str, str]) -> dict[str, Any]:
    user_text = format_user_prompt(ex)
    assistant_text = format_assistant_response(ex)

    # Give each augmented row its own stable identifier namespace while ensuring
    # IDs referred to in the assistant action remain visible in the prompt.
    augmented_episode_id = f"{ex.episode_id}__{augmentation_name}"
    user_text = user_text.replace(ex.episode_id, augmented_episode_id)
    assistant_text = assistant_text.replace(ex.episode_id, augmented_episode_id)

    user_text = apply_replacements(user_text, mapping)
    assistant_text = apply_replacements(assistant_text, mapping)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "source_episode_id": ex.episode_id,
            "augmented_episode_id": augmented_episode_id,
            "source_submission_id": ex.submission_id,
            "domain": ex.domain,
            "failure_type": ex.failure_type,
            "augmentation": augmentation_name,
            "actions": [t.action_type for t in ex.targets],
            "scientific_status": "silver_synthetic_training_only",
            "not_for_paper_main_results": True,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def main() -> None:
    episodes_with_gold = generate_expanded_episodes()

    labeled_examples: list[StageCTrainingExample] = []
    for ep, gold in episodes_with_gold:
        targets_by_submission: dict[str, list[TypedRevisionTarget]] = defaultdict(list)
        for target in gold.gold_typed_targets:
            targets_by_submission[target.submission_id].append(target)

        # IMPORTANT: retain only actual revision decisions.
        # Do not convert initial writes into artificial NO_REVISION labels.
        for submission in ep.submissions:
            targets = tuple(targets_by_submission.get(submission.submission_id, ()))
            if not targets:
                continue
            labeled_examples.append(make_example(ep, submission, targets))

    train_source = [ex for ex in labeled_examples if not ex.episode_id.endswith("_v5")]
    heldout_source = [ex for ex in labeled_examples if ex.episode_id.endswith("_v5")]

    if not train_source or not heldout_source:
        raise RuntimeError(
            f"Expected train and held-out sources; got train={len(train_source)}, heldout={len(heldout_source)}"
        )

    train_rows: list[dict[str, Any]] = []
    for ex in train_source:
        for aug_name, mapping in TRAIN_AUGMENTATIONS:
            train_rows.append(to_chat_row(ex, aug_name, mapping))

    # Keep heldout prompts unaugmented: no near-clone contamination into validation/test.
    rng = random.Random(RNG_SEED)
    rng.shuffle(heldout_source)
    midpoint = max(1, len(heldout_source) // 2)
    valid_source = heldout_source[:midpoint]
    test_source = heldout_source[midpoint:]
    if not test_source:
        test_source = valid_source[-1:]
        valid_source = valid_source[:-1]

    valid_rows = [to_chat_row(ex, "heldout_base", {}) for ex in valid_source]
    test_rows = [to_chat_row(ex, "heldout_base", {}) for ex in test_source]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_hash = write_jsonl(OUT_DIR / "train.jsonl", train_rows)
    valid_hash = write_jsonl(OUT_DIR / "valid.jsonl", valid_rows)
    test_hash = write_jsonl(OUT_DIR / "test.jsonl", test_rows)

    action_counts = Counter()
    failure_counts = Counter()
    for ex in labeled_examples:
        failure_counts[ex.failure_type] += 1
        for target in ex.targets:
            action_counts[target.action_type] += 1

    manifest = {
        "dataset_name": "retrace_stagec_qwen3_4b_silver_v0",
        "scientific_status": "silver_synthetic_training_only",
        "not_for_paper_main_results": True,
        "source": "experiments.multiagent.dev_expansion.generate_expanded_episodes",
        "external_benchmark_used_for_training": False,
        "stale_reserved_for_external_evaluation": True,
        "base_episode_count": len(episodes_with_gold),
        "revision_labeled_source_examples": len(labeled_examples),
        "train_source_examples_before_augmentation": len(train_source),
        "heldout_source_examples": len(heldout_source),
        "train_row_count": len(train_rows),
        "valid_row_count": len(valid_rows),
        "test_row_count": len(test_rows),
        "train_augmentation_count": len(TRAIN_AUGMENTATIONS),
        "split_rule": "episode template v1-v4 train with surface augmentations; v5 held out unaugmented for valid/test",
        "action_counts_before_augmentation": dict(sorted(action_counts.items())),
        "failure_type_counts_before_augmentation": dict(sorted(failure_counts.items())),
        "hashes": {
            "train_sha256": train_hash,
            "valid_sha256": valid_hash,
            "test_sha256": test_hash,
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        },
        "limitations": [
            "Synthetic silver data has not been individually human-approved for paper claims.",
            "Surface augmentations improve lexical variation but do not add new reasoning structures.",
            "Only actions present in current executable templates are supervised.",
            "Use this dataset for exploratory adapter training, not final reported evaluation.",
        ],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nWrote MLX training data to: {OUT_DIR}")
    print(f"Wrote manifest to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
