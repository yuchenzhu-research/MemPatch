"""Build MemPatch v1.3 scenario records from blueprint instances.

The upstream ``UnifiedRendererV13`` (ReTrace blueprint repo) is not vendored
here. This module defines the contract and metadata envelope; rendering raises
``RendererNotAvailableError`` until upstream is wired in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from benchmark.generation.blueprints import (
    BENCHMARK_VERSION,
    RENDERER,
    V13BlueprintInstance,
    V13DecisionVariant,
    V13PatternFamily,
)
from benchmark.generation.decision_resolver import resolve_expected_decision
from benchmark.generation.unified_renderer_v13 import DEFAULT_RENDERER, MemPatchUnifiedRendererV13


class RendererNotAvailableError(RuntimeError):
    """Raised when unified_renderer_v13 is not installed."""


class UnifiedRendererV13(Protocol):
    """Upstream renderer contract."""

    def render(
        self,
        *,
        blueprint: V13BlueprintInstance,
        variant: V13DecisionVariant,
        family: V13PatternFamily,
        seed: int,
    ) -> dict[str, Any]:
        """Return public_input, tasks, hidden_gold fields (except expected_decision)."""
        ...


@dataclass
class ScenarioBuildResult:
    scenario: dict[str, Any]
    resolver_trace: dict[str, Any]


def split_seed(blueprint: V13BlueprintInstance) -> int:
    payload = (
        f"{blueprint.split_seed_namespace}:{blueprint.split}:"
        f"{blueprint.pattern}:{blueprint.decision_variant}:{blueprint.scenario_num}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return 1_000_000 + (int(digest[:8], 16) % 8_000_000)


def core_event_signature_from_public(public_input: dict[str, Any], *, max_events: int = 3) -> str:
    """Readable signature stored in metadata for audit (hash computed by audit script)."""
    events = list(public_input.get("event_trace") or [])[:max_events]
    parts: list[str] = []
    for event in events:
        parts.append(
            "|".join(
                [
                    str(event.get("actor_role") or ""),
                    str(event.get("trust_level") or ""),
                    str(event.get("visibility_scope") or ""),
                    str(event.get("event_type") or ""),
                    str(event.get("text") or "")[:60],
                ]
            )
        )
    return " ;; ".join(parts) if parts else "<empty>"


def build_scenario_metadata(
    *,
    blueprint: V13BlueprintInstance,
    variant: V13DecisionVariant,
    seed: int,
    surface_template_id: str,
    public_input: dict[str, Any],
    resolver_trace: dict[str, Any],
    primary_failure_mode: str,
) -> dict[str, Any]:
    decision = variant.decision
    return {
        "schema_version": "retrace_bench_general_1",
        "renderer": RENDERER,
        "benchmark_version": BENCHMARK_VERSION,
        "split": blueprint.split,
        "pattern": blueprint.pattern,
        "pattern_trap_type": blueprint.pattern_trap_type,
        "decision_variant": blueprint.decision_variant,
        "decision_triggers": list(blueprint.decision_triggers),
        "surface_template_id": surface_template_id,
        "core_event_signature": core_event_signature_from_public(public_input),
        "split_seed_namespace": blueprint.split_seed_namespace,
        "seed": seed,
        "mark_ci_derived": variant.mark_ci_derived,
        "resolver_trace": resolver_trace,
        "has_distractor": True,
        "has_cross_scope_trap": True,
        "verified_contradicts_trusted_note": decision == "use_current_memory",
        "requires_rejecting_false_premise": decision == "use_current_memory",
        "requires_non_answer_action": decision
        in ("ask_clarification", "escalate", "mark_unresolved", "refuse_due_to_policy"),
        "canonical_failure_mode": primary_failure_mode,
    }


def build_scenario(
    *,
    blueprint: V13BlueprintInstance,
    variant: V13DecisionVariant,
    family: V13PatternFamily,
    renderer: UnifiedRendererV13 | MemPatchUnifiedRendererV13 | None = None,
) -> ScenarioBuildResult:
    """Build one scenario; gold decision comes from resolver after render."""
    if renderer is None:
        renderer = DEFAULT_RENDERER

    seed = split_seed(blueprint)
    rendered = renderer.render(
        blueprint=blueprint,
        variant=variant,
        family=family,
        seed=seed,
    )
    public_input = rendered["public_input"]
    expected_decision, resolver_trace = resolve_expected_decision(
        blueprint=blueprint,
        variant=variant,
        public_input=public_input,
    )

    hidden_gold = dict(rendered.get("hidden_gold") or {})
    hidden_gold["expected_decision"] = expected_decision

    scenario: dict[str, Any] = {
        "scenario_id": blueprint.scenario_id,
        "pattern": blueprint.pattern,
        "benchmark_version": BENCHMARK_VERSION,
        "public_split_name": blueprint.split,
        "domain": family.domain,
        "primary_failure_mode": family.primary_failure_mode,
        "difficulty": blueprint.difficulty,
        "workflow_context": rendered.get("workflow_context", ""),
        "source_type": "controlled_synthetic",
        "source_pointers": rendered.get(
            "source_pointers",
            [
                {
                    "kind": "synthetic_blueprint",
                    "url_or_id": blueprint.blueprint_id,
                    "license_or_terms_note": "v1.3 generator blueprint instance",
                }
            ],
        ),
        "public_input": public_input,
        "hidden_gold": hidden_gold,
        "metadata": build_scenario_metadata(
            blueprint=blueprint,
            variant=variant,
            seed=seed,
            surface_template_id=str(rendered.get("surface_template_id") or blueprint.decision_variant),
            public_input=public_input,
            resolver_trace=resolver_trace,
            primary_failure_mode=family.primary_failure_mode,
        ),
    }
    for key in (
        "black_box_task",
        "memory_state_task",
        "evidence_retrieval_task",
        "diagnostic_task",
        "tasks",
        "difficulty_level",
        "difficulty_factors",
    ):
        if key in rendered:
            scenario[key] = rendered[key]

    return ScenarioBuildResult(scenario=scenario, resolver_trace=resolver_trace)
