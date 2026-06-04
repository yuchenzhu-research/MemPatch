#!/usr/bin/env python3
"""Run ReTrace-Bench eval via SiliconFlow OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import canonical_hidden_gold_fields
from benchmark.retrace_bench.public_view import public_scenario_view
from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction
from scripts.run_retrace_bench_baseline import (
    _parse_llm_json_response,
    append_jsonl,
    read_jsonl,
    read_resume_jsonl,
)

REPORT_METRICS = (
    "decision_macro_f1",
    "black_box_decision_accuracy",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "minimal_evidence_exact_match",
    "evidence_precision",
    "overcitation_rate",
    "counterevidence_recall",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
    "latest_event_shortcut_failure_rate",
    "answer_state_consistency",
    "joint_revision_success",
    "format_failure_rate",
)

ALLOWED_DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

ALLOWED_MEMORY_STATUSES = (
    "current",
    "outdated",
    "blocked",
    "unresolved",
    "out_of_scope",
    "deleted",
    "should_not_store",
    "restored",
)

SILICONFLOW_DEFAULT_MODELS = (
    "Pro/moonshotai/Kimi-K2.6",
    "Pro/zai-org/GLM-5.1",
    "deepseek-ai/DeepSeek-V4-Pro",
)

ALLOWED_FAILURE_DIAGNOSIS = (
    "stale_memory_reuse",
    "under_update",
    "over_update",
    "conflict_collapse",
    "scope_leakage",
    "policy_violation",
    "wrong_source_attribution",
    "memory_hallucination",
    "unnecessary_memory_write",
    "failure_to_forget",
    "failure_to_release_or_restore",
)

CORE_INSTRUCTION = """You are evaluating an agent-memory revision benchmark.

Use only the visible scenario content. Do not use external knowledge.

Output strict JSON only, with this exact structure:
{
  "scenario_id": "<copy scenario_id>",
  "response": {
    "decision": "<one allowed decision label>",
    "answer": "<short answer grounded only in visible evidence>",
    "memory_state": {
      "<memory_id>": "<one allowed memory_state label>"
    },
    "evidence_event_ids": ["<minimal supporting event ids>"],
    "failure_diagnosis": "<one allowed failure_diagnosis label>"
  }
}

Rules:
1. The memory_state object must include every memory_id from public_input.initial_memory.
2. Cite only minimal supporting evidence_event_ids.
3. Do not cite every event.
4. Respect visibility_scope, branch, version, release state, CI status, actor_role, trust_level, reviewer/maintainer authority, and policy constraints.
5. If evidence is conflicting or insufficient, choose ask_clarification, mark_unresolved, or escalate.
6. If evidence applies only to workspace-beta/nightly/dev/next/staging/another scope, do not apply it to workspace-stable unless explicitly stated.
7. If a PR is opened, on hold, or lacks approval, do not treat it as merged, released, or authorized.
8. If CI fails, do not treat the claimed update as authorized.
9. If an untrusted user claims something is resolved, do not treat that as verified evidence.
10. Do not output markdown. Do not explain outside JSON.

Allowed decision labels: """ + ", ".join(ALLOWED_DECISIONS) + """

Allowed memory_state labels: """ + ", ".join(ALLOWED_MEMORY_STATUSES) + """

Allowed failure_diagnosis labels: """ + ", ".join(ALLOWED_FAILURE_DIAGNOSIS)


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def load_api_key_from_env_file(env_file: Path) -> str:
    if not env_file.exists():
        raise SystemExit(f"Env file not found: {env_file}")
    lines = [ln.strip() for ln in env_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"Env file {env_file} is empty; expected API key on first line.")
    first = lines[0]
    if first.startswith("SILICONFLOW_API_KEY="):
        key = first.split("=", 1)[1].strip().strip('"').strip("'")
    else:
        key = first.strip('"').strip("'")
    if not key:
        raise SystemExit(f"Could not parse API key from first line of {env_file}.")
    return key


def redact_key(text: str, api_key: str) -> str:
    return text.replace(api_key, "[REDACTED]") if api_key else text


def build_prompt(scenario: dict[str, Any]) -> str:
    visible = public_scenario_view(scenario)
    return CORE_INSTRUCTION + "\n\nScenario:\n" + json.dumps(visible, ensure_ascii=False)


def normalize_parsed(parsed: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    if "response" in parsed and isinstance(parsed["response"], dict):
        body = parsed["response"]
    else:
        body = parsed
    diag = body.get("failure_diagnosis")
    if isinstance(diag, list):
        body["failure_diagnosis"] = diag[0] if diag else None
    evidence = body.get("evidence_event_ids")
    if isinstance(evidence, str):
        body["evidence_event_ids"] = [evidence]
    elif evidence is None:
        body["evidence_event_ids"] = []
    if not isinstance(body.get("memory_state"), dict):
        body["memory_state"] = {}
    return {
        "scenario_id": parsed.get("scenario_id") or scenario_id,
        "response": {
            "decision": body.get("decision"),
            "answer": body.get("answer"),
            "memory_state": body.get("memory_state", {}),
            "evidence_event_ids": body.get("evidence_event_ids", []),
            "failure_diagnosis": body.get("failure_diagnosis"),
        },
    }


def call_model_with_retry(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    max_attempts: int = 3,
) -> str:
    last_err: Exception | None = None
    extra_kwargs: dict[str, Any] = {}
    if disable_thinking:
        extra_kwargs["extra_body"] = {"enable_thinking": False}
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=180.0,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if attempt < max_attempts - 1:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"request failed after {max_attempts} attempts: {type(last_err).__name__}: {last_err}")


def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["scenario_id"]: row for row in read_resume_jsonl(path)}


def compute_distributions(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    pattern = Counter()
    difficulty = Counter()
    expected_decision = Counter()
    for row in scenarios:
        pattern[row.get("pattern") or row.get("metadata", {}).get("pattern", "unknown")] += 1
        difficulty[row.get("difficulty") or row.get("difficulty_level", "unknown")] += 1
        gold = canonical_hidden_gold_fields(row["hidden_gold"])
        expected_decision[gold["expected_decision"]] += 1
    return {
        "pattern": dict(sorted(pattern.items())),
        "difficulty": dict(sorted(difficulty.items())),
        "expected_decision": dict(sorted(expected_decision.items())),
    }


def run_one_model(
    scenarios: list[dict[str, Any]],
    *,
    model: str,
    client: Any,
    out_dir: Path,
    api_key: str,
    temperature: float,
    max_tokens: int,
    resume: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    name = short_model_name(model)
    pred_path = out_dir / f"{name}.predictions.jsonl"
    raw_path = out_dir / f"{name}.raw_responses.jsonl"
    err_path = out_dir / f"{name}.errors.jsonl"
    metrics_path = out_dir / f"{name}.metrics.json"

    completed = load_completed_predictions(pred_path) if resume else {}
    predictions: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    error_count = 0
    format_failures = 0

    if resume and completed:
        print(f"[resume] {name}: loaded {len(completed)} predictions", flush=True)
        for sid, row in completed.items():
            scenario = next(s for s in scenarios if s["scenario_id"] == sid)
            pred = {
                "scenario_id": sid,
                "model": model,
                "response": row.get("response", {}),
                "metrics": row.get("metrics") or score_prediction(scenario, row),
            }
            predictions.append(pred)
            scored_rows.append(
                {
                    "scenario_id": sid,
                    "expected_decision": scenario["hidden_gold"]["expected_decision"],
                    "response": pred["response"],
                    "metrics": pred["metrics"],
                }
            )
            if float(pred["metrics"].get("format_failure_rate", 0.0)) >= 1.0:
                format_failures += 1

    total = len(scenarios)
    for idx, scenario in enumerate(scenarios, start=1):
        sid = scenario["scenario_id"]
        if sid in completed:
            continue
        if idx % 10 == 0 or idx == 1 or idx == total:
            print(f"[{idx}/{total}] {name} :: {sid}", flush=True)

        prompt = build_prompt(scenario)
        try:
            raw = call_model_with_retry(
                client,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            append_jsonl(raw_path, {"scenario_id": sid, "model": model, "raw_response": raw})
            try:
                parsed = normalize_parsed(_parse_llm_json_response(raw), sid)
            except json.JSONDecodeError as exc:
                parsed = {
                    "scenario_id": sid,
                    "response": {
                        "decision": None,
                        "answer": raw[:500],
                        "memory_state": {},
                        "evidence_event_ids": [],
                        "failure_diagnosis": None,
                    },
                }
                append_jsonl(
                    err_path,
                    {
                        "scenario_id": sid,
                        "model": model,
                        "error": f"json_parse: {exc}",
                    },
                )
            pred = {"scenario_id": sid, "model": model, "response": parsed["response"]}
            pred["metrics"] = score_prediction(scenario, pred)
            append_jsonl(pred_path, pred)
            predictions.append(pred)
            scored_rows.append(
                {
                    "scenario_id": sid,
                    "expected_decision": scenario["hidden_gold"]["expected_decision"],
                    "response": pred["response"],
                    "metrics": pred["metrics"],
                }
            )
            if float(pred["metrics"].get("format_failure_rate", 0.0)) >= 1.0:
                format_failures += 1
        except Exception as exc:
            error_count += 1
            append_jsonl(
                err_path,
                {
                    "scenario_id": sid,
                    "model": model,
                    "error": redact_key(str(exc), api_key),
                },
            )

    aggregate = aggregate_metrics(scored_rows)
    report = {k: float(aggregate.get("all_metrics", {}).get(k, 0.0)) for k in REPORT_METRICS}
    payload = {
        "model": model,
        "short_name": name,
        "count": total,
        "success_count": len(predictions),
        "error_count": error_count,
        "format_failure_count": format_failures,
        "all_metrics": aggregate.get("all_metrics", {}),
        "report_metrics": report,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {pred_path} and {metrics_path}", flush=True)
    return payload


def failure_summary(
    model: str,
    predictions: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fmt_fail = [p for p in predictions if float(p.get("metrics", {}).get("format_failure_rate", 0)) >= 1.0]
    lowest_joint = sorted(predictions, key=lambda p: float(p.get("metrics", {}).get("joint_revision_success", 0)))[:5]
    wrong_diag = Counter()
    overcites = []
    non_answer_fail = []
    for p in predictions:
        sid = p["scenario_id"]
        sc = scenarios[sid]
        gold = canonical_hidden_gold_fields(sc["hidden_gold"])
        pred_diag = (p.get("response") or {}).get("failure_diagnosis")
        if pred_diag and pred_diag != gold.get("expected_failure_diagnosis"):
            wrong_diag[str(pred_diag)] += 1
        oc = float(p.get("metrics", {}).get("overcitation_rate", 0))
        if oc > 0.5:
            overcites.append({"scenario_id": sid, "overcitation_rate": oc})
        exp_dec = gold.get("expected_decision")
        pred_dec = (p.get("response") or {}).get("decision")
        if exp_dec in {"ask_clarification", "mark_unresolved", "escalate", "refuse_due_to_policy"} and pred_dec != exp_dec:
            non_answer_fail.append({"scenario_id": sid, "expected": exp_dec, "predicted": pred_dec})
    return {
        "format_failures": len(fmt_fail),
        "format_failure_examples": [p["scenario_id"] for p in fmt_fail[:5]],
        "lowest_joint_examples": [
            {
                "scenario_id": p["scenario_id"],
                "joint_revision_success": p.get("metrics", {}).get("joint_revision_success"),
            }
            for p in lowest_joint
        ],
        "common_wrong_failure_diagnosis": dict(wrong_diag.most_common(5)),
        "overcitation_examples": overcites[:5],
        "non_answer_decision_failures": non_answer_fail[:5],
    }


def write_summaries(
    *,
    out_dir: Path,
    data_path: Path,
    scenarios: list[dict[str, Any]],
    model_payloads: dict[str, dict[str, Any]],
    legacy_deepseek_metrics: dict[str, float] | None,
) -> None:
    dist = compute_distributions(scenarios)
    scenarios_by_id = {s["scenario_id"]: s for s in scenarios}
    ranking = sorted(
        model_payloads.items(),
        key=lambda kv: float(kv[1].get("report_metrics", {}).get("joint_revision_success", 0.0)),
        reverse=True,
    )

    summary_metrics = {
        "dataset": str(data_path),
        "count": len(scenarios),
        "distributions": dist,
        "models": {name: payload.get("report_metrics", {}) for name, payload in model_payloads.items()},
        "ranking_by_joint_revision_success": [
            {
                "model": name,
                "joint_revision_success": payload.get("report_metrics", {}).get("joint_revision_success"),
            }
            for name, payload in ranking
        ],
    }
    (out_dir / "summary.metrics.json").write_text(
        json.dumps(summary_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ReTrace-Bench v1.1 Hard150 — SiliconFlow Three-Model Eval",
        "",
        f"Dataset: `{data_path}`",
        f"Scenarios: {len(scenarios)}",
        "",
        "## Dataset distributions",
        "",
        f"- difficulty: `{dist['difficulty']}`",
        f"- pattern: `{dist['pattern']}`",
        f"- expected_decision: `{dist['expected_decision']}`",
        "",
        "## Main metrics",
        "",
        "| model | " + " | ".join(REPORT_METRICS) + " |",
        "| --- | " + " | ".join(["---"] * len(REPORT_METRICS)) + " |",
    ]
    for name, payload in model_payloads.items():
        m = payload.get("report_metrics", {})
        row = " | ".join(f"{float(m.get(k, 0.0)):.3f}" for k in REPORT_METRICS)
        lines.append(f"| {name} | {row} |")

    lines.extend(["", "## Ranking by joint_revision_success", ""])
    for i, (name, payload) in enumerate(ranking, start=1):
        joint = payload.get("report_metrics", {}).get("joint_revision_success", 0.0)
        lines.append(f"{i}. **{name}** — joint_revision_success={joint:.3f}")

    lines.extend(["", "## Per-model failure summaries", ""])
    for name in model_payloads:
        preds = read_jsonl(out_dir / f"{name}.predictions.jsonl")
        fs = failure_summary(name, preds, scenarios_by_id)
        lines.extend(
            [
                f"### {name}",
                f"- format failures: {fs['format_failures']}",
                f"- lowest joint examples: `{fs['lowest_joint_examples']}`",
                f"- common wrong failure_diagnosis: `{fs['common_wrong_failure_diagnosis']}`",
                f"- overcitation examples: `{fs['overcitation_examples']}`",
                f"- non-answer decision failures: `{fs['non_answer_decision_failures']}`",
                "",
            ]
        )

    lines.extend(["", "## Legacy DeepSeek hard150 comparison", ""])
    new_ds = model_payloads.get("DeepSeek-V4-Pro", {}).get("report_metrics", {})
    if legacy_deepseek_metrics and new_ds:
        old_joint = legacy_deepseek_metrics.get("joint_revision_success")
        new_joint = new_ds.get("joint_revision_success")
        lines.append(
            f"- Previous run (`outputs/retrace_bench_hard150/`): joint={old_joint:.3f}, "
            f"failure_diagnosis={legacy_deepseek_metrics.get('failure_diagnosis_accuracy', 0):.3f}"
        )
        lines.append(
            f"- This run: joint={new_joint:.3f}, "
            f"failure_diagnosis={new_ds.get('failure_diagnosis_accuracy', 0):.3f}"
        )
        delta = abs(float(new_joint or 0) - float(old_joint or 0))
        if delta > 0.05:
            lines.append(
                "- Difference >0.05 likely due to prompt/runner path change (this eval uses "
                "public-field-only SiliconFlow runner with expanded instruction rules)."
            )
        else:
            lines.append("- Results are broadly consistent with the prior hard150 DeepSeek run.")
    else:
        lines.append("- Legacy DeepSeek metrics not available for comparison.")

    all_joint = [float(p.get("report_metrics", {}).get("joint_revision_success", 0)) for p in model_payloads.values()]
    max_joint = max(all_joint) if all_joint else 0.0
    hard_enough = max_joint < 0.7
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- **Hard150 difficulty:** {'sufficiently hard' if hard_enough else 'may be too easy'} "
            f"(max joint_revision_success={max_joint:.3f}).",
            f"- **Strongest model:** {ranking[0][0] if ranking else 'n/a'}.",
            "- **Most diagnostic metrics:** joint_revision_success, failure_diagnosis_accuracy, "
            "minimal_evidence_exact_match, overcitation_rate.",
            f"- **Expand to hard500:** {'yes, after fixing decision skew' if max_joint < 0.5 else 'conditional / not urgent'}.",
            f"- **Paper stress split:** {'yes — hard150 is suitable as v1.1 headline stress split' if max_joint < 0.5 else 'use with caution; some models may be saturating joint metrics'}.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmp_lines = [
        "# Model Comparison — SiliconFlow Hard150",
        "",
        "## Joint ranking",
        "",
        "| rank | model | joint_revision_success | failure_diagnosis_accuracy | evidence_f1 | format_failure_rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, (name, payload) in enumerate(ranking, start=1):
        m = payload.get("report_metrics", {})
        cmp_lines.append(
            f"| {i} | {name} | {m.get('joint_revision_success', 0):.3f} | "
            f"{m.get('failure_diagnosis_accuracy', 0):.3f} | {m.get('evidence_f1', 0):.3f} | "
            f"{m.get('format_failure_rate', 0):.3f} |"
        )
    cmp_lines.extend(["", "## Headline takeaway", ""])
    if ranking:
        best = ranking[0]
        cmp_lines.append(
            f"- Best overall: **{best[0]}** (joint={best[1]['report_metrics'].get('joint_revision_success', 0):.3f})"
        )
    (out_dir / "model_comparison.md").write_text("\n".join(cmp_lines) + "\n", encoding="utf-8")


def run_models_parallel(
    models: list[str],
    scenarios: list[dict[str, Any]],
    *,
    out_dir: Path,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    resume: bool,
    disable_thinking: bool,
) -> dict[str, dict[str, Any]]:
    def _worker(model: str) -> tuple[str, dict[str, Any]]:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        payload = run_one_model(
            scenarios,
            model=model,
            client=client,
            out_dir=out_dir,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            resume=resume,
            disable_thinking=disable_thinking,
        )
        return short_model_name(model), payload

    payloads: dict[str, dict[str, Any]] = {}
    workers = min(len(models), 3)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, model): model for model in models}
        for future in as_completed(futures):
            name, payload = future.result()
            payloads[name] = payload
    return payloads


def load_model_metrics_from_disk(out_dir: Path, models: list[str]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for model in models:
        name = short_model_name(model)
        path = out_dir / f"{name}.metrics.json"
        if path.exists():
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SiliconFlow models on ReTrace-Bench.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models",
        default=",".join(SILICONFLOW_DEFAULT_MODELS),
        help="Comma-separated model ids (default: gated pilot trio)",
    )
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple models concurrently (one thread per model).",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip API calls; rebuild summary files from existing metrics JSON.",
    )
    args = parser.parse_args(argv)

    api_key = load_api_key_from_env_file(Path(args.env_file))
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("openai package is required") from exc

    scenarios = read_jsonl(Path(args.data))
    if args.max_cases is not None:
        scenarios = scenarios[: args.max_cases]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    legacy_path = Path(
        "outputs/retrace_bench_siliconflow_hard150/DeepSeek-V4-Pro.metrics.json"
    )
    legacy_deepseek = None
    if legacy_path.exists():
        leg = json.loads(legacy_path.read_text(encoding="utf-8"))
        src = leg.get("report_metrics") or leg.get("all_metrics") or {}
        legacy_deepseek = {k: float(src[k]) for k in REPORT_METRICS if k in src}

    if args.summaries_only:
        model_payloads = load_model_metrics_from_disk(out_dir, models)
    elif args.parallel:
        model_payloads = run_models_parallel(
            models,
            scenarios,
            out_dir=out_dir,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=args.resume,
            disable_thinking=args.disable_thinking,
        )
    else:
        client = openai.OpenAI(api_key=api_key, base_url=args.base_url.rstrip("/"))
        model_payloads = {}
        for model in models:
            payload = run_one_model(
                scenarios,
                model=model,
                client=client,
                out_dir=out_dir,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                resume=args.resume,
                disable_thinking=args.disable_thinking,
            )
            model_payloads[short_model_name(model)] = payload

    write_summaries(
        out_dir=out_dir,
        data_path=Path(args.data),
        scenarios=scenarios,
        model_payloads=model_payloads,
        legacy_deepseek_metrics=legacy_deepseek,
    )
    print(f"Wrote summaries under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
