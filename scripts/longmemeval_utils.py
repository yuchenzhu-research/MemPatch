"""Shared helpers for LongMemEval baseline inference and local evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

POLICIES = ("direct", "retrieve-all", "latest-only")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to prior chat history. "
    "Answer the question using only information supported by the history. "
    "Reply with a concise direct answer only; do not explain your reasoning."
)


def load_longmemeval(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def filter_examples(
    examples: list[dict[str, Any]],
    *,
    question_type: str | None,
    limit: int | None,
    offset: int,
) -> list[dict[str, Any]]:
    selected = examples
    if question_type is not None:
        selected = [ex for ex in selected if ex.get("question_type") == question_type]
    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def select_sessions(example: dict[str, Any], policy: str, latest_sessions: int) -> tuple[list[str], list[list[dict[str, Any]]]]:
    dates = list(example.get("haystack_dates") or [])
    sessions = list(example.get("haystack_sessions") or [])
    if len(dates) != len(sessions):
        raise ValueError(
            f"{example.get('question_id')}: haystack_dates/sessions length mismatch "
            f"({len(dates)} vs {len(sessions)})"
        )

    if policy in {"direct", "retrieve-all"}:
        return dates, sessions
    if policy == "latest-only":
        if latest_sessions <= 0:
            raise ValueError("latest-only requires --latest-sessions > 0")
        return dates[-latest_sessions:], sessions[-latest_sessions:]
    raise ValueError(f"unsupported policy: {policy}")


def session_payload(date: str, turns: list[dict[str, Any]], *, user_only: bool) -> dict[str, Any]:
    cleaned_turns: list[dict[str, str]] = []
    for turn in turns:
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if user_only and role != "user":
            continue
        cleaned_turns.append({"role": role, "content": content})
    return {"date": date, "turns": cleaned_turns}


def build_history_block(
    example: dict[str, Any],
    *,
    policy: str,
    latest_sessions: int,
    user_only: bool,
) -> list[dict[str, Any]]:
    dates, sessions = select_sessions(example, policy, latest_sessions)
    return [session_payload(date, turns, user_only=user_only) for date, turns in zip(dates, sessions)]


def build_chat_messages(
    example: dict[str, Any],
    *,
    policy: str,
    latest_sessions: int,
    user_only: bool = False,
) -> list[dict[str, str]]:
    history = build_history_block(
        example,
        policy=policy,
        latest_sessions=latest_sessions,
        user_only=user_only,
    )
    question_date = example.get("question_date", "")
    question = example.get("question", "")
    user_content = (
        "Chat history (JSON):\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
        f"Question date: {question_date}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize_text(text: str | int | float) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def heuristic_label(
    question_type: str,
    question: str,
    answer: str | int | float,
    hypothesis: str,
) -> bool:
    hyp = normalize_text(hypothesis)
    gold = normalize_text(answer)
    if not hyp or not gold:
        return False
    if gold in hyp:
        return True
    if question_type == "temporal-reasoning":
        numbers = re.findall(r"\d+", gold)
        if numbers and all(num in hyp for num in numbers):
            return True
    return False


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    if not abstention:
        if task in {"single-session-user", "single-session-assistant", "multi-session"}:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
                "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
                "the model's response is still correct.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response "
                "should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long as "
                "it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(f"unsupported question_type: {task}")

    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def parse_yes_no(text: str) -> bool:
    cleaned = normalize_text(text)
    if cleaned.startswith("yes"):
        return True
    if cleaned.startswith("no"):
        return False
    return "yes" in cleaned and "no" not in cleaned
