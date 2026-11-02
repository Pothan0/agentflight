"""Distractor-leak check (build-prompt section 6 VERIFY).

Guarantee: the gold answer string must NOT appear in a task's distractor
passages (it must appear in the supporting passages). Yes/no and single-char
answers are exempt from the string test (any English corpus contains them);
this exemption is recorded in the suite notes.

The builder uses this check *during* distractor sampling (a leaking candidate
is rejected and replaced), and the frozen suite is re-checked wholesale by
`python -m agentflight.tasks.leak_check`.
"""

from __future__ import annotations

from agentflight.scoring.qa import normalize_answer
from agentflight.tasks.schema import AgentFlightSuite, Passage, QATask

_MIN_ANSWER_LEN = 4  # shorter entity strings are too leak-prone to test by substring


def is_string_checkable(answer: str, answer_type: str) -> bool:
    if answer_type == "yesno":
        return False
    return len(normalize_answer(answer)) >= _MIN_ANSWER_LEN


def contains_answer(text: str, answer: str) -> bool:
    """Normalized case-insensitive substring test."""
    na = normalize_answer(answer)
    if not na:
        return False
    return na in normalize_answer(text)


def check_task(task: QATask, docs_by_id: dict[str, Passage]) -> dict:
    """Returns {"pass": bool, "violations": [...], "exempt": bool, ...}."""
    violations: list[str] = []
    exempt = not is_string_checkable(task.gold_answer, task.answer_type)

    support_texts = [docs_by_id[d].text for d in task.support_doc_ids]
    distractor_texts = [docs_by_id[d].text for d in task.distractor_doc_ids]
    if not support_texts:
        violations.append("no supporting passages")

    if not exempt:
        for did in task.distractor_doc_ids:
            if contains_answer(docs_by_id[did].text, task.gold_answer):
                violations.append(f"answer appears in distractor {did}")
        if not any(contains_answer(t, task.gold_answer) for t in support_texts):
            violations.append("answer absent from ALL supporting passages (support check)")
    return {
        "task_id": task.task_id,
        "pass": not violations,
        "exempt": exempt,
        "violations": violations,
        "n_distractors": len(distractor_texts),
    }


def check_suite(suite: AgentFlightSuite) -> dict:
    docs_by_id = {p.doc_id: p for p in suite.corpus}
    results = [check_task(t, docs_by_id) for t in suite.tasks]
    failed = [r for r in results if not r["pass"]]
    return {
        "suite": suite.metadata.suite_name,
        "n_tasks": len(results),
        "n_exempt": sum(1 for r in results if r["exempt"]),
        "n_failed": len(failed),
        "pass": not failed,
        "failures": failed,
    }
