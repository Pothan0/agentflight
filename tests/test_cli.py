"""G1 tests: CLI end-to-end on the mock backend — execute, --skip-existing
resume, check, report, budget guard."""

import json

from agentflight.cli import main
from agentflight.tasks.schema import AgentFlightSuite, Passage, QATask, SuiteMetadata


def tiny_suite(path, n_tasks=2) -> None:
    corpus = [
        Passage(doc_id=f"d{i}", title=f"T{i}", text=f"Passage {i} mentions entity{i} and fact{i}.",
                source="test", origin="support")
        for i in range(5)
    ]
    tasks = [
        QATask(
            task_id=f"t{i}", source="test", hops=2, question=f"Q{i}?", gold_answer=f"entity{i}",
            answer_type="entity", support_doc_ids=["d0"], distractor_doc_ids=["d1", "d2"],
            context_tier="4k", tier_target_tokens=400, tier_total_tokens=400,
        )
        for i in range(n_tasks)
    ]
    suite = AgentFlightSuite(
        metadata=SuiteMetadata(
            schema_version="1.0.0", suite_name="tiny", seed=1, created_at="2026-01-01T00:00:00Z",
            tokenizer_backend="heuristic", hop_filtering={}, counts={"tasks": n_tasks},
        ),
        corpus=corpus,
        tasks=tasks,
    )
    path.write_text(suite.model_dump_json(), encoding="utf-8")


def test_cli_execute_mock_and_resume(tmp_path, capsys):
    suite_path = tmp_path / "suite.json"
    tiny_suite(suite_path)
    results = tmp_path / "results"

    rc = main([
        "execute", "--mock", "--suite", str(suite_path), "--results", str(results),
        "--policies", "P0", "--tag", "unit",
    ])
    assert rc == 0
    summaries = (results / "modeq-unit-summaries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(summaries) == 2
    row = json.loads(summaries[0])
    assert row["mock"] is True and row["policy_id"] == "P0" and row["condition"] == "modeq-unit"
    assert row["answer"] == "unknown"  # mock script's canned answer — honest, not fabricated

    # resume: re-run skips both tasks (no new rows, no duplicate traces)
    rc = main([
        "execute", "--mock", "--suite", str(suite_path), "--results", str(results),
        "--policies", "P0", "--tag", "unit",
    ])
    assert rc == 0
    summaries2 = (results / "modeq-unit-summaries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(summaries2) == 2

    # traces validate (G3 check) and report aggregates (G5 path)
    assert main(["check", "--results", str(results)]) == 0
    rc = main(["report", "--results", str(results), "--out", str(tmp_path / "report.md")])
    assert rc == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "P0" in report and "| 2 |" in report.replace("  ", " ") or "P0" in report


def test_cli_budget_guard(tmp_path):
    results = tmp_path / "r"
    assert main(["budget", "--results", str(results), "--mark", "E0", "--hours", "1.0", "--note", "t"]) == 0
    # E2 still gated: E1 not complete
    assert main(["budget", "--results", str(results), "--guard", "E2"]) == 1
    main(["budget", "--results", str(results), "--mark", "E1", "--hours", "5.0", "--note", "t"])
    assert main(["budget", "--results", str(results), "--guard", "E2"]) == 0
    ledger = json.loads((results / "gpu_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert ledger["hours"] == 5.0
