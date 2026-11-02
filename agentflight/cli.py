"""`agentflight` CLI: record | replay | execute | report | check | budget.

--mock runs everything against the deterministic MockBackend so the full
pipeline is exercisable on CPU (E1 dry runs, tests) — the mock is clearly
labeled in every output row it produces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from agentflight.budget.controller import BudgetController
from agentflight.budget.ledger import GpuLedger
from agentflight.harness.llm import MockBackend, OpenAIBackend
from agentflight.harness.loop import HarnessConfig, LLMSummarizer, NullSummarizer, run_task
from agentflight.policies import POLICY_RUN_ORDER
from agentflight.policies.base import PolicyConfig
from agentflight.scoring.abandonment import RunLimits
from agentflight.traces.schema import RunMetadata
from agentflight.traces.writer import TraceWriter, validate_trace_file


def _load_suite(path: str):
    from agentflight.tasks.schema import AgentFlightSuite

    with open(path, encoding="utf-8") as f:
        return AgentFlightSuite.model_validate(json.load(f))


def _select_tasks(suite, tier: str | None, limit: int | None, task_ids: list[str] | None):
    tasks = suite.tasks
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
    if tier:
        tasks = [t for t in tasks if t.context_tier == tier]
    if limit:
        tasks = tasks[:limit]
    return tasks


def _skip_existing(results_dir: Path, task_id: str, policy_id: str, condition: str) -> bool:
    marker = results_dir / f"{condition}-{policy_id}-{task_id}.json.gz"
    return marker.exists()


def _make_backend(args) -> object:
    if args.mock:
        # Deterministic scripted agent: search -> read -> final answer.
        return MockBackend(
            script=[
                {"think": "I should search for the first relevant entity.", "tool_calls": [{"name": "search", "arguments": {"query": "question"}}]},
                {"think": "Now read the most relevant passage.", "tool_calls": [{"name": "read", "arguments": {"doc_id": "placeholder"}}]},
                {"content": "Based on the passages, I have the answer.\nFINAL ANSWER: unknown"},
            ]
        )
    return OpenAIBackend(base_url=args.base_url, model_id=args.model, parse_mode=args.parse_mode)


def cmd_execute(args) -> int:
    """Mode Q: re-execute tasks under one (or all) policies."""
    suite = _load_suite(args.suite)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks = _select_tasks(suite, args.tier, args.limit, args.task_ids)
    policies = POLICY_RUN_ORDER if args.policies == "all" else args.policies.split(",")
    backend = _make_backend(args)
    model_id = "mock-model" if args.mock else args.model

    for policy_id in policies:
        for task in tasks:
            condition = f"modeq-{args.tag}" if args.tag else "modeq"
            if not args.overwrite and _skip_existing(results_dir, task.task_id, policy_id, condition):
                continue
            if args.mock:
                backend = _make_backend(args)  # fresh script state per task
            meta = RunMetadata(
                model_id=model_id,
                vllm_version=None if args.mock else _vllm_version(),
                temperature=0.0,
                seed=args.seed,
                tool_schema_hash="mock" if args.mock else "native",
                task_id=task.task_id,
                policy_id=policy_id,
                context_tier=task.context_tier,
                tokenizer_backend="heuristic",
            )
            writer = TraceWriter(
                results_dir / f"{condition}-{policy_id}-{task.task_id}.json.gz", meta
            )
            harness_cfg = HarnessConfig(
                policy_id=policy_id,
                policy=PolicyConfig(policy_id=policy_id, budget_tokens=args.budget_tokens),
                limits=RunLimits(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls),
            )
            summarizer = None
            if policy_id in ("P2", "P3", "P6"):
                summarizer = (
                    LLMSummarizer(backend, writer)
                    if not args.mock
                    else _mock_summarizer(writer)
                )
            t0 = time.perf_counter()
            result = run_task(task, suite.corpus, backend, writer, harness_cfg, summarizer=summarizer)
            summary = {
                "record": "task_summary",
                "task_id": task.task_id,
                "policy_id": policy_id,
                "context_tier": task.context_tier,
                "condition": condition,
                "model_id": model_id,
                "mock": bool(args.mock),
                "em": result.em,
                "f1": result.f1,
                "abandoned": result.abandoned,
                "abandonment_reason": result.abandonment_reason,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "gpu_seconds": result.gpu_seconds + (time.perf_counter() - t0) * 0.0,
                "prompt_tokens_total": result.prompt_tokens_total,
                "cached_tokens_total": result.cached_tokens_total,
                "answer": result.answer,
                "gold": task.gold_answer,
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with open(results_dir / f"{condition}-summaries.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
            print(
                f"[{policy_id}/{task.context_tier}] {task.task_id}: "
                f"em={result.em:.2f} turns={result.turns} abandoned={result.abandoned}"
            )
    return 0


def _mock_summarizer(writer):
    # record the call as an event for honesty about cost accounting in mock runs
    class _RecordingSum(NullSummarizer):
        def summarize(self, texts):
            writer.append(
                "policy_event",
                turn=-1,
                t_wall_ms=0.0,
                payload={
                    "policy_id": "mock-summarizer",
                    "decision": "summarized",
                    "detail": f"mock summarizer over {len(texts)} texts",
                    "tokens_before": 0,
                    "tokens_after": 0,
                },
            )
            return super().summarize(texts)

    return _RecordingSum()


def _vllm_version() -> str | None:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return None


def cmd_record(args) -> int:
    """E1: record 60 tasks under P0 (reference traces + P0 baseline)."""
    args.policies = "P0"
    return cmd_execute(args)


def cmd_check(args) -> int:
    """G3: validate trace files against the schema."""
    paths = sorted(Path(args.results).glob("**/*.json.gz"))
    ok = 0
    failures = []
    for p in paths:
        try:
            info = validate_trace_file(p)
            ok += 1
            if args.verbose:
                print(f"OK {p.name}: {info['events']} events, kinds={info['kinds']}")
        except Exception as e:
            failures.append(f"{p.name}: {e}")
    print(f"validated {ok}/{len(paths)} trace files")
    for f_ in failures:
        print("FAIL", f_)
    return 1 if failures else 0


def cmd_report(args) -> int:
    from agentflight.reporting import write_markdown_report

    info = write_markdown_report(args.results, args.out)
    print(json.dumps(info))
    print(f"report written to {args.out}")
    return 0


def cmd_budget(args) -> int:
    ledger = GpuLedger(Path(args.results) / "gpu_ledger.jsonl", weekly_quota_hours=args.weekly_quota)
    controller = BudgetController(Path(args.results) / "phase_state.jsonl")
    if args.mark:
        ledger.append(args.mark, args.hours, args.note, datetime.now(UTC).isoformat(timespec="seconds"))
        controller.mark_complete(args.mark, datetime.now(UTC).isoformat(timespec="seconds"))
    status = controller.status()
    print(json.dumps({**status, "gpu_hours_used": ledger.total_hours(), "gpu_hours_remaining": ledger.remaining()}, indent=2))
    if args.guard:
        nxt = status["next_phase"]
        target = args.guard
        ok, why = controller.can_start(target)
        if not ok:
            print(f"REFUSED: {why}", file=sys.stderr)
            return 1
        print(f"OK to start {target} (next incomplete: {nxt})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentflight")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, suite=True):
        if suite:
            p.add_argument("--suite", default="data/agentflight_qa_v1.json")
        p.add_argument("--results", default="results/")
        p.add_argument("--mock", action="store_true", help="use the deterministic MockBackend (CPU)")
        p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        p.add_argument("--model", default="Qwen/Qwen3-8B-GPTQ-Int4")
        p.add_argument("--parse-mode", default="native", choices=["native", "json_fallback"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--tier", default=None, choices=[None, "4k", "16k", "32k"])
        p.add_argument("--task-ids", default=None, help="comma-separated task ids")
        p.add_argument("--policies", default="all", help=f"comma-separated; all = {POLICY_RUN_ORDER}")
        p.add_argument("--tag", default=None, help="condition tag")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--max-turns", type=int, default=24)
        p.add_argument("--max-tool-calls", type=int, default=12)
        p.add_argument("--budget-tokens", type=int, default=8192)

    p = sub.add_parser("execute", help="Mode Q: run tasks under policies")
    common(p)
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser("record", help="E1: record traces under P0")
    common(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("check", help="validate trace files (G3)")
    p.add_argument("--results", default="results/")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("report", help="aggregate results -> markdown (G5)")
    p.add_argument("--results", default="results/")
    p.add_argument("--out", default="docs/results_report.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("budget", help="GPU ledger + phase controller")
    p.add_argument("--results", default="results/")
    p.add_argument("--weekly-quota", type=float, default=30.0)
    p.add_argument("--mark", default=None, help="mark phase complete (e.g. E0)")
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--note", default="")
    p.add_argument("--guard", default=None, help="exit 1 unless phase can start")
    p.set_defaults(func=cmd_budget)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
