"""The agent loop (minimal ReAct with OpenAI-compatible function calling).

The loop NEVER sees the raw history except through the policy: every turn,
the policy constructs the exact view; the loop records the view (and the
policy's decisions as policy_events), calls the backend, executes tool calls,
and appends canonical messages. The SAME loop code runs every policy — only
the view-construction function differs (build-prompt section 5).

P2/P3/P6 need an LLM inside the policy: the summarizer wraps the same
backend at temperature 0 and its calls are recorded as events too (their
cost counts in GPU-seconds).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from agentflight.harness.llm import LLMBackend, LLMResponse
from agentflight.harness.tools import TOOL_SCHEMAS, ToolRegistry, tool_schema_hash
from agentflight.policies import get_policy
from agentflight.policies.base import PolicyConfig, PolicyResult, Summarizer
from agentflight.scoring.abandonment import RunLimits, check_abandonment
from agentflight.scoring.qa import extract_final_answer, score_answer
from agentflight.tasks.schema import Passage, QATask
from agentflight.traces.schema import messages_view_hash
from agentflight.traces.writer import TraceWriter

SYSTEM_PROMPT = """You are a precise research assistant answering multi-hop questions.

You have these tools:
- search(query): search a local passage corpus, returns the top-5 passages with doc_ids.
- read(doc_id): read the full text of one passage.
- calculator(expr): evaluate an arithmetic expression.

Work step by step: think before acting, search for what you need, read the
relevant passages, and combine facts across documents. Do not guess — verify
each hop. When you know the answer, stop calling tools and reply with a
short final message that ends with exactly:
FINAL ANSWER: <the answer>
The final answer must be short: an entity, name, number, date, or yes/no."""


class LLMSummarizer(Summarizer):
    """Policy summarizer backed by the same served model, temperature 0.
    Calls are recorded as llm_request/llm_response events with a policy note
    so their GPU cost is counted."""

    def __init__(self, backend: LLMBackend, writer: TraceWriter | None = None):
        self.backend = backend
        self.writer = writer
        self.n_calls = 0

    def summarize(self, texts: list[str]) -> str:
        prompt = (
            "Summarize the following conversation excerpts into a compact, "
            "information-dense summary. Keep every fact, name, number and date "
            "that could be needed to answer a question later. Be brief.\n\n"
            + "\n".join(texts)
        )
        messages = [
            {"role": "system", "content": "You compress agent conversation history. Output only the summary."},
            {"role": "user", "content": prompt},
        ]
        t0 = time.perf_counter()
        resp = self.backend.chat(messages, tools=None, temperature=0.0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.n_calls += 1
        if self.writer is not None:
            self.writer.append(
                "llm_request",
                turn=-1,
                t_wall_ms=(time.perf_counter() - self._t_start) * 1000.0
                if hasattr(self, "_t_start")
                else 0.0,
                payload={"messages": messages, "note": "policy_summarizer"},
                messages_view_hash_=messages_view_hash(messages),
            )
            self._record_response(resp, total_ms)
        return resp.visible_text.strip()

    def _record_response(self, resp: LLMResponse, total_ms: float) -> None:
        self.writer.append(
            "llm_response",
            turn=-1,
            t_wall_ms=(time.perf_counter() - self._t_start) * 1000.0 if hasattr(self, "_t_start") else 0.0,
            payload={
                "usage": resp.usage,
                "timings_ms": {"ttft": resp.timings_ms.get("ttft"), "total": total_ms, "inter_token": []},
                "completion_text": resp.completion_text,
                "think_text": resp.think_text,
                "note": "policy_summarizer",
            },
        )

    def bind_clock(self, t_start: float) -> None:
        self._t_start = t_start


class NullSummarizer(Summarizer):
    """Deterministic scripted summarizer for CPU tests: keeps the first line
    of each text plus a count. Pure, no LLM."""

    def summarize(self, texts: list[str]) -> str:
        lines = []
        for t in texts[:12]:
            first = t.strip().splitlines()[0] if t.strip() else ""
            lines.append(first[:160])
        return " | ".join(lines) + f" ({len(texts)} messages)"


@dataclass
class HarnessConfig:
    policy_id: str = "P0"
    policy: PolicyConfig = None  # type: ignore[assignment]
    limits: RunLimits = None  # type: ignore[assignment]
    thinking: bool = True
    thinking_budget_tokens: int = 768
    parse_mode: str = "native"

    def __post_init__(self):
        if self.policy is None:
            self.policy = PolicyConfig(policy_id=self.policy_id)
        if self.limits is None:
            self.limits = RunLimits()


class AgentRunResult:
    def __init__(self):
        self.answer: str | None = None
        self.abandoned: bool = False
        self.abandonment_reason: str | None = None
        self.em: float = 0.0
        self.f1: float = 0.0
        self.turns: int = 0
        self.tool_calls: int = 0
        self.gpu_seconds: float = 0.0
        self.prompt_tokens_total: int = 0
        self.cached_tokens_total: int = 0


def run_task(
    task: QATask,
    corpus: list[Passage],
    backend: LLMBackend,
    writer: TraceWriter,
    config: HarnessConfig,
    summarizer: Summarizer | None = None,
) -> AgentRunResult:
    """Run ONE task under ONE policy; append all events to ``writer``."""
    policy = get_policy(config.policy_id)
    registry = ToolRegistry(corpus)
    result = AgentRunResult()
    t_start = time.perf_counter()
    if isinstance(summarizer, LLMSummarizer):
        summarizer.bind_clock(t_start)

    def wall_ms() -> float:
        return (time.perf_counter() - t_start) * 1000.0

    meta = writer.metadata
    writer.append(
        "task_start",
        turn=0,
        t_wall_ms=0.0,
        payload={
            "note": json.dumps(
                {
                    "question": task.question,
                    "context_tier": task.context_tier,
                    "hops": task.hops,
                    "tool_schema_hash": tool_schema_hash(),
                }
            )
        },
    )

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    task_msg = {
        "role": "user",
        "content": (
            f"Question: {task.question}\n\n"
            "Use the tools to find and verify the answer across documents, "
            "then end with FINAL ANSWER: <answer>."
        ),
    }
    history: list[dict] = [system_msg, task_msg]
    chat_template_kwargs = (
        {"enable_thinking": True, "thinking_budget": config.thinking_budget_tokens}
        if config.thinking
        else {"enable_thinking": False}
    )

    answered = False
    answer_text: str | None = None
    turn = 0
    while turn < config.limits.max_turns:
        if wall_ms() >= config.limits.wall_timeout_ms:
            break
        writer.append("turn_start", turn=turn, t_wall_ms=wall_ms(), payload={})
        policy_result: PolicyResult = policy.run(history, config.policy, summarizer)
        for d in policy_result.decisions:
            writer.append(
                "policy_event",
                turn=turn,
                t_wall_ms=wall_ms(),
                payload=d.to_payload(config.policy_id),
            )
        writer.append(
            "llm_request",
            turn=turn,
            t_wall_ms=wall_ms(),
            payload={"messages": policy_result.view},
            messages_view_hash_=messages_view_hash(policy_result.view),
        )
        resp = backend.chat(
            policy_result.view,
            tools=TOOL_SCHEMAS,
            temperature=meta.temperature,
            seed=meta.seed,
            chat_template_kwargs=chat_template_kwargs,
        )
        writer.append(
            "llm_response",
            turn=turn,
            t_wall_ms=wall_ms(),
            payload={
                "usage": resp.usage,
                "timings_ms": resp.timings_ms,
                "completion_text": resp.completion_text,
                "think_text": resp.think_text,
                "tool_calls": [
                    {"name": c["name"], "args": c["arguments"], "call_id": c["id"]} for c in resp.tool_calls
                ],
            },
        )
        result.turns += 1
        result.gpu_seconds += resp.timings_ms.get("total", 0.0) / 1000.0
        result.prompt_tokens_total += resp.usage.get("prompt_tokens", 0)
        result.cached_tokens_total += resp.usage.get("cached_tokens", 0)

        assistant_msg = {
            "role": "assistant",
            "content": resp.visible_text,
            "thinking": resp.think_text,
        }
        if resp.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in resp.tool_calls
            ]
        history.append(assistant_msg)

        if not resp.tool_calls:
            answered = True
            answer_text = extract_final_answer(resp.visible_text)
            break

        for tc in resp.tool_calls:
            result.tool_calls += 1
            writer.append(
                "tool_call",
                turn=turn,
                t_wall_ms=wall_ms(),
                payload={"tool_call": {"name": tc["name"], "args": tc["arguments"], "call_id": tc["id"]}},
            )
            tr = registry.execute(tc["name"], tc["arguments"])
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": tr.output})
            writer.append(
                "tool_result",
                turn=turn,
                t_wall_ms=wall_ms(),
                payload={
                    "tool_call": {"name": tc["name"], "args": tc["arguments"], "call_id": tc["id"]},
                    "tool_result": tr.output if tr.ok else None,
                    "tool_error": None if tr.ok else tr.output,
                },
            )
        # Execute every tool call of a message before considering the limit, so
        # history stays API-legal (no assistant tool_calls without replies).
        if result.tool_calls >= config.limits.max_tool_calls:
            turn += 1
            break
        turn += 1

    abandoned, reason = check_abandonment(
        turns_used=result.turns,
        tool_calls_used=result.tool_calls,
        wall_ms_used=wall_ms(),
        answered=answered,
        limits=config.limits,
    )
    scores = score_answer(answer_text or "", task.gold_answer) if answered else {"em": 0.0, "f1": 0.0}
    result.answer = answer_text
    result.abandoned = abandoned
    result.abandonment_reason = reason
    result.em = scores["em"]
    result.f1 = scores["f1"]
    result.gpu_seconds += summarizer_gpu_seconds(summarizer)
    writer.append(
        "task_end",
        turn=turn,
        t_wall_ms=wall_ms(),
        payload={
            "answer": answer_text,
            "abandoned": abandoned,
            "abandonment_reason": reason,
            "em": result.em,
            "f1": result.f1,
            "note": json.dumps(
                {
                    "turns": result.turns,
                    "tool_calls": result.tool_calls,
                    "gpu_seconds": result.gpu_seconds,
                    "prompt_tokens_total": result.prompt_tokens_total,
                    "cached_tokens_total": result.cached_tokens_total,
                }
            ),
        },
    )
    return result


def summarizer_gpu_seconds(summarizer: Summarizer | None) -> float:
    if isinstance(summarizer, LLMSummarizer):
        # llm_response events for the summarizer already carry totals; here we
        # only add what the loop itself did not record: nothing (events carry it).
        return 0.0
    return 0.0
