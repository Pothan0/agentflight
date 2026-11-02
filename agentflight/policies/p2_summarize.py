"""P2 summarize-compaction: when the full history exceeds budget B tokens,
LLM-summarize the oldest 50% of turns into one summary block (temperature 0
on the served model in real runs). Classic compaction; recomputed from the
full history every turn (the policy is pure, no state)."""

from __future__ import annotations

from agentflight.policies.base import (
    ContextPolicy,
    PolicyConfig,
    PolicyDecision,
    PolicyResult,
    Summarizer,
    find_first_user_group,
    group_turns,
    view_message,
)

SUMMARY_HEADER = "[Compacted earlier history — summary follows.]"


class SummarizeCompactionPolicy(ContextPolicy):
    policy_id = "P2"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        groups = group_turns(history)
        if not groups:
            return PolicyResult(view=[], decisions=[])
        if config.message_tokens(history) <= config.budget_tokens:
            return PolicyResult(view=[view_message(m, include_think=True) for m in history], decisions=[])
        if summarizer is None:
            raise ValueError("P2 requires a summarizer (LLM-backed in real runs)")

        system_idx = 0 if groups and groups[0][0]["role"] == "system" else -1
        task_idx = find_first_user_group(groups)
        # Summarizable groups: everything except system and the task statement.
        pool = [i for i in range(len(groups)) if i not in (system_idx, task_idx)]
        n_summary = int(len(pool) * config.summarize_fraction)
        to_summarize = pool[:n_summary]
        to_keep = pool[n_summary:]

        texts: list[str] = []
        summarized_tokens = 0
        for idx in to_summarize:
            for m in groups[idx]:
                if m["role"] in ("user", "assistant", "tool"):
                    texts.append(f"{m['role']}: {m.get('content') or ''}")
            summarized_tokens += config.message_tokens(
                [view_message(m, include_think=True) for m in groups[idx]]
            )
        summary = summarizer.summarize(texts) if texts else ""
        summary_block = f"{SUMMARY_HEADER}\n{summary}"
        summary_tokens = config.tokens(summary_block)

        view: list[dict] = []
        if system_idx >= 0:
            view.append(view_message(groups[system_idx][0], include_think=True))
        if task_idx >= 0:
            view.append(view_message(groups[task_idx][0], include_think=True))
        if summary:
            view.append(view_message({"role": "user", "content": summary_block}, include_think=False))
        for idx in to_keep:
            for m in groups[idx]:
                view.append(view_message(m, include_think=True))

        decisions = [
            PolicyDecision(
                action="summarized",
                detail=f"summarized {len(to_summarize)} turn groups into one block ({len(texts)} messages)",
                tokens_before=summarized_tokens,
                tokens_after=summary_tokens,
            )
        ]
        return PolicyResult(view=view, decisions=decisions)
