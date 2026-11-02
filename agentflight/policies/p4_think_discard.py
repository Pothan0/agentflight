"""P4 think-discard: drop <think> content from all PRIOR turns (the current
turn's thinking is generated fresh by the server and never part of the view).
This is the harness-level analog of NVIDIA Dynamo's selective retention of
discarded reasoning KV."""

from __future__ import annotations

from agentflight.policies.base import (
    ContextPolicy,
    PolicyConfig,
    PolicyDecision,
    PolicyResult,
    Summarizer,
    view_message,
)


class ThinkDiscardPolicy(ContextPolicy):
    policy_id = "P4"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        view: list[dict] = []
        think_tokens = 0
        turns_affected = 0
        for m in history:
            if m["role"] == "assistant" and m.get("thinking"):
                think_tokens += config.tokens(m["thinking"])
                turns_affected += 1
            view.append(view_message(m, include_think=False))
        decisions: list[PolicyDecision] = []
        if think_tokens:
            decisions.append(
                PolicyDecision(
                    action="think_dropped",
                    detail=f"dropped think blocks from {turns_affected} prior assistant turns",
                    tokens_before=think_tokens,
                    tokens_after=0,
                )
            )
        return PolicyResult(view=view, decisions=decisions)
