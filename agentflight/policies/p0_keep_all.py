"""P0 keep-everything: baseline. Full history, prior think blocks included,
nothing dropped."""

from __future__ import annotations

from agentflight.policies.base import (
    ContextPolicy,
    PolicyConfig,
    PolicyResult,
    Summarizer,
    view_message,
)


class KeepEverythingPolicy(ContextPolicy):
    policy_id = "P0"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        view = [view_message(m, include_think=True) for m in history]
        return PolicyResult(view=view, decisions=[])
