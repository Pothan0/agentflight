"""P6 re-compaction cascade: like P3 (chunked fresh compaction) but when the
total view still exceeds budget B, the compressed chunks THEMSELVES get
re-compressed ("JPEG of a JPEG"). Tests Baseten's 4–16pp re-compaction
penalty on an open model. Cascades until under budget or a single merged
summary remains."""

from __future__ import annotations

from agentflight.policies.base import (
    ContextPolicy,
    PolicyConfig,
    PolicyDecision,
    PolicyResult,
    Summarizer,
)
from agentflight.policies.p3_chunked import ChunkedCompactionPolicy


class ReCompactionCascadePolicy(ContextPolicy):
    policy_id = "P6"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        if summarizer is None:
            raise ValueError("P6 requires a summarizer (LLM-backed in real runs)")
        # Stage 1: exactly P3.
        p3 = ChunkedCompactionPolicy().build_view(history, config, summarizer)
        view = list(p3.view)
        decisions = list(p3.decisions)
        if config.message_tokens(view) <= config.budget_tokens:
            return PolicyResult(view=view, decisions=decisions)

        # Stage 2: cascade. Summary blocks live in view as user messages that
        # start with "[Chunk". Merge ALL of them into one re-compressed block,
        # then re-check; repeat until under budget or one block left.
        cascade_round = 0
        while True:
            cascade_round += 1
            summary_msgs = [
                (i, m)
                for i, m in enumerate(view)
                if m["role"] == "user" and (m["content"].startswith("[Chunk") or m["content"].startswith("[Re-compressed"))
            ]
            if len(summary_msgs) <= 1:
                break
            before = config.message_tokens([m for _, m in summary_msgs])
            merged = summarizer.summarize([m["content"] for _, m in summary_msgs])
            block = f"[Re-compressed round {cascade_round}: {len(summary_msgs)} summaries merged.]\n{merged}"
            keep_head = [m for m in view[: summary_msgs[0][0]]]
            keep_tail = [m for m in view[summary_msgs[-1][0] + 1 :]]
            new_view = keep_head + [{"role": "user", "content": block}] + keep_tail
            decisions.append(
                PolicyDecision(
                    action="recompressed",
                    detail=f"cascade round {cascade_round}: merged {len(summary_msgs)} summary blocks",
                    tokens_before=before,
                    tokens_after=config.tokens(block),
                )
            )
            view = new_view
            if config.message_tokens(view) <= config.budget_tokens:
                break
        return PolicyResult(view=view, decisions=decisions)
