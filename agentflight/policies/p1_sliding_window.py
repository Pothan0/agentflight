"""P1 sliding-window: system + first user (task) message + last N turns;
older turns replaced by a one-line note."""

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


class SlidingWindowPolicy(ContextPolicy):
    policy_id = "P1"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        groups = group_turns(history)
        if not groups:
            return PolicyResult(view=[], decisions=[])
        first_user = find_first_user_group(groups)
        keep_tail = config.window_turns
        kept_idxs = set(range(len(groups) - keep_tail, len(groups)))
        if first_user >= 0:
            kept_idxs.add(first_user)
        if groups[0][0]["role"] == "system":
            kept_idxs.add(0)

        decisions: list[PolicyDecision] = []
        view: list[dict] = []
        dropped_tokens = 0
        dropped_turns: list[int] = []
        for idx, group in enumerate(groups):
            if idx in kept_idxs:
                for m in group:
                    view.append(view_message(m, include_think=True))
            else:
                dropped_tokens += config.message_tokens(
                    [view_message(m, include_think=True) for m in group]
                )
                dropped_turns.append(idx)
        if dropped_turns:
            note = f"[{len(dropped_turns)} earlier turns were dropped by the sliding-window policy.]"
            # Insert the note right after the head (system/task) so recency order holds.
            head_len = 0
            if groups[0][0]["role"] == "system":
                head_len += 1
            if first_user >= 0:
                head_len += 1
            note_msg = {"role": "user", "content": note}
            view.insert(head_len, view_message(note_msg, include_think=False))
            note_tokens = config.tokens(note)
            decisions.append(
                PolicyDecision(
                    action="note_added",
                    detail=f"dropped turn groups {dropped_turns}",
                    tokens_before=dropped_tokens,
                    tokens_after=note_tokens,
                )
            )
        return PolicyResult(view=view, decisions=decisions)
