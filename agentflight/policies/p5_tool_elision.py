"""P5 tool-result-elision: replace tool results older than K turns with a
stub (first 200 + last 200 chars), then rehydrate a BM25-retrieved excerpt
from the result's OWN original text when the current query matches. CPU-side
retrieval only (rank_bm25) — no external service."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

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

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text) if s and s.strip()]


def window_text(sents: list[str], start: int, size: int) -> str:
    return " ".join(sents[start : start + size])


_TOKEN_RE = re.compile(r"\w+")


def _bm25_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def current_query(task_text: str, history: list[dict], max_chars: int = 2000) -> str:
    """The 'current turn's query': the task statement plus the most recent
    assistant activity (what the agent is chasing right now)."""
    parts: list[str] = [task_text]
    for m in reversed(history):
        if m["role"] == "assistant" and (m.get("content") or "").strip():
            parts.append(m.get("content") or "")
            break
    return " ".join(parts)[:max_chars]


class ToolResultElisionPolicy(ContextPolicy):
    policy_id = "P5"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        groups = group_turns(history)
        task_idx = find_first_user_group(groups)
        task_text = groups[task_idx][0].get("content") or "" if task_idx >= 0 else ""
        query = current_query(task_text, history)
        n_groups = len(groups)
        view: list[dict] = []
        decisions: list[PolicyDecision] = []
        stubbed = 0
        rehydrated = 0
        for gidx, group in enumerate(groups):
            age = n_groups - 1 - gidx  # 0 = most recent turn
            for m in group:
                if m["role"] == "tool" and age > config.tool_stub_age_turns:
                    original = m.get("content") or ""
                    head, tail = config.tool_stub_head_chars, config.tool_stub_tail_chars
                    keep_margin = head + tail + 40  # don't stub tiny results
                    if len(original) <= keep_margin:
                        view.append(view_message(m, include_think=True))
                        continue
                    stub = f"{original[:head]}\n[... elided by tool-result elision ...]\n{original[-tail:]}"
                    excerpt = self._rehydrate(original, query, config)
                    if excerpt:
                        stub += f"\n[rehydrated excerpt matching current query]\n{excerpt}"
                        rehydrated += 1
                    view.append(
                        {"role": "tool", "tool_call_id": m["tool_call_id"], "content": stub}
                    )
                    stubbed += 1
                    decisions.append(
                        PolicyDecision(
                            action="stubbed",
                            detail=f"tool result in turn -{age} stubbed"
                            + (" + rehydrated" if excerpt else ""),
                            tokens_before=config.tokens(original),
                            tokens_after=config.tokens(stub),
                        )
                    )
                else:
                    view.append(view_message(m, include_think=True))
        return PolicyResult(view=view, decisions=decisions)

    def _rehydrate(self, original: str, query: str, config: PolicyConfig) -> str:
        sents = sentences(original)
        if not sents or not query.strip():
            return ""
        corpus = [window_text(sents, i, 3) for i in range(len(sents))]
        bm25 = BM25Okapi([_bm25_tokens(d) for d in corpus])
        scores = bm25.get_scores(_bm25_tokens(query))
        best = max(range(len(corpus)), key=lambda i: scores[i])
        if scores[best] <= 0:
            return ""
        return corpus[best]
