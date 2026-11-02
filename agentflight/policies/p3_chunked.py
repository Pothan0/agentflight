"""P3 chunked-compaction (Baseten-style): compress history in fixed ~2k-token
chunks AS THEY AGE, keeping the most recent chunks verbatim. A compressed
chunk produces a FRESH summary that is never re-compressed by this policy
(that cascade is P6). Tests Baseten's 90.7%-vs-81.5% denoising finding on an
open model.

System + first user (task) message are never chunked: the question itself
must stay verbatim; the distractor noise lives in the aging tool traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Chunk:
    group_idxs: list[int]
    tokens: int


def make_chunks(groups: list[list[dict]], config: PolicyConfig) -> list[Chunk]:
    """Pack consecutive turn groups into ~chunk_tokens-sized chunks (a group
    always lands wholly in one chunk so views stay API-legal)."""
    chunks: list[Chunk] = []
    cur: list[int] = []
    cur_tokens = 0
    for idx, group in enumerate(groups):
        t = config.message_tokens([view_message(m, include_think=True) for m in group])
        if cur and cur_tokens + t > config.chunk_tokens:
            chunks.append(Chunk(group_idxs=cur, tokens=cur_tokens))
            cur, cur_tokens = [], 0
        cur.append(idx)
        cur_tokens += t
    if cur:
        chunks.append(Chunk(group_idxs=cur, tokens=cur_tokens))
    return chunks


def groups_text(groups: list[list[dict]]) -> list[str]:
    texts = []
    for group in groups:
        for m in group:
            if m["role"] in ("user", "assistant", "tool"):
                texts.append(f"{m['role']}: {m.get('content') or ''}")
    return texts


class ChunkedCompactionPolicy(ContextPolicy):
    policy_id = "P3"

    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult:
        groups = group_turns(history)
        if not groups:
            return PolicyResult(view=[], decisions=[])
        system_idx = 0 if groups and groups[0][0]["role"] == "system" else -1
        task_idx = find_first_user_group(groups)
        pool = [i for i in range(len(groups)) if i not in (system_idx, task_idx)]
        if len(pool) == 0:
            return PolicyResult(view=[view_message(m, include_think=True) for m in history], decisions=[])
        if summarizer is None:
            raise ValueError("P3 requires a summarizer (LLM-backed in real runs)")

        pool_groups = [groups[i] for i in pool]
        chunks = make_chunks(pool_groups, config)
        n_verbatim = min(config.keep_recent_chunks, len(chunks))
        verbatim_chunks = chunks[len(chunks) - n_verbatim :]
        compressed_chunks = chunks[: len(chunks) - n_verbatim]

        decisions: list[PolicyDecision] = []
        view: list[dict] = []
        if system_idx >= 0:
            view.append(view_message(groups[system_idx][0], include_think=True))
        if task_idx >= 0:
            view.append(view_message(groups[task_idx][0], include_think=True))

        verbatim_idxs = {i for c in verbatim_chunks for i in c.group_idxs}
        for ci, chunk in enumerate(compressed_chunks):
            cgroups = [pool_groups[i] for i in chunk.group_idxs]
            texts = groups_text(cgroups)
            before = config.message_tokens([view_message(m, include_think=True) for g in cgroups for m in g])
            summary = summarizer.summarize(texts)
            block = f"[Chunk {ci + 1} summary: {len(cgroups)} turns compressed.]\n{summary}"
            view.append(view_message({"role": "user", "content": block}, include_think=False))
            decisions.append(
                PolicyDecision(
                    action="summarized",
                    detail=f"chunk {ci + 1}/{len(chunks)} ({len(cgroups)} turns) compressed once, never re-compressed",
                    tokens_before=before,
                    tokens_after=config.tokens(block),
                )
            )
        for i in verbatim_idxs:
            for m in pool_groups[i]:
                view.append(view_message(m, include_think=True))

        return PolicyResult(view=view, decisions=decisions)
