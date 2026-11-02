"""Session scheduler for Mode-R (build-prompt section 2, C3).

The serving variable E2 isolates is INTERLEAVING ORDER under concurrency:
- ``affinity``: all requests of one session (task) are emitted contiguously,
  so the per-session prefix cache stays warm.
- ``round_robin``: requests rotate across sessions turn-by-turn, so the
  cache is repeatedly interleaved/evicted.

Both orderings emit the same multiset of requests; only order differs.
Execution uses a worker pool of size ``concurrency`` (1/2/4 per the matrix).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agentflight.replay.client import ReplayClient, ReplayRecord


@dataclass
class ReplayRequest:
    session_id: str
    seq_in_session: int
    messages: list[dict]


def order_requests(requests: list[ReplayRequest], mode: str) -> list[ReplayRequest]:
    """Return the emission order. Deterministic: stable within session id and
    seq (input lists must already be grouped/sorted by (session, seq))."""
    by_session: dict[str, list[ReplayRequest]] = {}
    for r in sorted(requests, key=lambda r: (r.session_id, r.seq_in_session)):
        by_session.setdefault(r.session_id, []).append(r)
    if mode == "affinity":
        out: list[ReplayRequest] = []
        for sid in sorted(by_session):
            out.extend(by_session[sid])
        return out
    if mode == "round_robin":
        queues = [by_session[sid] for sid in sorted(by_session)]
        out = []
        i = 0
        while any(queues):
            q = queues[i % len(queues)]
            if q:
                out.append(q.pop(0))
            i += 1
        return out
    raise ValueError(f"unknown scheduling mode {mode!r}")


def run_replay(
    client: ReplayClient,
    requests: list[ReplayRequest],
    mode: str,
    concurrency: int,
    condition: str,
) -> list[ReplayRecord]:
    ordered = order_requests(requests, mode)
    records: list[ReplayRecord] = [None] * len(ordered)  # type: ignore[list-item]

    def work(idx: int, req: ReplayRequest) -> None:
        records[idx] = client.replay_one(
            req.messages,
            session_id=req.session_id,
            seq_in_session=req.seq_in_session,
            condition=condition,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, i, r) for i, r in enumerate(ordered)]
        for f in futures:
            f.result()
    return records
