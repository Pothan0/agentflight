"""G1 tests: replay ordering, scheduler with a fake client, server config."""

from agentflight.replay.scheduler import ReplayRequest, order_requests, run_replay
from agentflight.replay.server import ServerConfig


def reqs() -> list[ReplayRequest]:
    out = []
    for sid in ("s1", "s2"):
        for seq in range(3):
            out.append(ReplayRequest(session_id=sid, seq_in_session=seq, messages=[{"role": "user", "content": f"{sid}-{seq}"}]))
    return out


def test_affinity_groups_sessions_contiguously():
    ordered = order_requests(reqs(), "affinity")
    assert [r.session_id for r in ordered] == ["s1"] * 3 + ["s2"] * 3
    assert [r.seq_in_session for r in ordered] == [0, 1, 2, 0, 1, 2]


def test_round_robin_interleaves():
    ordered = order_requests(reqs(), "round_robin")
    assert [r.session_id for r in ordered] == ["s1", "s2"] * 3
    # same multiset of requests, only the order differs
    assert sorted((r.session_id, r.seq_in_session) for r in ordered) == sorted(
        (r.session_id, r.seq_in_session) for r in reqs()
    )


def test_ordering_unknown_mode_rejected():
    try:
        order_requests(reqs(), "random")
        raise AssertionError("unknown mode must raise")
    except ValueError:
        pass


class FakeClient:
    def __init__(self):
        self.log: list[tuple[str, int]] = []

    def replay_one(self, messages, session_id, seq_in_session, condition, **kw):
        self.log.append((session_id, seq_in_session))
        from agentflight.replay.client import ReplayRecord

        return ReplayRecord(
            session_id=session_id, seq_in_session=seq_in_session, condition=condition,
            ttft_ms=10.0, total_ms=20.0, prompt_tokens=100, completion_tokens=1, cached_tokens=50,
        )


def test_run_replay_preserves_order_and_records():
    client = FakeClient()
    records = run_replay(client, reqs(), "affinity", concurrency=1, condition="unit")
    assert len(records) == 6
    assert client.log[:3] == [("s1", 0), ("s1", 1), ("s1", 2)]  # submission order == affinity order
    assert all(r.condition == "unit" for r in records)


def test_server_config_flags():
    on = ServerConfig(enable_prefix_caching=True)
    off = ServerConfig(enable_prefix_caching=False)
    assert "--enable-prefix-caching" in on.argv()
    assert "--no-enable-prefix-caching" in off.argv()
    lm = ServerConfig(lmcache_offload=True)
    argv = lm.argv()
    assert "--kv-transfer-config" in argv
    assert "LMCacheConnectorV1" in argv[argv.index("--kv-transfer-config") + 1]
    env = lm.env(base_env={})
    assert env["LMCACHE_USE_EXPERIMENTAL"] == "True"
    base = ServerConfig()
    assert "LMCACHE_USE_EXPERIMENTAL" not in base.env(base_env={})
    assert base.base_url().endswith("/v1")
