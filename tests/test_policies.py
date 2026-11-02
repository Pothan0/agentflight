"""G1 tests: policy purity (same inputs -> byte-identical view), correctness
of each policy's contract, input immutability, and API-legality of views."""

import copy
import json

import pytest

from agentflight.harness.loop import NullSummarizer
from agentflight.policies import POLICIES, POLICY_RUN_ORDER, get_policy
from agentflight.policies.base import PolicyConfig, group_turns


def make_history(n_turns: int = 8, long_tool_result: bool = True) -> list[dict]:
    """A canonical multi-turn history: system + task + N agent turns with
    tool calls, thinking, and long tool results."""
    h: list[dict] = [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Question: who directed the film X and when was it released?"},
    ]
    for i in range(n_turns):
        filler = " ".join(f"detail{i}_{j}" for j in range(30))
        result_text = (
            f"Passage about entity{i}: it was founded in 190{i} by person{i}. " + filler * (6 if long_tool_result else 1)
        )
        h.append(
            {
                "role": "assistant",
                "content": f"Let me look up entity{i}.",
                "thinking": f"I need to find entity{i}. Plan: search then read. " + filler,
                "tool_calls": [{"id": f"c{i}", "name": "read", "arguments": {"doc_id": f"doc{i}"}}],
            }
        )
        h.append({"role": "tool", "tool_call_id": f"c{i}", "content": result_text})
    h.append(
        {
            "role": "assistant",
            "content": "The film X was directed by Alice.\nFINAL ANSWER: Alice",
            "thinking": "Combining facts now.",
        }
    )
    return h


def cfg(policy_id: str, **kw) -> PolicyConfig:
    defaults = dict(policy_id=policy_id, window_turns=3, budget_tokens=600, chunk_tokens=250,
                    keep_recent_chunks=1, tool_stub_age_turns=1)
    defaults.update(kw)
    return PolicyConfig(**defaults)


def assert_api_legal(view: list[dict]) -> None:
    """Every tool message must answer a preceding assistant tool_call (same
    id); every assistant tool_call must be followed by its tool replies."""
    open_calls: set[str] = set()
    for m in view:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                open_calls.add(tc["id"])
        elif m["role"] == "tool":
            assert m["tool_call_id"] in open_calls, f"orphan tool reply {m['tool_call_id']}"
    assert not open_calls or all(
        any(mm["role"] == "tool" and mm.get("tool_call_id") == tc_id for mm in view)
        for tc_id in _last_assistant_calls(view)
    ), "assistant tool_calls left unanswered at the end of the view"


def _last_assistant_calls(view):
    for m in reversed(view):
        if m["role"] == "assistant" and m.get("tool_calls"):
            return [tc["id"] for tc in m["tool_calls"]]
    return []


@pytest.mark.parametrize("policy_id", list(POLICIES))
def test_policy_purity(policy_id):
    """Same history + config -> byte-identical view. Input NEVER mutated."""
    policy = get_policy(policy_id)
    history = make_history()
    snapshot = copy.deepcopy(history)
    r1 = policy.run(history, cfg(policy_id), NullSummarizer())
    r2 = policy.run(history, cfg(policy_id), NullSummarizer())
    assert json.dumps(r1.view, sort_keys=True) == json.dumps(r2.view, sort_keys=True)
    assert history == snapshot  # deep equality: policy did not touch input
    assert len(r1.decisions) == len(r2.decisions)


@pytest.mark.parametrize("policy_id", list(POLICIES))
def test_policy_views_are_api_legal(policy_id):
    view = get_policy(policy_id).run(make_history(), cfg(policy_id), NullSummarizer()).view
    assert view, "empty view"
    assert view[0]["role"] == "system"
    assert_api_legal(view)
    # wire-format sanity: no local-only keys leak into the view
    for m in view:
        assert "thinking" not in m


def test_run_order_priority():
    # Cheap core policies first; publishable prefix on quota death (section 8).
    assert POLICY_RUN_ORDER[0] == "P0"
    assert set(POLICY_RUN_ORDER) == set(POLICIES)


def test_p0_keeps_everything_including_think():
    view = get_policy("P0").run(make_history(2), cfg("P0"), None).view
    assert any("<think>" in (m.get("content") or "") for m in view)


def test_p1_sliding_window():
    view = get_policy("P1").run(make_history(8), cfg("P1"), None).view
    # system + task + note + last 3 turn-groups + final assistant
    assert view[0]["role"] == "system"
    assert "Question:" in view[1]["content"]
    assert any("sliding-window policy" in (m.get("content") or "") for m in view)
    assert_api_legal(view)
    # recent verbatim: the last tool result survives
    assert any("entity7" in (m.get("content") or "") for m in view)
    # old verbatim content is gone
    assert not any("detail0_" in (m.get("content") or "") for m in view)


def test_p2_passthrough_under_budget():
    history = make_history(2)
    r = get_policy("P2").run(history, cfg("P2", budget_tokens=10**9), None)
    assert not r.decisions
    assert any("<think>" in (m.get("content") or "") for m in r.view)


def test_p2_summarizes_over_budget():
    history = make_history(8)
    r = get_policy("P2").run(history, cfg("P2", budget_tokens=800), NullSummarizer())
    assert any(d.action == "summarized" for d in r.decisions)
    assert any("Compacted earlier history" in (m.get("content") or "") for m in r.view)
    assert r.decisions[0].tokens_after < r.decisions[0].tokens_before
    # question stays verbatim
    assert any(m.get("content", "").startswith("Question:") for m in r.view)


def test_p3_chunked_compaction():
    history = make_history(8)
    r = get_policy("P3").run(history, cfg("P3"), NullSummarizer())
    assert r.decisions and all(d.action == "summarized" for d in r.decisions)
    # recent chunk stays verbatim (contains the last turn's raw tool text)
    assert any("detail7_" in (m.get("content") or "") for m in r.view)
    # aged chunks are compressed exactly once each (no cascade in P3)
    n_chunk_msgs = sum(1 for m in r.view if m.get("content", "").startswith("[Chunk"))
    assert n_chunk_msgs == len(r.decisions)
    assert all(d.action != "recompressed" for d in r.decisions)
    assert all(d.tokens_after < d.tokens_before for d in r.decisions)


def test_p4_think_discard():
    history = make_history(8)
    r = get_policy("P4").run(history, cfg("P4"), None)
    assert not any("<think>" in (m.get("content") or "") for m in r.view)
    # everything else is kept verbatim: raw tool results still present
    assert any("detail0_" in (m.get("content") or "") for m in r.view)
    d = r.decisions[0]
    assert d.action == "think_dropped" and d.tokens_before > 0 and d.tokens_after == 0


def test_p5_tool_elision():
    history = make_history(6)
    r = get_policy("P5").run(history, cfg("P5"), None)
    stubs = [m for m in r.view if m["role"] == "tool" and "elided" in (m.get("content") or "")]
    assert stubs, "expected stubbed old tool results"
    # recent tool results (age <= 1) stay verbatim
    recents = [m for m in r.view if m["role"] == "tool" and "detail5_" in (m.get("content") or "")]
    assert recents and not any("elided" in m["content"] for m in recents)
    assert all(d.action == "stubbed" for d in r.decisions)
    assert all(d.tokens_after < d.tokens_before for d in r.decisions)


def test_p6_cascade_recompresses():
    history = make_history(10)
    # budget sits BELOW the P3 view size (~564 tok with these fixtures) so the
    # cascade must fire and merge chunk summaries
    r = get_policy("P6").run(history, cfg("P6", budget_tokens=550, chunk_tokens=200), NullSummarizer())
    actions = [d.action for d in r.decisions]
    assert "summarized" in actions
    assert "recompressed" in actions, f"expected cascade under budget, got {actions}"
    assert any(m.get("content", "").startswith("[Re-compressed") for m in r.view)


def test_group_turns_keeps_tool_replies_with_their_call():
    h = make_history(3)
    groups = group_turns(h)
    # system, task, then 3 x [assistant+tool], then final assistant
    assert len(groups) == 2 + 3 + 1
    for g in groups[2:5]:
        assert g[0]["role"] == "assistant" and g[1]["role"] == "tool"
        assert g[1]["tool_call_id"] == g[0]["tool_calls"][0]["id"]


def test_llm_policies_require_summarizer():
    for pid in ("P2", "P3", "P6"):
        with pytest.raises(ValueError, match="summarizer"):
            get_policy(pid).run(make_history(8), cfg(pid, budget_tokens=10), None)
