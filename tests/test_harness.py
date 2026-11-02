"""G1 tests: tool registry, parsers, MockBackend, and the full agent loop."""


import pytest

from agentflight.harness.llm import MockBackend
from agentflight.harness.loop import HarnessConfig, NullSummarizer, run_task
from agentflight.harness.parsers import parse_json_fallback, split_think
from agentflight.harness.tools import ToolRegistry, safe_calc, tool_schema_hash
from agentflight.policies.base import PolicyConfig
from agentflight.scoring.abandonment import RunLimits
from agentflight.tasks.schema import Passage, QATask
from agentflight.traces.schema import RunMetadata
from agentflight.traces.writer import TraceWriter, read_trace, validate_trace_file

CORPUS = [
    Passage(doc_id="d-alpha", title="Alpha", text="Alpha Corp was founded in 1901 by Alice Smith in Berlin.",
            source="test", origin="support"),
    Passage(doc_id="d-beta", title="Beta", text="Beta Ltd makes widgets. The director of Beta is Bob Jones.",
            source="test", origin="support"),
    Passage(doc_id="d-gamma", title="Gamma", text="Gamma GmbH released its film in 1994, directed by Carol Wei.",
            source="test", origin="distractor_pool"),
]


def make_task() -> QATask:
    return QATask(
        task_id="t-test", source="test", hops=2,
        question="Who founded Alpha Corp?", gold_answer="Alice Smith", answer_type="entity",
        support_doc_ids=["d-alpha"], context_tier="4k", tier_target_tokens=4000, tier_total_tokens=4000,
    )


def make_writer(tmp_path, policy_id="P0") -> TraceWriter:
    return TraceWriter(
        tmp_path / f"trace-{policy_id}.json.gz",
        RunMetadata(
            model_id="mock-model", tool_schema_hash=tool_schema_hash(),
            task_id="t-test", policy_id=policy_id, context_tier="4k",
        ),
    )


# --- tools ---

def test_search_ranks_and_returns_doc_ids():
    reg = ToolRegistry(CORPUS)
    out = reg.execute("search", {"query": "founded Alpha"})
    assert out.ok
    assert "d-alpha" in out.output and "doc_id=" in out.output


def test_read_and_errors():
    reg = ToolRegistry(CORPUS)
    assert "Alice Smith" in reg.execute("read", {"doc_id": "d-alpha"}).output
    err = reg.execute("read", {"doc_id": "nope"})
    assert not err.ok and "unknown doc_id" in err.output
    miss = reg.execute("search", {})
    assert not miss.ok and "missing argument" in miss.output


def test_calculator_whitelist():
    assert safe_calc("6*7") == 42.0
    assert safe_calc("(3+4)*2.5") == 17.5
    with pytest.raises(ValueError):
        safe_calc("__import__('os')")
    with pytest.raises((ValueError, SyntaxError)):
        safe_calc("x + 1")
    assert safe_calc("2**10") == 1024.0


# --- parsers ---

def test_split_think_variants():
    think, vis = split_think("<think>\nplan\n</think>\nAnswer: 42")
    assert think == "plan" and vis == "Answer: 42"
    think2, vis2 = split_think("<think>unterminated")
    assert think2 == "unterminated" and vis2 == ""
    think3, vis3 = split_think("no think here")
    assert think3 == "" and vis3 == "no think here"


def test_json_fallback_tool_call():
    text = 'Hmm. <tool_call>{"name": "search", "arguments": {"query": "alpha corp"}}</tool_call>'
    calls, visible = parse_json_fallback(text)
    assert calls == [{"name": "search", "arguments": {"query": "alpha corp"}}]
    assert "tool_call" not in visible


# --- agent loop end-to-end on the mock ---

SCRIPT = [
    {"think": "search first", "tool_calls": [{"name": "search", "arguments": {"query": "Alpha Corp founded"}}]},
    {"think": "read it", "tool_calls": [{"name": "read", "arguments": {"doc_id": "d-alpha"}}]},
    {"content": "I have it.\nFINAL ANSWER: Alice Smith"},
]


@pytest.mark.parametrize("policy_id", ["P0", "P4", "P5"])
def test_full_loop_with_mock_backend(policy_id, tmp_path):
    backend = MockBackend(script=SCRIPT)
    writer = make_writer(tmp_path, policy_id)
    cfg = HarnessConfig(policy_id=policy_id, policy=PolicyConfig(policy_id=policy_id))
    result = run_task(make_task(), CORPUS, backend, writer, cfg, summarizer=None)
    assert result.answer == "Alice Smith"
    assert result.em == 1.0 and result.f1 == 1.0
    assert not result.abandoned
    assert result.turns == 3 and result.tool_calls == 2
    # trace validates (G3-style check) and the view sent is byte-recorded
    info = validate_trace_file(writer.path)
    assert info["kinds"]["llm_request"] == 3
    meta, events = read_trace(writer.path)
    reqs = [e for e in events if e.kind == "llm_request"]
    assert reqs[0].payload.messages[0]["role"] == "system"
    # P0/P5 keep prior think blocks; P4 drops them
    has_think = any("<think>" in (m.get("content") or "") for m in reqs[2].payload.messages)
    assert has_think == (policy_id in ("P0", "P5"))
    if policy_id == "P4":
        pev = [e for e in events if e.kind == "policy_event"]
        assert pev and pev[0].payload.decision == "think_dropped"


def test_loop_abandons_on_turn_limit(tmp_path):
    backend = MockBackend(script=[{"tool_calls": [{"name": "search", "arguments": {"query": "x"}}]}])
    writer = make_writer(tmp_path)
    cfg = HarnessConfig(policy_id="P0", limits=RunLimits(max_turns=2, max_tool_calls=99))
    result = run_task(make_task(), CORPUS, backend, writer, cfg)
    assert result.abandoned and result.abandonment_reason == "max_turns"
    assert result.em == 0.0
    validate_trace_file(writer.path)  # abandoned traces must still validate


def test_summarizer_policies_run_with_null_summarizer(tmp_path):
    backend = MockBackend(script=SCRIPT)
    writer = make_writer(tmp_path, "P3")
    cfg = HarnessConfig(policy_id="P3", policy=PolicyConfig(policy_id="P3", budget_tokens=300, chunk_tokens=150))
    result = run_task(make_task(), CORPUS, backend, writer, cfg, summarizer=NullSummarizer())
    assert result.turns >= 1
    validate_trace_file(writer.path)
