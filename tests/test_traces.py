"""G1 tests: trace schema, append-only writer/reader round-trip, validation."""

import gzip
import json

import pytest

from agentflight.traces.schema import RunMetadata, messages_view_hash
from agentflight.traces.writer import TraceWriter, read_trace, validate_event, validate_trace_file

META = RunMetadata(
    model_id="mock-model",
    tool_schema_hash="deadbeef",
    task_id="t0",
    policy_id="P0",
    context_tier="4k",
)


def _write_full_trace(tmp_path):
    w = TraceWriter(tmp_path / "tr.json.gz", META)
    view = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    w.append("task_start", turn=0, t_wall_ms=0.0, payload={"note": "start"})
    w.append("turn_start", turn=0, t_wall_ms=1.0, payload={})
    w.append(
        "llm_request",
        turn=0,
        t_wall_ms=2.0,
        payload={"messages": view},
        messages_view_hash_=messages_view_hash(view),
    )
    w.append(
        "llm_response",
        turn=0,
        t_wall_ms=50.0,
        payload={
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0},
            "timings_ms": {"ttft": 10.0, "total": 40.0, "inter_token": [5.0]},
            "completion_text": "ok",
            "think_text": "hmm",
        },
    )
    w.append("task_end", turn=0, t_wall_ms=60.0, payload={"abandoned": False, "em": 0.0, "f1": 0.0})
    return w, tmp_path / "tr.json.gz"


def test_round_trip_byte_exact(tmp_path):
    _, path = _write_full_trace(tmp_path)
    meta, events = read_trace(path)
    assert meta == META
    assert [e.seq for e in events] == [0, 1, 2, 3, 4]
    assert [e.kind for e in events] == ["task_start", "turn_start", "llm_request", "llm_response", "task_end"]
    assert events[2].payload.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert events[2].messages_view_hash == messages_view_hash(events[2].payload.messages)
    # logical byte-exactness: re-serialize == file contents
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines[0]["record"] == "metadata"
    for line, ev in zip(lines[1:], events, strict=False):
        assert json.loads(ev.model_dump_json()) == line


def test_append_only_header_written_once(tmp_path):
    w, path = _write_full_trace(tmp_path)
    w.append("task_start", turn=1, t_wall_ms=100.0, payload={"note": "again"})
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    assert sum(1 for line in lines if '"record": "metadata"' in line or '"record":"metadata"' in line) == 1
    assert len(lines) == 7  # header + 6 events


def test_validate_rejects_view_hash_mismatch():
    from agentflight.traces.schema import TraceEvent

    ev = TraceEvent(
        seq=0,
        turn=0,
        t_wall_ms=0.0,
        kind="llm_request",
        messages_view_hash="wrong",
        payload={"messages": [{"role": "user", "content": "x"}]},
    )
    with pytest.raises(ValueError, match="messages_view_hash"):
        validate_event(ev)


def test_validate_requires_kind_specific_keys():
    from agentflight.traces.schema import TraceEvent

    bad = TraceEvent(seq=0, turn=0, t_wall_ms=0.0, kind="llm_response", payload={})
    with pytest.raises(ValueError, match="missing required payload keys"):
        validate_event(bad)
    bad2 = TraceEvent(
        seq=0, turn=0, t_wall_ms=0.0, kind="task_end", payload={"abandoned": True}
    )
    with pytest.raises(ValueError, match="abandonment_reason"):
        validate_event(bad2)


def test_validate_trace_file_flags_truncated_run(tmp_path):
    w = TraceWriter(tmp_path / "t2.json.gz", META)
    w.append("task_start", turn=0, t_wall_ms=0.0, payload={"note": "start"})
    with pytest.raises(ValueError, match="task_end"):
        validate_trace_file(tmp_path / "t2.json.gz")


def test_valid_file_passes_g3(tmp_path):
    _, path = _write_full_trace(tmp_path)
    info = validate_trace_file(path)
    assert info["events"] == 5
    assert info["task_id"] == "t0"
