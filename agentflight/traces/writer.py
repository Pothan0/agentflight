"""Append-only trace writer / reader (gzipped JSONL).

Rules enforced here:
- APPEND-ONLY: append() never rewrites the file. Checkpointing works because
  each task's rows are self-contained.
- Per-event validation with per-kind required fields (writer is stricter than
  the pydantic schema so malformed events fail at write time, not review time).
- Round-trip is byte-exact for the logical content: read(write(events)) ==
  events (verified by tests).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from agentflight.traces.schema import (
    EVENT_KIND_LIST,
    RunMetadata,
    TraceEvent,
    messages_view_hash,
)

# Required payload keys per event kind (minimum contract of section 4).
_REQUIRED: dict[str, tuple[str, ...]] = {
    "task_start": ("note",),
    "turn_start": (),
    "policy_event": ("policy_id", "decision"),
    "llm_request": ("messages",),
    "llm_response": ("usage", "timings_ms"),
    "tool_call": ("tool_call",),
    "tool_result": ("tool_call",),
    "task_end": ("abandoned",),
}


def validate_event(event: TraceEvent) -> None:
    if event.kind not in EVENT_KIND_LIST:
        raise ValueError(f"unknown event kind {event.kind!r}")
    missing = [k for k in _REQUIRED[event.kind] if getattr(event.payload, k) is None]
    if missing:
        raise ValueError(f"event kind {event.kind!r} missing required payload keys: {missing}")
    if event.kind == "llm_request":
        msgs = event.payload.messages or []
        if not msgs:
            raise ValueError("llm_request must carry a non-empty messages view")
        expected = messages_view_hash(msgs)
        if event.messages_view_hash != expected:
            raise ValueError("llm_request messages_view_hash does not match payload.messages")
    if event.kind == "task_end":
        if event.payload.abandoned and not event.payload.abandonment_reason:
            raise ValueError("abandoned task_end requires abandonment_reason")


class TraceWriter:
    """Appends validated events to a gzipped JSONL file."""

    def __init__(self, path: str | Path, metadata: RunMetadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata
        self._seq = 0
        # Header row carries the run metadata; readers use it to validate.
        header = {"record": "metadata", **metadata.model_dump()}
        with gzip.open(self.path, "at", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    def append(
        self,
        kind: str,
        turn: int,
        t_wall_ms: float,
        payload: dict,
        messages_view_hash_: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            seq=self._seq,
            turn=turn,
            t_wall_ms=t_wall_ms,
            kind=kind,  # type: ignore[arg-type]
            messages_view_hash=messages_view_hash_,
            payload=payload,
        )
        validate_event(event)
        with gzip.open(self.path, "at", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        self._seq += 1
        return event

    def next_seq(self) -> int:
        return self._seq


def read_trace(path: str | Path) -> tuple[RunMetadata, list[TraceEvent]]:
    """Read a gzipped JSONL trace -> (metadata, events). Validates every row."""
    meta: RunMetadata | None = None
    events: list[TraceEvent] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record") == "metadata":
                row = {k: v for k, v in row.items() if k != "record"}
                meta = RunMetadata.model_validate(row)
                continue
            try:
                ev = TraceEvent.model_validate(row)
            except Exception as e:  # surface the file+line, not just the pydantic error
                raise ValueError(f"{path}:{lineno + 1}: invalid event: {e}") from e
            validate_event(ev)
            if meta is None:
                raise ValueError(f"{path}: event row appears before metadata header")
            events.append(ev)
    if meta is None:
        raise ValueError(f"{path}: no metadata header")
    return meta, events


def validate_trace_file(path: str | Path) -> dict:
    """Full-file validation used by `agentflight check`. Returns a summary."""
    meta, events = read_trace(path)
    kinds: dict[str, int] = {}
    seqs = [e.seq for e in events]
    if seqs != list(range(len(events))):
        raise ValueError(f"{path}: seq is not 0..n-1 monotonic")
    if events and events[-1].kind != "task_end":
        raise ValueError(f"{path}: trace does not end with task_end (truncated run?)")
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {"task_id": meta.task_id, "policy_id": meta.policy_id, "events": len(events), "kinds": kinds}
