"""Trace schema (build-prompt section 4).

JSONL, one event per line, gzipped, pydantic-validated. The ``messages``
payload stores the EXACT post-policy view sent to the model; the pre-policy
history is reconstructible from the event stream, and policy_events record
what each policy dropped/replaced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Kind = Literal[
    "task_start",
    "turn_start",
    "policy_event",
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "task_end",
]

# Event kinds, as a plain tuple for validation and iteration.
EVENT_KIND_LIST = (
    "task_start",
    "turn_start",
    "policy_event",
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "task_end",
)


def messages_view_hash(messages: list[dict]) -> str:
    """sha256 of the exact context view sent (reproducibility anchor)."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RunMetadata(BaseModel):
    """Per-trace run metadata: everything needed to reproduce the run."""

    model_config = {"extra": "forbid"}

    model_id: str
    model_revision: str = "main"
    vllm_version: str | None = None
    temperature: float = 0.0
    seed: int = 0
    tool_schema_hash: str
    task_id: str
    policy_id: str
    context_tier: str  # "4k" | "16k" | "32k"
    tokenizer_backend: str = "heuristic"
    max_turns: int = 24
    max_tool_calls: int = 12
    wall_timeout_ms: int = 8 * 60 * 1000


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    # thinking tokens if the server reports them (Qwen3 reasoning)
    completion_tokens_details: dict[str, Any] | None = None


class Timings(BaseModel):
    ttft_ms: float | None = None
    total_ms: float | None = None
    inter_token_ms: list[float] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    call_id: str


class EventPayload(BaseModel):
    """Loose payload: exact shape depends on ``kind``. Extra keys allowed so
    the schema survives honest evolution; the REQUIRED core of each kind is
    asserted by the writer (see writer.append)."""

    model_config = {"extra": "allow"}

    messages: list[dict[str, Any]] | None = None
    usage: Usage | None = None
    timings_ms: Timings | None = None
    completion_text: str | None = None
    think_text: str | None = None
    tool_calls: list[ToolCallRecord] | None = None
    tool_call: ToolCallRecord | None = None
    tool_result: str | None = None
    tool_error: str | None = None
    decision: str | None = None
    detail: str | None = None
    tokens_before: int | None = None
    tokens_after: int | None = None
    policy_id: str | None = None
    answer: str | None = None
    abandoned: bool | None = None
    abandonment_reason: str | None = None
    em: float | None = None
    f1: float | None = None
    note: str | None = None
    summarizer_calls: int | None = None


class TraceEvent(BaseModel):
    model_config = {"extra": "forbid"}

    seq: int = Field(ge=0)
    # turn = agent turn index; -1 marks out-of-band calls (policy summarizer)
    turn: int = Field(ge=-1)
    t_wall_ms: float
    kind: Kind
    messages_view_hash: str | None = None
    payload: EventPayload = Field(default_factory=EventPayload)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        if v not in EVENT_KIND_LIST:
            raise ValueError(f"unknown event kind {v!r}")
        return v


class Trace(BaseModel):
    """A complete trace = metadata + ordered events."""

    metadata: RunMetadata
    events: list[TraceEvent]
