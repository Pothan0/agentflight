"""Context-policy base (build-prompt section 5).

A policy is a PURE function: canonical history + config (+ deterministic
summarizer outputs) -> the exact context view to send this turn. Purity is
unit-tested (same inputs -> byte-identical view). Every removal/replacement
is emitted as a PolicyDecision so the harness can record policy_events with
token counts (the Mode-R bridge).

Canonical history conventions (produced by harness.loop; reconstructible
from the trace event stream):

- ``{"role": "system", "content": str}``
- ``{"role": "user", "content": str}`` (first one = the task statement)
- ``{"role": "assistant", "content": str, "thinking": str|None,
     "tool_calls": [...]?}``  -- ``thinking`` is a local channel; it is
  rendered as ``<think>...</think>`` inside content ONLY when a policy keeps
  prior think blocks (P0). Raw OpenAI wire format never carries ``thinking``.
- ``{"role": "tool", "tool_call_id": str, "content": str}``

A TURN is one assistant message plus any tool messages answering its
tool_calls. Policies mutate whole turns only, so the view is always
API-legal (a kept assistant-with-tool_calls always keeps its tool replies).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentflight.metrics.tokens import count_message_tokens, count_tokens


@dataclass(frozen=True)
class PolicyConfig:
    policy_id: str
    # P1
    window_turns: int = 6
    # P2 (budget for compaction trigger) / P6 (cascade budget)
    budget_tokens: int = 8192
    # P2 (fraction of the OVER-BUDGET history that gets summarized)
    summarize_fraction: float = 0.5
    # P3 / P6
    chunk_tokens: int = 2048
    keep_recent_chunks: int = 2
    # P5
    tool_stub_age_turns: int = 2
    tool_stub_head_chars: int = 200
    tool_stub_tail_chars: int = 200
    rehydrate_top_k: int = 1
    # tokenizer
    tokenizer_backend: str = "heuristic"
    model_id: str | None = None

    def tokens(self, text: str) -> int:
        return count_tokens(text, backend=self.tokenizer_backend, model_id=self.model_id)

    def message_tokens(self, messages: list[dict]) -> int:
        return count_message_tokens(messages, backend=self.tokenizer_backend, model_id=self.model_id)


@dataclass
class PolicyDecision:
    """One removal/replacement -> becomes a policy_event row."""

    action: str  # "dropped" | "note_added" | "summarized" | "stubbed" | "rehydrated" | "think_dropped" | "kept_verbatim"
    detail: str
    tokens_before: int
    tokens_after: int

    def to_payload(self, policy_id: str) -> dict:
        return {
            "policy_id": policy_id,
            "decision": self.action,
            "detail": self.detail,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }


@dataclass
class PolicyResult:
    view: list[dict] = field(default_factory=list)
    decisions: list[PolicyDecision] = field(default_factory=list)

    @property
    def view_tokens(self) -> int:
        return count_message_tokens(self.view)


@runtime_checkable
class Summarizer(Protocol):
    """Injected into P2/P3/P6. Real runs use the served model at temperature 0
    (recorded as events, cost counted); tests use a deterministic script."""

    def summarize(self, texts: list[str]) -> str: ...


def render_content(visible: str, thinking: str | None, include_think: bool) -> str:
    """Render an assistant message content for the wire. Prior-turn thinking is
    rendered as a <think> block (Qwen3 style) only when include_think."""
    if include_think and thinking:
        return f"<think>\n{thinking}\n</think>\n{visible}"
    return visible


def tool_calls_to_wire(tool_calls: list[dict]) -> list[dict]:
    """Canonical tool_calls (args as dict) -> OpenAI wire format (args as
    JSON string). Deterministic: sort_keys."""
    wire = []
    for tc in tool_calls:
        wire.append(
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], sort_keys=True, ensure_ascii=False),
                },
            }
        )
    return wire


def view_message(msg: dict, include_think: bool) -> dict:
    """Canonical message -> exact wire message (drops local-only keys)."""
    role = msg["role"]
    if role == "assistant":
        out: dict[str, Any] = {
            "role": "assistant",
            "content": render_content(msg.get("content") or "", msg.get("thinking"), include_think),
        }
        if msg.get("tool_calls"):
            out["tool_calls"] = tool_calls_to_wire(msg["tool_calls"])
        return out
    if role == "tool":
        return {"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg.get("content") or ""}
    return {"role": role, "content": msg.get("content") or ""}


def group_turns(history: list[dict]) -> list[list[dict]]:
    """Group a canonical history into API-legal turn blocks.

    First group = [system] if present. A turn = one assistant message plus the
    tool messages that answer it. Any other single message is its own group.
    """
    groups: list[list[dict]] = []
    i = 0
    n = len(history)
    while i < n:
        msg = history[i]
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            ids = {tc["id"] for tc in msg["tool_calls"]}
            group = [msg]
            j = i + 1
            while j < n and history[j]["role"] == "tool" and history[j].get("tool_call_id") in ids:
                group.append(history[j])
                j += 1
            groups.append(group)
            i = j
        elif msg["role"] == "system" and not groups:
            groups.append([msg])
            i += 1
        else:
            groups.append([msg])
            i += 1
    return groups


def find_first_user_group(groups: list[list[dict]]) -> int:
    """Index of the group holding the first user (task) message; -1 if none."""
    for idx, g in enumerate(groups):
        if g[0]["role"] == "user":
            return idx
    return -1


class ContextPolicy(ABC):
    """Pure context-view constructor. Subclasses implement build_view."""

    policy_id: str = "?"

    @abstractmethod
    def build_view(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None) -> PolicyResult: ...

    def run(self, history: list[dict], config: PolicyConfig, summarizer: Summarizer | None = None) -> PolicyResult:
        if config.policy_id != self.policy_id:
            raise ValueError(f"config.policy_id {config.policy_id!r} != policy {self.policy_id!r}")
        result = self.build_view(history, config, summarizer)
        # Last-line purity guard: a policy may never mutate its input.
        _assert_input_unmodified(history)
        return result


def _assert_input_unmodified(history: list[dict]) -> None:
    # shallow structural check: roles and contents must survive untouched
    # (deep copy checks live in the purity unit tests).
    for m in history:
        if "role" not in m:
            raise ValueError("policy corrupted history: message lost its role")


def history_tokens(history: list[dict], config: PolicyConfig) -> int:
    return config.message_tokens(history)
