"""LLM backends.

- ``OpenAIBackend``: vLLM's OpenAI-compatible server, STREAMING MANDATORY
  (TTFT/TPOT come from client-side chunk timestamps; server is localhost so
  network noise is negligible — stated in the paper). Captures
  ``usage.prompt_tokens_details.cached_tokens`` when the installed vLLM
  version provides it (day-0 VERIFIED; falls back to /metrics gauges in the
  replay layer) and ``reasoning_content`` for Qwen3 thinking.
- ``MockBackend``: deterministic scripted responses so the full agent loop is
  testable on CPU (build-prompt section 15 step 4).

Both return the same ``LLMResponse`` shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from agentflight.harness.parsers import normalize_server_tool_calls, split_think


@dataclass
class LLMResponse:
    completion_text: str  # raw text incl. <think> if the server inlines it
    think_text: str
    visible_text: str
    tool_calls: list[dict]  # canonical: [{id, name, arguments(dict)}]
    usage: dict  # {prompt_tokens, completion_tokens, cached_tokens, ...}
    timings_ms: dict  # {ttft, total, inter_token: [...]}


class LLMBackend(Protocol):
    model_id: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> LLMResponse: ...


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        parse_mode: str = "native",  # "native" | "json_fallback"
        request_timeout_s: float = 600.0,
    ):
        from openai import OpenAI  # local import: client is a core dep but lazy here

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout_s)
        self.model_id = model_id
        self.parse_mode = parse_mode

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = tools
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if chat_template_kwargs:
            kwargs["chat_template_kwargs"] = chat_template_kwargs

        stream = self.client.chat.completions.create(**kwargs)
        parts: list[str] = []
        reasoning_parts: list[str] = []
        server_calls: list = []
        ttft: float | None = None
        inter: list[float] = []
        last_t = None
        usage: dict = {}
        for chunk in stream:
            now = time.perf_counter()
            if chunk.usage:
                u = chunk.usage
                cached = 0
                details = getattr(u, "prompt_tokens_details", None)
                if details is not None:
                    cached = int(getattr(details, "cached_tokens", 0) or 0)
                usage = {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "cached_tokens": cached,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if delta.content:
                if ttft is None:
                    ttft = (now - t0) * 1000.0
                elif last_t is not None:
                    inter.append((now - last_t) * 1000.0)
                parts.append(delta.content)
            if delta.tool_calls:
                server_calls.extend(delta.tool_calls)
                if ttft is None:
                    ttft = (now - t0) * 1000.0
            last_t = now
        total_ms = (time.perf_counter() - t0) * 1000.0
        raw = "".join(parts)
        server_calls = _merge_streamed_tool_calls(server_calls)
        if self.parse_mode == "native":
            tool_calls = normalize_server_tool_calls(server_calls)
            visible = raw
            # reasoning may arrive via reasoning_content OR inline <think>
            think, visible2 = split_think(visible)
            visible = visible2 if think else visible
            think_text = "".join(reasoning_parts) or think
        else:
            think, visible = split_think(raw)
            think_text = "".join(reasoning_parts) or think
            calls, visible = _fallback_calls(visible)
            tool_calls = [
                {"id": f"fb_{i}", **c} for i, c in enumerate(calls)
            ]
        return LLMResponse(
            completion_text=raw,
            think_text=think_text,
            visible_text=visible,
            tool_calls=tool_calls,
            usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
            timings_ms={"ttft": ttft, "total": total_ms, "inter_token": inter},
        )


def _merge_streamed_tool_calls(streamed: list) -> list:
    """Reassemble streamed tool_call deltas into complete tool_call objects."""
    by_index: dict[int, dict] = {}
    for tc in streamed:
        idx = getattr(tc, "index", 0) or 0
        slot = by_index.setdefault(idx, {"id": None, "name": "", "arguments": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name if not slot["name"] else slot["name"] + fn.name
            if getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments
    out = []
    for idx in sorted(by_index):
        slot = by_index[idx]
        out.append(_SimpleToolCall(slot["id"] or f"call_{idx}", slot["name"], slot["arguments"]))
    return out


class _SimpleToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _SimpleFunction(name, arguments)


class _SimpleFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


def _fallback_calls(visible: str) -> tuple[list[dict], str]:
    from agentflight.harness.parsers import parse_json_fallback

    calls, visible = parse_json_fallback(visible)
    return [{"name": c["name"], "arguments": c["arguments"]} for c in calls], visible


class MockBackend:
    """Deterministic scripted backend for CPU tests and E1 dry runs.

    ``script`` is a list of steps; each step is a dict with any of:
      - ``tool_calls``: [{name, arguments}]
      - ``content``: visible text
      - ``think``: thinking text
    The LAST step repeats once exhausted. Every call returns the step at
    index = (number of prior calls), so the loop is fully deterministic.
    """

    def __init__(self, script: list[dict], model_id: str = "mock-model"):
        if not script:
            raise ValueError("MockBackend needs a non-empty script")
        self.script = script
        self.model_id = model_id
        self.calls_made = 0
        self.seen_views: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=None, chat_template_kwargs=None) -> LLMResponse:
        self.seen_views.append(json.loads(json.dumps(messages)))
        step = self.script[min(self.calls_made, len(self.script) - 1)]
        self.calls_made += 1
        raw_parts = []
        think = step.get("think", "")
        if think:
            raw_parts.append(f"<think>\n{think}\n</think>\n")
        content = step.get("content", "")
        raw_parts.append(content)
        calls = [
            {"id": f"mock_{self.calls_made}_{i}", "name": c["name"], "arguments": c.get("arguments", {})}
            for i, c in enumerate(step.get("tool_calls", []))
        ]
        n_completion = max(1, len("".join(raw_parts)) // 4)
        return LLMResponse(
            completion_text="".join(raw_parts),
            think_text=think,
            visible_text=content,
            tool_calls=calls,
            usage={"prompt_tokens": 100, "completion_tokens": n_completion, "cached_tokens": 0},
            timings_ms={"ttft": 10.0, "total": 20.0, "inter_token": [5.0]},
        )
