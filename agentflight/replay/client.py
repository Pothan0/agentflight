"""Replay client: re-serves recorded token sequences verbatim
(max_tokens=1, no generation) while measuring TTFT and captured
cached_tokens. Streaming is mandatory — TTFT comes from the first streamed
chunk timestamp (build-prompt section 2, Mode R)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentflight.replay.server import ServerConfig


@dataclass
class ReplayRecord:
    session_id: str
    seq_in_session: int
    condition: str
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    error: str | None = None
    request_t: float = 0.0  # wall-clock send time (epoch s)


@dataclass
class _Acc:
    parts: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ReplayClient:
    def __init__(self, config: ServerConfig, api_key: str = "EMPTY"):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(base_url=config.base_url(), api_key=api_key, timeout=600.0)

    def replay_one(
        self,
        messages: list[dict],
        session_id: str,
        seq_in_session: int,
        condition: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
    ) -> ReplayRecord:
        t0 = time.time()
        start = time.perf_counter()
        ttft: float | None = None
        usage: dict = {}
        err: str | None = None
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                now = time.perf_counter()
                if chunk.usage:
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                    }
                if chunk.choices:
                    if ttft is None:
                        ttft = (now - start) * 1000.0
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        total_ms = (time.perf_counter() - start) * 1000.0
        return ReplayRecord(
            session_id=session_id,
            seq_in_session=seq_in_session,
            condition=condition,
            ttft_ms=ttft,
            total_ms=total_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            error=err,
            request_t=t0,
        )
