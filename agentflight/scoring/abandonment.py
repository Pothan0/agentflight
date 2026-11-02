"""Abandonment detection (METR-style early abandonment is itself a tracked
metric, build-prompt section 6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunLimits:
    max_turns: int = 24
    max_tool_calls: int = 12
    wall_timeout_ms: float = 8 * 60 * 1000


def check_abandonment(
    *,
    turns_used: int,
    tool_calls_used: int,
    wall_ms_used: float,
    answered: bool,
    limits: RunLimits,
) -> tuple[bool, str | None]:
    """Return (abandoned, reason). A run is abandoned when it produced no
    answer and hit any limit. Reason names the FIRST limit hit."""
    if answered:
        return (False, None)
    if wall_ms_used >= limits.wall_timeout_ms:
        return (True, "wall_timeout")
    if turns_used >= limits.max_turns:
        return (True, "max_turns")
    if tool_calls_used >= limits.max_tool_calls:
        return (True, "max_tool_calls")
    return (False, None)
