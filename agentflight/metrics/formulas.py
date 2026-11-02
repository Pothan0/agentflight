"""Metric formulas (build-prompt section 7).

Every formula here is a pure function with a unit test against a
hand-computed fixture. NO aggregation logic reads anything but its inputs.
"""

from __future__ import annotations

import math
import random


def ttft_ms(t_request_sent_ms: float, t_first_token_ms: float) -> float:
    """TTFT = t(first streamed token) - t(request sent)."""
    return t_first_token_ms - t_request_sent_ms


def tpot_ms(t_first_token_ms: float, t_last_token_ms: float, n_tokens: int) -> float:
    """TPOT = (t(last token) - t(first token)) / (n_tokens - 1).

    Returns NaN when n_tokens <= 1 (undefined: no inter-token intervals).
    """
    if n_tokens <= 1:
        return math.nan
    return (t_last_token_ms - t_first_token_ms) / (n_tokens - 1)


def prefix_cache_hit_rate(cached_tokens: list[int], prompt_tokens: list[int]) -> float:
    """sum(cached) / sum(prompt) over a set of requests. 0.0 when no prompts."""
    total_prompt = sum(prompt_tokens)
    if total_prompt <= 0:
        return 0.0
    return sum(cached_tokens) / total_prompt


def percentiles(values: list[float], qs: list[float]) -> dict[float, float]:
    """Linear-interpolation percentiles (numpy 'linear' method equivalent)."""
    if not values:
        return {q: math.nan for q in qs}
    s = sorted(values)
    out: dict[float, float] = {}
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile {q} outside [0,1]")
        if len(s) == 1:
            out[q] = float(s[0])
            continue
        pos = q * (len(s) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            out[q] = float(s[lo])
        else:
            out[q] = float(s[lo] + (pos - lo) * (s[hi] - s[lo]))
    return out


def cost_per_completed_task(gpu_seconds: float, accuracy: float) -> float:
    """GPU-seconds divided by accuracy (fraction 0..1). Inf when accuracy is 0."""
    if accuracy <= 0.0:
        return math.inf
    return gpu_seconds / accuracy


def bootstrap_ci(
    values: list[float],
    stat,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat(values)`` with a FIXED seed.

    10k resamples per build-prompt section 7. Deterministic: same values +
    seed -> identical interval.
    """
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[max(0, math.floor(alpha * len(stats)))]
    hi = stats[min(len(stats) - 1, math.ceil((1 - alpha) * len(stats)) - 1)]
    return (float(lo), float(hi))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan
