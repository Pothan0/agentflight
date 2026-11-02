"""GPU-hour ledger (build-prompt section 8).

Append-only JSONL at results/gpu_ledger.jsonl. Every phase appends consumed
hours; `agentflight budget` prints the running total against the weekly
quota. The KAGGLE weekly reset (30 h) is a config constant here, overridable
via --weekly-quota.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LedgerRow:
    ts: str
    phase: str
    hours: float
    note: str
    session_id: str = ""


class GpuLedger:
    def __init__(self, path: str | Path, weekly_quota_hours: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.weekly_quota_hours = weekly_quota_hours

    def append(self, phase: str, hours: float, note: str, ts: str, session_id: str = "") -> LedgerRow:
        if hours < 0:
            raise ValueError("hours must be >= 0 (honest accounting)")
        row = LedgerRow(ts=ts, phase=phase, hours=hours, note=note, session_id=session_id)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        return row

    def rows(self) -> list[LedgerRow]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(LedgerRow(**json.loads(line)))
        return out

    def total_hours(self) -> float:
        return sum(r.hours for r in self.rows())

    def hours_by_phase(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rows():
            out[r.phase] = out.get(r.phase, 0.0) + r.hours
        return out

    def remaining(self) -> float:
        return self.weekly_quota_hours - self.total_hours()
