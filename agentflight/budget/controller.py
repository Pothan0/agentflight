"""Adaptive scope controller (build-prompt section 8).

Knows the phase priority order and REFUSES to start phase N+1's full matrix
until N's core is done. Phases are resumable via --skip-existing; the
controller only tracks which phases have recorded completion rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase:
    id: str  # "E0".."E6"
    name: str
    est_gpu_hours: str
    priority: str  # "MUST" | "SHOULD" | "STRETCH"
    requires: tuple[str, ...]  # prerequisite phase ids


PHASES: tuple[Phase, ...] = (
    Phase("E0", "day-0 verification + smoke runs", "1.0", "MUST", ()),
    Phase("E1", "recording: 60 tasks x P0 (Mode-R input + P0 baseline)", "4-6", "MUST", ("E0",)),
    Phase("E2", "Mode-R replay matrix", "3-4", "MUST", ("E1",)),
    Phase("E3", "Mode-Q core policies {P0, P4, P5}", "5-7", "MUST", ("E1",)),
    Phase("E4", "Mode-Q {P1, P2, P3}", "6-8", "SHOULD", ("E3",)),
    Phase("E5", "P6 cascade on 32k tier + second P0 for variance", "2-3", "SHOULD", ("E4",)),
    Phase("E6", "stretch: 4B subset or 8B-fp16 TP=2 subset", "2-4", "STRETCH", ("E4",)),
)

# Budget-shortfall rule (section 8): cut tasks 60->40 before cutting
# policies; cut the 32k tier LAST (where the interesting effects live).
CUT_ORDER = ["drop tasks 60->40", "drop 16k tier", "drop 32k tier (LAST)"]


class BudgetController:
    def __init__(self, completed_path: str | Path):
        self.path = Path(completed_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("record") == "phase_complete":
                        self.completed.add(row["phase"])

    def mark_complete(self, phase: str, ts: str, note: str = "") -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"record": "phase_complete", "phase": phase, "ts": ts, "note": note}) + "\n")
        self.completed.add(phase)

    def can_start(self, phase_id: str) -> tuple[bool, str]:
        phase = next((p for p in PHASES if p.id == phase_id), None)
        if phase is None:
            return (False, f"unknown phase {phase_id!r}")
        missing = [r for r in phase.requires if r not in self.completed]
        if missing:
            return (False, f"{phase_id} requires {missing} to be complete first")
        return (True, "")

    def next_phase(self) -> Phase | None:
        """First phase (in fixed order) that is not complete and whose
        prerequisites are all complete."""
        for p in PHASES:
            if p.id not in self.completed and all(r in self.completed for r in p.requires):
                return p
        return None

    def status(self) -> dict:
        return {
            "completed": sorted(self.completed),
            "next_phase": (self.next_phase().id if self.next_phase() else None),
            "phases": [
                {"id": p.id, "name": p.name, "priority": p.priority, "est_gpu_hours": p.est_gpu_hours}
                for p in PHASES
            ],
            "cut_order_if_short": CUT_ORDER,
        }
