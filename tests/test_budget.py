"""G1 tests: GPU ledger accounting + adaptive phase controller gating."""

import pytest

from agentflight.budget.controller import CUT_ORDER, PHASES, BudgetController
from agentflight.budget.ledger import GpuLedger


def test_ledger_append_only_and_totals(tmp_path):
    led = GpuLedger(tmp_path / "gpu_ledger.jsonl", weekly_quota_hours=30.0)
    led.append("E0", 1.0, "day-0", "2026-11-01T10:00:00")
    led.append("E1", 4.5, "recording", "2026-11-01T18:00:00")
    led.append("E1", 1.5, "more", "2026-11-02T10:00:00")
    assert led.total_hours() == 7.0
    assert led.hours_by_phase() == {"E0": 1.0, "E1": 6.0}
    assert led.remaining() == 23.0
    with pytest.raises(ValueError, match="honest"):
        led.append("E2", -0.1, "negative", "now")


def test_controller_gates_phase_order(tmp_path):
    c = BudgetController(tmp_path / "phase_state.jsonl")
    ok, why = c.can_start("E1")
    assert not ok and "E0" in why  # E1 requires E0
    assert c.can_start("E0") == (True, "")
    c.mark_complete("E0", "now")
    assert c.can_start("E1") == (True, "")
    assert not c.can_start("E3")[0]  # E3 requires E1
    assert c.next_phase().id == "E1"
    c.mark_complete("E1", "now")
    assert c.next_phase().id == "E2"  # E2 (MUST) before E3? both unlocked: fixed order
    c.mark_complete("E2", "now")
    assert c.next_phase().id == "E3"


def test_controller_unknown_phase_and_status(tmp_path):
    c = BudgetController(tmp_path / "x.jsonl")
    assert not c.can_start("E9")[0]
    status = c.status()
    assert status["next_phase"] == "E0"
    assert [p.id for p in PHASES] == ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    assert status["cut_order_if_short"] == CUT_ORDER
    assert CUT_ORDER[-1].startswith("drop 32k tier")  # 32k cut LAST per section 8
