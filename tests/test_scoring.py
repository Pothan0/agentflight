"""G1 tests: EM/F1 vs SQuAD hand-computed fixtures, answer extraction,
abandonment logic."""

from agentflight.scoring.abandonment import RunLimits, check_abandonment
from agentflight.scoring.qa import (
    exact_match,
    extract_final_answer,
    normalize_answer,
    score_answer,
    token_f1,
)

LIMITS = RunLimits(max_turns=24, max_tool_calls=12, wall_timeout_ms=1000.0)


def test_normalize():
    assert normalize_answer("The  Sky   is Blue!") == "sky is blue"
    assert normalize_answer("  A  example ") == "example"


def test_exact_match_cases():
    assert exact_match("Alice Smith", "alice smith") == 1.0
    assert exact_match("The Alice Smith", "Alice Smith") == 1.0  # article stripped
    assert exact_match("Bob Jones", "Alice Smith") == 0.0
    assert exact_match("yes", "Yes.") == 1.0


def test_token_f1_hand_computed():
    # pred={novak, m, schmook}, gold={novak, djokovic}: common=1
    # precision=1/3, recall=1/2 -> f1 = 2*(1/3*1/2)/(1/3+1/2) = 0.4
    assert abs(token_f1("Novak M. Schmook", "Novak Djokovic") - 0.4) < 1e-9
    assert token_f1("Alice Smith", "Alice Smith") == 1.0
    assert token_f1("Berlin", "Tokyo") == 0.0
    assert token_f1("", "") == 1.0  # both empty convention


def test_score_answer_dict():
    s = score_answer("the Alice Smith", "Alice Smith")
    assert s["em"] == 1.0 and s["f1"] == 1.0


def test_extract_final_answer():
    assert extract_final_answer("Bla.\nFINAL ANSWER: Alice Smith") == "Alice Smith"
    assert extract_final_answer("final answer - Bob Jones") == "Bob Jones"
    assert extract_final_answer('FINAL ANSWER: "Carol Wei"') == "Carol Wei"
    # multiple: take the LAST
    assert extract_final_answer("FINAL ANSWER: x\nmore\nFINAL ANSWER: y") == "y"
    # no marker: whole text
    assert extract_final_answer("just an answer") == "just an answer"


def test_abandonment_reasons():
    a, r = check_abandonment(turns_used=3, tool_calls_used=2, wall_ms_used=100.0, answered=True, limits=LIMITS)
    assert not a and r is None
    # only wall hit
    a, r = check_abandonment(turns_used=2, tool_calls_used=2, wall_ms_used=1000.0, answered=False, limits=LIMITS)
    assert a and r == "wall_timeout"
    # wall AND turns both over: wall reported first
    a, r = check_abandonment(turns_used=24, tool_calls_used=2, wall_ms_used=1000.0, answered=False, limits=LIMITS)
    assert a and r == "wall_timeout"
    a, r = check_abandonment(turns_used=24, tool_calls_used=2, wall_ms_used=50.0, answered=False, limits=LIMITS)
    assert a and r == "max_turns"
    a, r = check_abandonment(turns_used=2, tool_calls_used=12, wall_ms_used=50.0, answered=False, limits=LIMITS)
    assert a and r == "max_tool_calls"
    a, r = check_abandonment(turns_used=2, tool_calls_used=2, wall_ms_used=50.0, answered=False, limits=LIMITS)
    assert not a  # unanswered but no limit hit: loop will keep going
