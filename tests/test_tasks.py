"""G1 tests: distractor-leak check + frozen-suite validation (skipped if the
suite file is absent, e.g. before first build)."""

import json
from pathlib import Path

import pytest

from agentflight.tasks.leak_check import (
    check_suite,
    check_task,
    contains_answer,
    is_string_checkable,
)
from agentflight.tasks.schema import AgentFlightSuite, Passage, QATask

SUITE_PATH = Path(__file__).resolve().parents[1] / "data" / "agentflight_qa_v1.json"

DOC_LEAK = Passage(doc_id="d-leak", title="Leak", text="Alice Smith founded the company.",
                   source="test", origin="distractor_pool")
DOC_CLEAN = Passage(doc_id="d-clean", title="Clean", text="Nothing relevant here.",
                    source="test", origin="distractor_pool")
DOC_SUPPORT = Passage(doc_id="d-sup", title="Sup", text="Founded by Alice Smith in 1901.",
                      source="test", origin="support")
DOCS = {d.doc_id: d for d in (DOC_LEAK, DOC_CLEAN, DOC_SUPPORT)}


def _task(answer="Alice Smith", answer_type="entity", distractors=("d-leak",), supports=("d-sup",)) -> QATask:
    return QATask(
        task_id="t1", source="test", hops=2, question="Who founded it?", gold_answer=answer,
        answer_type=answer_type, support_doc_ids=list(supports), distractor_doc_ids=list(distractors),
        context_tier="4k", tier_target_tokens=4000, tier_total_tokens=4000,
    )


def test_leak_detected_in_distractor():
    r = check_task(_task(), DOCS)
    assert not r["pass"] and any("distractor" in v for v in r["violations"])


def test_clean_task_passes():
    r = check_task(_task(distractors=("d-clean",)), DOCS)
    assert r["pass"]


def test_yesno_exempt_from_string_check():
    assert not is_string_checkable("yes", "yesno")
    r = check_task(_task(answer="yes", answer_type="yesno", distractors=("d-leak",)), DOCS)
    assert r["pass"] and r["exempt"]


def test_support_must_contain_answer():
    r = check_task(_task(supports=("d-clean",)), DOCS)
    assert not r["pass"] and any("supporting" in v for v in r["violations"])


def test_contains_answer_normalization():
    assert contains_answer("The ALICE   SMITH story", "alice smith")
    assert not contains_answer("Bob Jones", "alice smith")


@pytest.mark.skipif(not SUITE_PATH.exists(), reason="suite not built yet (run agentflight.tasks.build_suite)")
class TestFrozenSuite:
    @classmethod
    def setup_class(cls):
        with open(SUITE_PATH, encoding="utf-8") as f:
            cls.suite = AgentFlightSuite.model_validate(json.load(f))

    def test_shape(self):
        m = self.suite.metadata.counts
        assert m["tasks"] == 60 and m["two_hop"] == 30 and m["three_hop"] == 30
        assert m["tier_4k"] == m["tier_16k"] == m["tier_32k"] == 20

    def test_distractor_dials_hit_targets(self):
        target = {"4k": 4000, "16k": 16000, "32k": 32000}
        for t in self.suite.tasks:
            assert t.tier_total_tokens >= target[t.context_tier], (
                f"{t.task_id} tier {t.context_tier} = {t.tier_total_tokens} < {target[t.context_tier]}"
            )
            assert t.tier_total_tokens < target[t.context_tier] * 1.10

    def test_all_doc_ids_resolve(self):
        ids = {p.doc_id for p in self.suite.corpus}
        for t in self.suite.tasks:
            for d in t.support_doc_ids + t.distractor_doc_ids:
                assert d in ids

    def test_wholesale_leak_check_passes(self):
        report = check_suite(self.suite)
        assert report["pass"], report["failures"][:5]
