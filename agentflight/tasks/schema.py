"""AgentFlight-QA task schema (build-prompt section 6).

Self-contained, verifiable, context-length-dialable multi-hop QA:
2WikiMultihopQA + HotpotQA (hard subset), 60 tasks (30 two-hop, 30 three-hop),
each with a distractor corpus parameterized to {4k, 16k, 32k} total context
tokens. Answers are short entities -> EM/F1 is objective.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Passage(BaseModel):
    doc_id: str
    title: str
    text: str  # full passage text (sentences joined by space)
    source: str  # "hotpot" | "2wiki"
    origin: str  # "support" | "distractor_pool"


class QATask(BaseModel):
    task_id: str
    source: str  # "hotpot" | "2wiki"
    hops: int  # 2 | 3
    question: str
    gold_answer: str
    answer_type: str  # "entity" | "yesno" | "number"
    support_doc_ids: list[str]
    distractor_doc_ids: list[str] = Field(default_factory=list)
    context_tier: str  # "4k" | "16k" | "32k"
    tier_target_tokens: int
    tier_total_tokens: int  # measured total of support + distractors (heuristic tokenizer)


class SuiteMetadata(BaseModel):
    schema_version: str
    suite_name: str  # "AgentFlight-QA v1"
    seed: int
    created_at: str
    tokenizer_backend: str
    hop_filtering: dict[str, str]  # documented source->hops mapping
    counts: dict[str, int]
    notes: list[str] = Field(default_factory=list)


class AgentFlightSuite(BaseModel):
    metadata: SuiteMetadata
    corpus: list[Passage]
    tasks: list[QATask]
