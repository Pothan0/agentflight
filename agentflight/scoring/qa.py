"""QA scoring: Exact Match (primary) + token-F1 (secondary), SQuAD-style
normalization. Short-entity answers -> objective scoring, no LLM judge
(build-prompt section 6)."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation/articles/whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def _f1_tokens(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold: str) -> float:
    return _f1_tokens(prediction, gold)


def extract_final_answer(completion_text: str) -> str:
    """Extract the answer from a final assistant message.

    Convention: the agent is instructed to end its final message with
    ``FINAL ANSWER: <answer>``. If absent, the whole visible text is used.
    """
    matches = re.findall(r"final answer\s*[:\-]\s*(.+)", completion_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().strip('"')
    return completion_text.strip()


def score_answer(prediction: str, gold: str) -> dict[str, float]:
    return {"em": exact_match(prediction, gold), "f1": token_f1(prediction, gold)}
