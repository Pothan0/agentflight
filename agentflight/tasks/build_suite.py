"""AgentFlight-QA v1 builder (build-prompt section 6).

Downloads 2WikiMultihopQA + HotpotQA (hard subset), samples 60 tasks with a
FIXED seed (30 two-hop, 30 three-hop), builds the shared passage corpus
(support + distractor pool, ~3k passages), attaches per-task distractor
corpora parameterized to {4k, 16k, 32k} total context tokens, and enforces
the distractor-leak check DURING sampling.

Run:  python -m agentflight.tasks.build_suite --out data/agentflight_qa_v1.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random

from agentflight.metrics.tokens import count_tokens
from agentflight.tasks.leak_check import contains_answer, is_string_checkable
from agentflight.tasks.schema import AgentFlightSuite, Passage, QATask, SuiteMetadata

SEED = 42
N_TWO_HOP = 30
N_THREE_HOP = 30
TIERS = {"4k": 4000, "16k": 16000, "32k": 32000}
CORPUS_TARGET = 3000

# Documented source->hops mapping (recorded in the suite metadata).
HOP_FILTERING = {
    "hotpot": "level == 'hard' -> 2-hop",
    "2wiki_2hop": "type in {comparison} -> 2-hop",
    "2wiki_3hop": "type in {compositional, bridge_comparison, inference} -> 3-hop",
}


def _doc_id(source: str, title: str, text: str) -> str:
    h = hashlib.sha1(f"{title}|{text}".encode()).hexdigest()[:12]
    safe = "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
    return f"{source}-{safe}-{h}"


def _join_sents(sents: list[str]) -> str:
    return " ".join(s.strip() for s in sents if s and s.strip())


def load_hotpot(limit: int | None = None) -> list[dict]:
    """HotpotQA distractor validation, hard level -> 2-hop tasks + passages."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    out = []
    for x in ds:
        if x["level"] != "hard":
            continue
        ctx_titles = x["context"]["title"]
        ctx_sents = x["context"]["sentences"]
        support_titles = sorted(set(x["supporting_facts"]["title"]))
        out.append(
            {
                "source": "hotpot",
                "hops": 2,
                "question": x["question"],
                "answer": x["answer"],
                "answer_type": _answer_type(x["answer"]),
                "support_titles": support_titles,
                "passages": [
                    {"title": t, "text": _join_sents(s), "support": t in support_titles}
                    for t, s in zip(ctx_titles, ctx_sents, strict=False)
                ],
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def load_2wiki(limit: int | None = None) -> list[dict]:
    """2WikiMultihopQA validation -> 2-hop (comparison) + 3-hop tasks."""
    from datasets import load_dataset

    last_err: Exception | None = None
    # xanhho/ hosts a legacy dataset script (unsupported by datasets>=3);
    # framolfese/ is parquet-native with the same content. Verified 2026-11.
    for repo in ("framolfese/2WikiMultihopQA", "scholarly-shadows-syndicate/2WikiMultihopQA"):
        try:
            ds = load_dataset(repo, split="validation")
            break
        except Exception as e:  # try next mirror
            last_err = e
    else:
        raise RuntimeError(f"could not load 2WikiMultihopQA from any known mirror: {last_err}")

    TWO_HOP_TYPES = {"comparison"}
    THREE_HOP_TYPES = {"compositional", "bridge_comparison", "inference"}
    out = []
    for x in ds:
        qtype = (x.get("type") or "").strip().lower()
        if qtype not in TWO_HOP_TYPES | THREE_HOP_TYPES:
            continue
        hops = 2 if qtype in TWO_HOP_TYPES else 3
        question = x.get("question") or ""
        answer = x.get("answer") or ""
        ctx = x.get("context") or {}
        sf_titles = set(x.get("supporting_facts", {}).get("title", []))
        sf_lower = {t.lower(): t for t in sf_titles}
        passages: list[dict] = []
        support_titles: list[str] = []
        if isinstance(ctx, dict):  # HF context dict-of-lists (support + distractors mixed)
            titles = ctx.get("title", [])
            sent_lists = ctx.get("sentences", [[]] * len(titles))
            for t, s in zip(titles, sent_lists, strict=False):
                is_support = t in sf_titles or t.lower() in sf_lower
                passages.append({"title": t, "text": _join_sents(s), "support": is_support})
                if is_support:
                    support_titles.append(t)
        elif isinstance(ctx, list):  # list of [title, [sents]] pairs (all supporting)
            for item in ctx:
                t, s = item[0], item[1]
                passages.append({"title": t, "text": _join_sents(s), "support": True})
                support_titles.append(t)
        out.append(
            {
                "source": "2wiki",
                "hops": hops,
                "question": question,
                "answer": answer,
                "answer_type": _answer_type(answer),
                "support_titles": sorted(set(support_titles)),
                "passages": passages,
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def _answer_type(answer: str) -> str:
    if answer.strip().lower() in ("yes", "no"):
        return "yesno"
    try:
        float(answer.replace(",", ""))
        return "number"
    except ValueError:
        return "entity"


def build_corpus(pools: list[dict], filler_pools: list[dict], rng: random.Random) -> tuple[list[Passage], dict[str, str]]:
    """Shared corpus: all support passages first, then non-support paragraphs
    from the SELECTED tasks, then filler paragraphs from the FULL datasets
    (unrelated Wikipedia paragraphs) up to CORPUS_TARGET. Filler keeps the
    distractor dials reachable at the 16k/32k tiers."""
    corpus: dict[str, Passage] = {}
    support_ids_by_task: list[list[str]] = []

    def add(title: str, text: str, source: str, origin: str) -> str | None:
        if not text or not text.strip():
            return None
        did = _doc_id(source, title, text)
        if did not in corpus:
            corpus[did] = Passage(doc_id=did, title=title, text=text, source=source, origin=origin)
        return did

    for item in pools:
        ids: list[str] = []
        for p in item["passages"]:
            if p["support"]:
                did = add(p["title"], p["text"], item["source"], "support")
                if did:
                    ids.append(did)
        support_ids_by_task.append(ids)

    # Non-support paragraphs of the selected tasks first...
    candidates: list[tuple[str, str, str]] = []
    for item in pools:
        for p in item["passages"]:
            if not p["support"] and p["text"].strip():
                candidates.append((p["title"], p["text"], item["source"]))
    rng.shuffle(candidates)
    for title, text, source in candidates:
        add(title, text, source, "distractor_pool")

    # ...then filler from the full datasets (unrelated paragraphs), dedup via doc_id.
    filler: list[tuple[str, str, str]] = []
    for item in filler_pools:
        for p in item["passages"]:
            if p["text"].strip():
                filler.append((p["title"], p["text"], item["source"]))
    rng.shuffle(filler)
    for title, text, source in filler:
        if len(corpus) >= CORPUS_TARGET:
            break
        add(title, text, source, "distractor_pool")
    return list(corpus.values()), {item["question"]: ids for item, ids in zip(pools, support_ids_by_task, strict=False)}


def sample_distractors(
    task: dict,
    support_ids: list[str],
    corpus: list[Passage],
    target_tokens: int,
    rng: random.Random,
    support_id_set: set[str],
) -> tuple[list[str], int, int]:
    """Seeded, leak-checked distractor sampling until ~target_tokens total.
    Returns (distractor_ids, n_rejected_leaks, total_tokens)."""
    answer, atype = task["answer"], task["answer_type"]
    string_check = is_string_checkable(answer, atype)
    support_texts = [p.text for p in corpus if p.doc_id in support_id_set]
    candidates = [
        p for p in corpus if p.doc_id not in support_id_set and p.origin == "distractor_pool"
    ]
    rng.shuffle(candidates)
    chosen: list[str] = []
    total = sum(count_tokens(t) for t in support_texts)
    rejected = 0
    for p in candidates:
        if total >= target_tokens:
            break
        if string_check and contains_answer(p.text, answer):
            rejected += 1
            continue
        chosen.append(p.doc_id)
        total += count_tokens(p.text)
    return chosen, rejected, total


def build_suite(out_path: str, n_two: int = N_TWO_HOP, n_three: int = N_THREE_HOP) -> AgentFlightSuite:
    rng = random.Random(SEED)
    hotpot = load_hotpot()
    wiki = load_2wiki()
    two_pool = [x for x in hotpot] + [x for x in wiki if x["hops"] == 2]
    three_pool = [x for x in wiki if x["hops"] == 3]
    if len(two_pool) < n_two:
        raise RuntimeError(f"only {len(two_pool)} 2-hop candidates")
    if len(three_pool) < n_three:
        raise RuntimeError(f"only {len(three_pool)} 3-hop candidates")

    rng.shuffle(two_pool)
    rng.shuffle(three_pool)
    selected = two_pool[:n_two] + three_pool[:n_three]

    # Filler pool = the FULL loaded datasets (unrelated Wikipedia paragraphs);
    # dedup happens inside build_corpus via doc_id.
    corpus, support_map = build_corpus(selected, two_pool + three_pool, random.Random(SEED + 1))
    corpus_ids = {p.doc_id for p in corpus}
    support_id_sets = {q: {d for d in ids if d in corpus_ids} for q, ids in support_map.items()}

    # Tier assignment: cycle {4k, 16k, 32k} over the shuffled task order so
    # each tier holds a mix of hops; 20 tasks per tier.
    tier_names = list(TIERS)
    tasks: list[QATask] = []
    total_rejected = 0
    for i, item in enumerate(selected):
        tier = tier_names[i % len(tier_names)]
        target = TIERS[tier]
        sid = sorted(support_id_sets.get(item["question"], set()))
        dids, rejected, total = sample_distractors(
            item, sid, corpus, target, random.Random(SEED + 1000 + i), support_id_sets.get(item["question"], set())
        )
        total_rejected += rejected
        tasks.append(
            QATask(
                task_id=f"afqa-{i:04d}-{item['source']}-{item['hops']}hop",
                source=item["source"],
                hops=item["hops"],
                question=item["question"],
                gold_answer=item["answer"],
                answer_type=item["answer_type"],
                support_doc_ids=sid,
                distractor_doc_ids=dids,
                context_tier=tier,
                tier_target_tokens=target,
                tier_total_tokens=total,
            )
        )

    suite = AgentFlightSuite(
        metadata=SuiteMetadata(
            schema_version="1.0.0",
            suite_name="AgentFlight-QA v1",
            seed=SEED,
            created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            tokenizer_backend="heuristic",
            hop_filtering=HOP_FILTERING,
            counts={
                "tasks": len(tasks),
                "two_hop": sum(1 for t in tasks if t.hops == 2),
                "three_hop": sum(1 for t in tasks if t.hops == 3),
                "corpus_passages": len(corpus),
                "tier_4k": sum(1 for t in tasks if t.context_tier == "4k"),
                "tier_16k": sum(1 for t in tasks if t.context_tier == "16k"),
                "tier_32k": sum(1 for t in tasks if t.context_tier == "32k"),
                "leak_rejected_distractors": total_rejected,
            },
            notes=[
                "Answers scored by EM (primary) + token-F1 (secondary); no LLM judge.",
                "Distractors leak-checked DURING sampling; yes/no and <4-char answers exempt "
                "from the string test (recorded per task via answer_type).",
                "Distractor paragraphs drawn from each task's own dataset context paragraphs "
                "(unrelated to that question) from the shared ~3k-passage corpus.",
                "Tier totals measured with the heuristic tokenizer; the GPU host re-verifies "
                "with the model tokenizer at E0 and records any drift in ENVIRONMENT.json.",
            ],
        ),
        corpus=corpus,
        tasks=tasks,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(suite.model_dump_json(indent=1))
    return suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + freeze AgentFlight-QA v1")
    ap.add_argument("--out", default="data/agentflight_qa_v1.json")
    args = ap.parse_args()
    suite = build_suite(args.out)
    print(json.dumps(suite.metadata.counts, indent=2))
    # Post-build whole-suite leak re-check (build-prompt VERIFY).
    from agentflight.tasks.leak_check import check_suite

    report = check_suite(suite)
    print(f"leak-check pass={report['pass']} n_failed={report['n_failed']} n_exempt={report['n_exempt']}")
    if not report["pass"]:
        for f_ in report["failures"][:10]:
            print("FAIL", f_)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
