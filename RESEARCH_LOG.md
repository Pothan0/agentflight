# AgentFlight — RESEARCH LOG

Append one entry per working session: GPU-hours spent, what ran, findings,
surprises. This log is part of the deliverable (build-prompt section 9).

---

## 2026-11-01 — build session 0 (CPU, local machine, 0 GPU-h)

**What was done (kickoff steps 2–5 of the brief):**

- Scaffolded the `agentflight` package: traces (gzipped-JSONL event stream,
  pydantic-validated, append-only writer), policies P0–P6 (pure functions,
  purity-tested), harness (3-tool registry, native + JSON-fallback tool-call
  parsing, streaming OpenAI backend + deterministic MockBackend, agent loop),
  tasks (suite builder + leak check), replay (vLLM server manager, streaming
  replay client, affinity/round-robin scheduler), metrics (KV constants from
  config.json, TTFT/TPOT/cache-rate/cost formulas, bootstrap CIs), scoring
  (SQuAD-style EM/F1, abandonment), budget (GPU ledger + phase controller),
  CLI (`record|replay|execute|report|check|budget`), experiments E0/E2,
  kaggle/ bootstrap + session notebook, sglang/ UNTESTED configs,
  docs/claims.md written BEFORE any run (7 pre-registered claims).
- Verified data sources live: HotpotQA validation = 7,405 rows, ALL level
  `hard` (convenient for the brief's hard subset). 2WikiMultihopQA via
  `framolfese/2WikiMultihopQA` (parquet-native; `xanhho/` mirror hosts a
  legacy dataset script that datasets>=3 refuses). Type mapping: comparison→
  2-hop (3,040 rows); compositional/bridge_comparison/inference→3-hop (9,536).
- Built + froze `data/agentflight_qa_v1.json`: 60 tasks (30×2-hop, 30×3-hop),
  ~3k-passage shared corpus, distractor dials {4k,16k,32k}, leak-check
  enforced during sampling.

**Findings / decisions recorded:**

- Policy purity design win: policies recompute the view from the FULL history
  every turn → no state, deterministic, testable on CPU with a scripted
  summarizer.
- Yes/no answers are exempt from the string leak-check (any corpus contains
  "yes"/"no"); exemption recorded per-task via `answer_type`.
- T4 kernel reality baked into code: GPTQ-Int4 defaults, no AWQ, no fp8-KV,
  fp16 KV = 144 KiB/token for Qwen3-8B.

**Surprises:** HotpotQA `validation` split being entirely `hard` simplified
sampling (no subsetting needed).

## 2026-11-02 — build session 0b (CPU, local machine, 0 GPU-h)

**Verification state at end of session:**

- G1 gate: 76/76 unit tests green (policy purity ×7, trace round-trip,
  metrics vs hand-computed fixtures incl. the 147,456 B/token Qwen3-8B KV
  constant, `--skip-existing` resume, ledger accounting, leak check on the
  frozen suite). ruff clean.
- Suite freeze verified: tiers hit targets (4k: 4005–4286, 16k: 16008–16365,
  32k: 32006–32352 heuristic tokens), 3,000-passage corpus, leak-check PASS
  (34 leaking distractor candidates rejected during sampling; 14 yes/no
  tasks exempt from the string test).
- E1 dry-run against the frozen suite on the MockBackend: 4 tasks ×
  {P0,P3,P4,P5} = 16 traces, all validate; `agentflight report` generates
  the Mode-Q table purely from results files (G5 path exercised; EM=0 is the
  mock's canned answer, rows labeled `mock: true`).
- E0 ran in CPU-MODE (verdict recorded in results/ENVIRONMENT.json): no
  T4s on this machine, so gate G2 is still OPEN and intentionally not marked
  complete in the ledger.

**Next:** Kaggle session 1 → run `experiments/e0_day0_check.py` on 2×T4,
mark E0 in the ledger, then E1 recording at the 4k tier first.
