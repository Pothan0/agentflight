# AgentFlight

**An open agent-trace replay & systems-quality benchmark.**

AgentFlight records real multi-turn agent traces on open models, then:

- **Mode R — REPLAY**: re-serves the recorded token sequences verbatim under different
  serving-layer configurations (prefix caching on/off, session-affinity vs round-robin
  interleaving, LMCache CPU-offload on/off), measuring TTFT, TPOT, throughput,
  `cached_tokens` / prefix-cache hit rate, KV-bytes touched, GPU-seconds.
- **Mode Q — RE-EXECUTION**: re-runs the agent task from scratch under a deterministic
  **context policy** (keep-everything, sliding window, LLM summarization, chunked
  compaction, think-discard, tool-result elision, re-compaction cascade), measuring
  end-task accuracy (EM/F1), turns-to-answer, abandonment, and prompt-token efficiency.

The headline artifact is the **joint curve** — accuracy (Mode Q) x cost (Mode R) per
policy per context tier: the systems-quality frontier for open models.

**Thesis under test:** *agent reliability is a systems problem.* KV management, context
compaction, think-token retention, and session routing are one pipeline — and nobody has
measured how serving-layer/context-layer policies affect end-task agent accuracy
end-to-end on open models.

## Status

Under active construction. See `docs/claims.md` (falsifiable claims, written before
running), `docs/design.md`, and `RESEARCH_LOG.md`. The build brief is the spec of record.

## Install

```bash
pip install -e ".[dev]"        # code + tests (CPU-only, no GPU needed)
pytest                          # G1 gate: unit tests green
```

GPU-host deps (`vllm`, `lmcache`) are intentionally **not** package dependencies; they
are installed per-session by `kaggle/bootstrap.sh` with pinned versions recorded into
`results/ENVIRONMENT.json`.

## Quickstart (CPU, mock LLM)

```bash
agentflight execute --mock --tasks data/agentflight_qa_v1.json --policy P0 --limit 3
agentflight report --results results/
```

## Layout

See the build brief (section 3) — `agentflight/` package (traces, policies, harness,
tasks, replay, metrics, scoring, budget, cli), `experiments/` (E0–E4 runners),
`kaggle/` (session bootstrap), `sglang/` (UNTESTED-on-T4 configs for Ampere+ GPUs),
`tests/`, `results/` (append-only, published to HF), `docs/`.

## License

Apache-2.0.
