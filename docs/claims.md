# AgentFlight — falsifiable claims (build-prompt section 10)

> WRITTEN BEFORE ANY EXPERIMENT RUN (2026-11-01). Each claim gets a
> PASS / REFUTED / INCONCLUSIVE verdict with measured numbers and CIs in the
> final report. Refutations get equal billing — that is what makes the
> scoreboard credible. NEVER edit these claims after results are known; add
> post-hoc claims in a separate section if needed.

| ID | Claim | Source being tested | Verdict | Measured |
|----|-------|--------------------|---------|----------|
| C1 | Chunked compaction (P3) >= keep-everything (P0) accuracy on 16k/32k tiers while cutting prompt tokens >= 40%. | Baseten: 90.7% vs 81.5% (denoising effect) | UNTESTED | — |
| C2 | Think-discard (P4) cuts prompt tokens >= 15% with accuracy delta <= 2pp vs P0. | Dynamo: ~40% of reasoning tokens discarded | UNTESTED | — |
| C3 | Session-affinity interleaved replay raises prefix-cache hit rate >= 20pp and cuts p50 TTFT >= 30% vs round-robin under 4-concurrent interleaving. | Dynamo: 4x p50 TTFT from session-aware routing | UNTESTED | — |
| C4 | LLM summarization (P2) underperforms chunked compaction (P3) on multi-hop QA. | Baseten: summarization 30.4% vs full-context 81.5% | UNTESTED | — |
| C5 | Under P0, accuracy decreases along the 4k -> 16k -> 32k distractor dial; compaction policies flatten the curve. | Chroma context rot | UNTESTED | — |
| C6 | P6 (re-compaction cascade) loses more accuracy than P3. | Baseten: 4–16pp re-compaction penalty | UNTESTED | — |
| C7 | At least one policy dominates P0 on the cost-per-completed-task frontier at every context tier. | Anthropic: token spend ~80% of variance | UNTESTED | — |

## Verdict rules

- **PASS**: claim holds with the stated margin, bootstrap 95% CI excludes the null.
- **REFUTED**: claim fails with CI excluding the stated margin. Published identically.
- **INCONCLUSIVE**: CI spans the margin (task count too small / variance too high), or a gate (G4) failed upstream.

## Pre-registered analysis notes

- P0 runs twice (E1 + E5) to quantify run-to-run variance at temperature 0
  before any policy delta is interpreted (temperature 0 is not perfectly
  deterministic under batching — measure, don't assume).
- The 4k tier is the sanity anchor: G4 requires P0 >= 25% EM there before
  further quota is spent.
- Policy execution order is fixed (P0, P4, P5, P3, P2, P1, P6) so a quota
  death mid-matrix leaves a publishable prefix.
