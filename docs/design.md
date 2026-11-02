# AgentFlight — design (condensed from the build brief)

## Two modes, one benchmark

- **Mode R (REPLAY)** — recorded token sequences re-served verbatim
  (`max_tokens=1`) under different serving configs. Measures TTFT, TPOT,
  throughput, cached_tokens / prefix-cache hit rate, KV-bytes touched,
  GPU-seconds. Serving axis: prefix caching on/off × session-affinity vs
  round-robin × LMCache CPU-offload.
- **Mode Q (RE-EXECUTION)** — the agent re-runs from scratch; a context
  policy (P0–P6) deterministically transforms the history at every turn.
  Measures EM/F1, turns-to-answer, abandonment, prompt-token efficiency,
  GPU-seconds/task. Policy axis.

**Bridge:** each policy decision is recorded with token counts (what was
dropped/replaced), mapping to a simulated KV footprint; Mode R measures what
those footprints do to real serving metrics. Headline artifact: the joint
accuracy × cost curve per policy per tier.

## Why separate modes

Replay can't change accuracy (same tokens → same-ish output); re-execution
can't isolate serving effects (no server reconfig mid-trajectory). Separating
the axes keeps every claim clean.

## Hardware reality (T4 ×2)

- vLLM runs on T4 (PagedAttention needs ≥7.0); Marlin/AWQ and fp8-KV need
  sm_80/89+ → **GPTQ-Int4 models only**, fp16 KV, no AWQ, no fp8.
- SGLang is out (Ampere+) → configs only, labeled UNTESTED.
- 30 GPU-h/week quota, 9-h session cap → adaptive controller + per-task
  checkpointing + `--skip-existing` resume everywhere.

## Metrics

Exact definitions in `agentflight/metrics/` (pure functions, unit-tested
against hand-computed fixtures): TTFT, TPOT, KV bytes/token derived from the
model's config.json (Qwen3-8B = 147,456 B/token), prefix-cache hit rate,
cost-per-completed-task, EM + token-F1, abandonment, bootstrap CIs (10k,
fixed seed). P0 runs twice to quantify run-to-run variance at temperature 0.

## Policies (P0–P6)

Pure functions: `(history, config) → view`. Purity is unit-tested. P2/P3/P6
inject a summarizer (same model, temperature 0, calls recorded + costed).
System + task message always preserved (the question itself never compacted
— the noise lives in aging tool traffic). Fixed run order P0→P4→P5→P3→P2→
P1→P6 so quota death mid-matrix leaves a publishable prefix.

## Verification gates

G1 CPU unit tests → G2 Kaggle smoke (<1 GPU-h) → G3 trace validation →
G4 P0 ≥ 25% EM @4k before further quota → G5 every reported number generated
by `agentflight report`.
