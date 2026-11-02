"""E2 — Mode-R replay matrix (build-prompt sections 2, 8).

Inputs: E1 recorded traces (P0). For each recorded llm_request, re-serve the
EXACT messages verbatim (max_tokens=1) under:
  {prefix caching on/off} x {affinity, round_robin} x {concurrency 1,2,4}
LMCache CPU-offload arms run as a separate server instance (env + kv config).

Writes results/replay_records.jsonl (one row per request) — Mode-R's raw data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentflight.replay.client import ReplayClient
from agentflight.replay.scheduler import ReplayRequest, run_replay
from agentflight.replay.server import ServerConfig, ServerManager
from agentflight.traces.writer import read_trace


def requests_from_trace(path: Path, max_requests: int | None = None) -> list[ReplayRequest]:
    """Extract llm_request message views, in order, from one trace."""
    _, events = read_trace(path)
    reqs = []
    for e in events:
        if e.kind == "llm_request" and e.payload.messages:
            reqs.append(ReplayRequest(session_id=path.stem, seq_in_session=len(reqs), messages=e.payload.messages))
            if max_requests and len(reqs) >= max_requests:
                break
    return reqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/")
    ap.add_argument("--results", default="results/")
    ap.add_argument("--model", default="Qwen/Qwen3-8B-GPTQ-Int4")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-requests-per-trace", type=int, default=None)
    ap.add_argument("--modes", default="affinity,round_robin")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--prefix-caching", default="on,off")
    ap.add_argument("--lmcache-offload", action="store_true", help="also run LMCache CPU-offload arms")
    args = ap.parse_args()

    trace_paths = sorted(Path(args.traces).glob("modeq-P0-*.json.gz")) or sorted(
        Path(args.traces).glob("*.json.gz")
    )
    if not trace_paths:
        raise SystemExit("no traces found — run E1 (agentflight record) first")
    requests: list[ReplayRequest] = []
    for p in trace_paths:
        requests.extend(requests_from_trace(p, args.max_requests_per_trace))
    print(f"loaded {len(requests)} requests from {len(trace_paths)} traces")

    out_path = Path(args.results) / "replay_records.jsonl"
    modes = args.modes.split(",")
    concs = [int(c) for c in args.concurrency.split(",")]
    pc_flags = [f == "on" for f in args.prefix_caching.split(",")]

    for pc in pc_flags:
        for lmcache in ([False, True] if args.lmcache_offload else [False]):
            cfg = ServerConfig(
                model_id=args.model,
                max_model_len=args.max_model_len,
                enable_prefix_caching=pc,
                lmcache_offload=lmcache,
            )
            condition = f"pc={'on' if pc else 'off'},lmcache={'on' if lmcache else 'off'}"
            t0 = time.perf_counter()
            with ServerManager(cfg) as _:
                client = ReplayClient(cfg)
                for mode in modes:
                    for conc in concs:
                        records = run_replay(client, requests, mode, conc, condition=f"{condition},sched={mode},conc={conc}")
                        with open(out_path, "a", encoding="utf-8") as f:
                            for r in records:
                                f.write(json.dumps(r.__dict__) + "\n")
                        hit = sum(r.cached_tokens for r in records) / max(1, sum(r.prompt_tokens for r in records))
                        print(
                            f"{condition} sched={mode} conc={conc}: "
                            f"n={len(records)} cache_hit={hit:.3f} "
                            f"({time.perf_counter() - t0:.0f}s elapsed)"
                        )
    print(f"records appended to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
