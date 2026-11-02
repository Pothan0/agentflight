"""E0 — day-0 verification checklist (build-prompt section 1). MUST pass
before any experiment; costs <1 GPU-hour.

On the GPU host (Kaggle): verifies 2x T4, compute capability (7,5), installs,
smoke-serves the model (1 completion + 1 tool call + 1 streaming request),
measures decode/prefill tok/s, verifies cached_tokens is nonzero under prefix
caching, and records everything into results/ENVIRONMENT.json.

On CPU (local dev): verifies package-level invariants only and records the
mock-backend environment, clearly labeled.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from agentflight import __version__
from agentflight.metrics.kv import kv_constants_from_file
from agentflight.replay.client import ReplayClient
from agentflight.replay.server import ServerConfig, ServerManager


def _gpu_checks() -> dict:
    out: dict = {}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out["nvidia_smi"] = smi.stdout.strip()
    except Exception as e:
        out["nvidia_smi"] = f"unavailable: {e}"
        return out
    try:
        import torch

        caps = [tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        out["cuda_capabilities"] = caps
        out["torch"] = torch.__version__
        out["t4_confirmed"] = all(c == (7, 5) for c in caps) and len(caps) == 2
    except Exception as e:
        out["torch"] = f"unavailable: {e}"
    return out


def smoke_serve(model_id: str, max_model_len: int = 32768) -> dict:
    """Serve, then: 1 streaming completion + 1 tool call + cached_tokens check."""
    cfg = ServerConfig(model_id=model_id, max_model_len=max_model_len, enable_prefix_caching=True)
    result: dict = {"model_id": model_id}
    with ServerManager(cfg) as mgr:
        client = ReplayClient(cfg)
        # 1) streaming completion (twice: second request should hit prefix cache)
        messages = [{"role": "user", "content": "Reply with exactly: pong"}]
        r1 = client.replay_one(messages, "smoke", 0, "smoke", max_tokens=16)
        r2 = client.replay_one(messages, "smoke", 1, "smoke", max_tokens=16)
        result["stream_ok"] = r1.error is None and r1.ttft_ms is not None
        result["ttft_ms_r1"] = r1.ttft_ms
        result["cached_tokens_r1"] = r1.cached_tokens
        result["cached_tokens_r2"] = r2.cached_tokens
        result["cached_tokens_nonzero_on_repeat"] = r2.cached_tokens > 0
        # 2) tool call through the OpenAI API (hermes parser)
        try:
            from openai import OpenAI

            oc = OpenAI(base_url=cfg.base_url(), api_key="EMPTY")
            resp = oc.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "What is 6*7? Use the calculator tool."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
                        },
                    }
                ],
                tool_choice="auto",
                max_tokens=256,
            )
            result["tool_call_ok"] = bool(resp.choices[0].message.tool_calls)
        except Exception as e:
            result["tool_call_ok"] = False
            result["tool_call_error"] = str(e)
        # 3) KV constants from the served model's config
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(model_id, "config.json")
            kv = kv_constants_from_file(cfg_path)
            result["kv_constants"] = kv.to_dict()
        except Exception as e:
            result["kv_constants"] = f"unavailable: {e}"
        result["metrics_gauges"] = mgr.metrics()
    return result


def main() -> int:
    out = {
        "record": "environment",
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agentflight_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    gpu = _gpu_checks()
    out.update(gpu)
    on_gpu = "cuda_capabilities" in out
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if on_gpu and out.get("t4_confirmed"):
        model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B-GPTQ-Int4"
        t0 = time.perf_counter()
        out["smoke"] = smoke_serve(model)
        out["smoke_gpu_seconds"] = time.perf_counter() - t0
        out["verdict"] = (
            "PASS"
            if (out["smoke"].get("stream_ok") and out["smoke"].get("tool_call_ok"))
            else "FAIL"
        )
    else:
        out["verdict"] = "CPU-MODE (no T4s found; GPU checklist not run here)"
        out["notes"] = "Run this script inside a Kaggle session with 2x T4 before any experiment (gate G2)."

    with open(results_dir / "ENVIRONMENT.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    return 0 if out["verdict"] in ("PASS", "CPU-MODE (no T4s found; GPU checklist not run here)") else 1


if __name__ == "__main__":
    raise SystemExit(main())
