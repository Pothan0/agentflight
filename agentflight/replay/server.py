"""vLLM server manager for Mode-R replay.

Launches ``vllm serve`` as a subprocess with the condition under test:
- prefix caching on/off        (--enable-prefix-caching / --no-enable-prefix-caching)
- LMCache CPU-offload on/off   (env vars + --kv-transfer-config; Kaggle CPU tier)
- tensor parallel size, model id, max-model-len (T4-verified defaults)

Waits on /health before returning. Prometaus /metrics gauges are scraped for
cross-checking usage.cached_tokens (build-prompt section 4: reconcile
discrepancies explicitly).
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "Qwen/Qwen3-8B-GPTQ-Int4"
DEFAULT_PORT = 8000


class ServerConfig:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        port: int = DEFAULT_PORT,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        enable_prefix_caching: bool = True,
        lmcache_offload: bool = False,
        extra_args: list[str] | None = None,
        tool_call_parser: str = "hermes",
        enable_auto_tool_choice: bool = True,
    ):
        self.model_id = model_id
        self.port = port
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.lmcache_offload = lmcache_offload
        self.extra_args = extra_args or []
        self.tool_call_parser = tool_call_parser
        self.enable_auto_tool_choice = enable_auto_tool_choice

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def argv(self) -> list[str]:
        args = [
            "vllm",
            "serve",
            self.model_id,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--enable-prefix-caching" if self.enable_prefix_caching else "--no-enable-prefix-caching",
        ]
        if self.enable_auto_tool_choice:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", self.tool_call_parser]
        if self.lmcache_offload:
            # CPU-offload tier via LMCache (the Kaggle-relevant experiment).
            args += [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LMCacheConnectorV1",
                        "kv_role": "kv_both",
                    }
                ),
            ]
        args += self.extra_args
        return args

    def env(self, base_env: dict | None = None) -> dict:
        import os

        env = dict(base_env if base_env is not None else os.environ)
        if self.lmcache_offload:
            env.update(
                {
                    "LMCACHE_USE_EXPERIMENTAL": "True",
                    "LMCACHE_REMOTE_URL": f"file:///tmp/lmcache-{self.port}",
                }
            )
        return env


class ServerManager:
    def __init__(self, config: ServerConfig, health_timeout_s: float = 1200.0):
        self.config = config
        self.health_timeout_s = health_timeout_s
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.proc is not None:
            raise RuntimeError("server already running")
        self.proc = subprocess.Popen(
            self.config.argv(),
            env=self.config.env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + self.health_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.config.port}/health", timeout=2.0) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(3.0)
        self.stop()
        raise TimeoutError("vllm /health not OK before health_timeout_s")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def __enter__(self) -> ServerManager:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def metrics(self) -> dict:
        """Scrape the /metrics gauges relevant to prefix caching."""
        text = urllib.request.urlopen(
            f"http://127.0.0.1:{self.config.port}/metrics", timeout=10.0
        ).read().decode("utf-8")
        gauges: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            key, _, value = line.partition(" ")
            if any(s in key for s in ("cache_queries", "cache_hits", "prefix_cache")):
                try:
                    gauges[key] = float(value)
                except ValueError:
                    pass
        return gauges
