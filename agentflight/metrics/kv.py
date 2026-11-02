"""KV-cache byte accounting.

The per-token KV footprint is derived PROGRAMMATICALLY from the served model's
config.json (build-prompt section 7: "never hardcode"). For GQA models:

    kv_bytes_per_token = 2 (K and V) * num_hidden_layers * num_key_value_heads
                         * head_dim * dtype_bytes

Qwen3-8B (36 layers, 8 KV heads, head_dim 128, fp16 KV): 2*36*8*128*2 = 147,456 B
= 144 KiB/token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Supported KV cache dtypes and their byte width. fp8 is intentionally absent:
# it requires sm_89+ and cannot run on T4 (build-prompt section 1).
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class KVConstants:
    """Per-model KV constants resolved from a HF config.json."""

    model_id: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * DTYPE_BYTES[self.dtype]

    @property
    def kv_bytes_per_kib(self) -> float:
        return self.kv_bytes_per_token / 1024.0

    def kv_bytes(self, n_tokens: int) -> int:
        return n_tokens * self.kv_bytes_per_token

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }


def kv_constants_from_config(config: dict, model_id: str = "unknown", dtype: str = "float16") -> KVConstants:
    """Build KVConstants from a parsed HF config.json dict.

    head_dim may be absent on some configs; it is then derived from
    hidden_size / num_attention_heads (standard MHA/GQA relation).
    """
    num_layers = config.get("num_hidden_layers")
    kv_heads = config.get("num_key_value_heads") or config.get("num_attention_heads")
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden = config.get("hidden_size")
        heads = config.get("num_attention_heads")
        if hidden is None or heads is None:
            raise ValueError("config.json lacks head_dim and cannot derive it (no hidden_size/num_attention_heads)")
        head_dim = hidden // heads
    if num_layers is None or kv_heads is None:
        raise ValueError("config.json lacks num_hidden_layers or num_key_value_heads")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported KV dtype {dtype!r}; supported: {sorted(DTYPE_BYTES)}")
    return KVConstants(
        model_id=model_id,
        num_hidden_layers=int(num_layers),
        num_key_value_heads=int(kv_heads),
        head_dim=int(head_dim),
        dtype=dtype,
    )


def kv_constants_from_file(config_path: str | Path, dtype: str = "float16") -> KVConstants:
    p = Path(config_path)
    with open(p, encoding="utf-8") as f:
        config = json.load(f)
    return kv_constants_from_config(config, model_id=p.parent.name, dtype=dtype)


def kv_bytes_touched(prompt_tokens: int, cached_tokens: int, kv: KVConstants) -> int:
    """KV bytes actually (re)processed for one request = uncached prompt tokens
    * per-token constant. Cached prefix tokens reuse KV already resident."""
    uncached = max(0, prompt_tokens - cached_tokens)
    return kv.kv_bytes(uncached)
