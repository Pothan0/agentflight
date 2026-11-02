"""Token counting.

Two implementations behind one interface:

- ``heuristic``: deterministic regex word/punctuation split. Available
  everywhere (CPU tests, policy purity checks, CI). It is NOT the model's
  tokenizer; token counts from it are internally consistent but not equal to
  vLLM's counts.
- ``hf``: the served model's tokenizer via ``transformers.AutoTokenizer``,
  used on the GPU host so policy budgets operate in true model tokens.

Every component counts tokens ONLY through this module, so swapping the
backend never changes policy logic. The backend in force is recorded in
results/ENVIRONMENT.json.
"""

from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_HF_MODEL_ALIASES = {
    # Reasonable tokenizer substitutions when the exact GPTQ repo has no
    # tokenizer files: same Qwen3 family/tokenizer.
    "Qwen/Qwen3-8B-GPTQ-Int4": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B-GPTQ-Int4": "Qwen/Qwen3-4B",
}


@lru_cache(maxsize=4)
def _hf_tokenizer(model_id: str):
    from transformers import AutoTokenizer  # imported lazily: GPU-only dep

    target = _HF_MODEL_ALIASES.get(model_id, model_id)
    return AutoTokenizer.from_pretrained(target)


def count_tokens(text: str, backend: str = "heuristic", model_id: str | None = None) -> int:
    """Count tokens of ``text``.

    backend="heuristic" needs no dependencies; backend="hf" loads the model
    tokenizer (requires transformers, GPU host).
    """
    if not text:
        return 0
    if backend == "heuristic":
        return len(_TOKEN_RE.findall(text))
    if backend == "hf":
        if model_id is None:
            raise ValueError("backend='hf' requires model_id")
        tok = _hf_tokenizer(model_id)
        return len(tok.encode(text))
    raise ValueError(f"unknown tokenizer backend {backend!r}")


def count_message_tokens(messages: list[dict], backend: str = "heuristic", model_id: str | None = None) -> int:
    """Approximate total tokens of an OpenAI-format message list.

    Sums per-message content tokens plus a small per-message overhead
    constant (role + framing), mirroring how server-side counting includes
    chat-template framing. The constant is fixed and deterministic; exactness
    against the server is verified in E0 (determinism probe), not assumed.
    """
    total = 0
    for m in messages:
        total += 4  # role/name/framing overhead
        for part in _message_text_parts(m):
            total += count_tokens(part, backend=backend, model_id=model_id)
    total += 2  # primer
    return total


def _message_text_parts(message: dict) -> list[str]:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):  # multimodal-style content parts
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(str(fn.get("name", "")))
        parts.append(str(fn.get("arguments", "")))
    return parts
