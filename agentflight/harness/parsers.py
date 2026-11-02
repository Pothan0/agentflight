"""Tool-call parsing.

Primary path: native OpenAI function calling (vLLM served with
``--enable-auto-tool-choice --tool-call-parser hermes`` for Qwen3) — the
server returns structured ``tool_calls`` and nothing is parsed client-side.

Fallback path (config flag, build-prompt section 6): the model emits
``<tool_call>{...}</tool_call>`` or a ```json``` block, parsed here. Kept
behind ``parse_mode="json_fallback"`` and verified on day 0.
"""

from __future__ import annotations

import json
import re

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_json_fallback(completion_text: str) -> tuple[list[dict], str]:
    """Extract tool calls from <tool_call> tags or a ```json``` block.

    Returns (tool_calls, remaining_visible_text). Tool-call spans are removed
    from the visible text.
    """
    calls: list[dict] = []
    visible = completion_text
    for regex in (_TOOL_CALL_TAG_RE, _JSON_BLOCK_RE):
        for m in regex.finditer(visible):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name, args = _extract_name_args(obj)
            if name:
                calls.append({"name": name, "arguments": args})
        if calls:
            visible = regex.sub("", visible)
            break
    return calls, visible.strip()


def _extract_name_args(obj: dict) -> tuple[str, dict]:
    name = obj.get("name") or obj.get("function", {}).get("name")
    args = obj.get("arguments", obj.get("parameters", obj.get("function", {}).get("arguments", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return (name or "", args)


def normalize_server_tool_calls(server_tool_calls: list) -> list[dict]:
    """OpenAI-SDK tool_call objects -> canonical dicts (args as dict)."""
    out = []
    for tc in server_tool_calls or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        raw_args = fn.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        out.append({"id": tc.id, "name": fn.name, "arguments": args})
    return out


def split_think(text: str) -> tuple[str, str]:
    """Split a completion into (think_text, visible_text) for Qwen3-style
    <think>...</think> blocks. Handles unterminated think blocks too."""
    m = re.match(r"^\s*<think>(.*?)</think>\s*(.*)$", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if text.lstrip().startswith("<think>"):  # truncated think block
        body = text.lstrip()[len("<think>") :]
        return body.strip(), ""
    return "", text
