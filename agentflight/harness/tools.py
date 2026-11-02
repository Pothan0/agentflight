"""Tool registry — exactly 3 tools (build-prompt section 6):

1. search(query) -> top-5 passages (BM25 over the fixed local corpus)
2. read(doc_id) -> full passage text
3. calculator(expr) -> float

rank_bm25, offline, deterministic (ties broken by doc_id order).
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from agentflight.tasks.schema import Passage

_TOP_K = 5

_SEARCH_DOC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the local passage corpus. Returns the top-5 matching passages "
        "(doc_id, title, and a text excerpt).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}
_READ_DOC = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read the full text of a passage by doc_id (as returned by search).",
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
}
_CALC_DOC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(3 + 4) * 2.5'.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
}

TOOL_SCHEMAS: list[dict] = [_SEARCH_DOC, _READ_DOC, _CALC_DOC]


def tool_schema_hash(schemas: list[dict] | None = None) -> str:
    import hashlib
    import json

    blob = json.dumps(schemas or TOOL_SCHEMAS, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- calculator: AST-whitelisted arithmetic (no eval of arbitrary code) ---

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calc(expr: str) -> float:
    node = ast.parse(expr.strip(), mode="eval")

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(n.op)](ev(n.operand))
        raise ValueError(f"calculator: unsupported syntax {ast.dump(n)[:60]}")

    return ev(node)


@dataclass
class ToolResult:
    ok: bool
    output: str


class ToolRegistry:
    """Executes the 3 tools against the task's corpus view."""

    def __init__(self, corpus: list[Passage]):
        self._by_id = {p.doc_id: p for p in corpus}
        ordered = sorted(corpus, key=lambda p: p.doc_id)  # deterministic tie-break
        self._corpus_tokens = [_tokenize(p.title + " " + p.text) for p in ordered]
        self._ordered = ordered
        if ordered:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "search":
                return ToolResult(True, self._search(args["query"]))
            if name == "read":
                return ToolResult(True, self._read(args["doc_id"]))
            if name == "calculator":
                return ToolResult(True, repr(safe_calc(args["expr"])))
            return ToolResult(False, f"error: unknown tool {name!r}")
        except KeyError as e:
            if name == "read" and "doc_id" in args:
                return ToolResult(False, f"error: unknown doc_id {e.args[0]!r}")
            return ToolResult(False, f"error: missing argument {e}")
        except Exception as e:
            return ToolResult(False, f"error: {type(e).__name__}: {e}")

    def _search(self, query: str) -> str:
        if not self._bm25:
            raise RuntimeError("empty corpus")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self._ordered)), key=lambda i: (-scores[i], self._ordered[i].doc_id))
        top = ranked[:_TOP_K]
        lines = []
        for i in top:
            p = self._ordered[i]
            excerpt = p.text[:300]
            lines.append(f"doc_id={p.doc_id} | title={p.title} | score={scores[i]:.2f}\n{excerpt}")
        return "\n\n".join(lines) if lines else "no results"

    def _read(self, doc_id: str) -> str:
        p = self._by_id.get(doc_id)
        if p is None:
            raise KeyError(doc_id)
        return f"[{p.doc_id}] {p.title}\n{p.text}"


_TOKEN_RE = __import__("re").compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def tool_executor(corpus: list[Passage]) -> Callable[[str, dict], ToolResult]:
    return ToolRegistry(corpus).execute
