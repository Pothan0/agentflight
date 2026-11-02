"""The policy matrix (P0–P6). Registered by ID; applied to the same recorded
task set with fixed seeds (build-prompt section 5)."""

from agentflight.policies.base import ContextPolicy, PolicyConfig, PolicyResult, Summarizer
from agentflight.policies.p0_keep_all import KeepEverythingPolicy
from agentflight.policies.p1_sliding_window import SlidingWindowPolicy
from agentflight.policies.p2_summarize import SummarizeCompactionPolicy
from agentflight.policies.p3_chunked import ChunkedCompactionPolicy
from agentflight.policies.p4_think_discard import ThinkDiscardPolicy
from agentflight.policies.p5_tool_elision import ToolResultElisionPolicy
from agentflight.policies.p6_recompact import ReCompactionCascadePolicy

POLICIES: dict[str, type[ContextPolicy]] = {
    "P0": KeepEverythingPolicy,
    "P1": SlidingWindowPolicy,
    "P2": SummarizeCompactionPolicy,
    "P3": ChunkedCompactionPolicy,
    "P4": ThinkDiscardPolicy,
    "P5": ToolResultElisionPolicy,
    "P6": ReCompactionCascadePolicy,
}

# Fixed execution order for Mode-Q matrices: cheap core policies first so a
# quota death mid-matrix still leaves a publishable prefix (section 8).
POLICY_RUN_ORDER = ["P0", "P4", "P5", "P3", "P2", "P1", "P6"]


def get_policy(policy_id: str) -> ContextPolicy:
    try:
        return POLICIES[policy_id]()
    except KeyError:
        raise ValueError(f"unknown policy {policy_id!r}; known: {sorted(POLICIES)}") from None


__all__ = [
    "POLICIES",
    "POLICY_RUN_ORDER",
    "ContextPolicy",
    "PolicyConfig",
    "PolicyResult",
    "Summarizer",
    "get_policy",
]
