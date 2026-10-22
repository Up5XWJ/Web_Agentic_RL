"""Structured ReAct + overwrite-memory agent for OpenApps."""

from .memory import MemoryConfig, MemoryManager
from .schemas import AgentResponse, MemoryBlock


def __getattr__(name):
    # The reusable parser/memory components do not need an inference client.
    if name in {"AgentArgs", "ReactMemoryAgent"}:
        from .agent import AgentArgs, ReactMemoryAgent

        return {"AgentArgs": AgentArgs, "ReactMemoryAgent": ReactMemoryAgent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentArgs",
    "ReactMemoryAgent",
    "MemoryConfig",
    "MemoryManager",
    "AgentResponse",
    "MemoryBlock",
]
