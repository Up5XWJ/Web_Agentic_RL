"""Serializable contracts shared by environment, harness and trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class InfrastructureError(RuntimeError):
    """A broken environment or inference service, not a policy failure."""


class ContextLimitError(InfrastructureError):
    """The exact context does not fit; never silently truncate a recorded prompt."""


@dataclass
class TaskSpec:
    task_id: str
    goal: str
    task_config: dict[str, Any]
    family_id: str = ""
    seed: int = 0
    max_steps: int = 20
    overrides: list[str] = field(default_factory=list)
    progress_predicates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResetResult:
    observation: dict[str, Any]
    initial_state_hash: str
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """success means a verified finish submission, not merely a satisfied state."""
    observation: dict[str, Any]
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    progress: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    old_log_probs: list[float]
    policy_version: str
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    """One actual Harness call, identified independently of scheduling order."""

    request_id: str
    messages: list[dict[str, Any]]
    sample_seed: int
    policy_version: str


class Policy(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> Generation: ...


class BatchPolicy(Protocol):
    policy_version: str

    def generate_batch(self, requests: list[GenerationRequest]) -> list[Generation | Exception]: ...


class Environment(Protocol):
    def reset(self, task: TaskSpec) -> ResetResult: ...
    def step(self, action: str) -> StepResult: ...
    def close(self) -> None: ...


@dataclass
class AttemptRecord:
    step_index: int
    attempt_index: int
    messages: list[dict[str, Any]]
    generation: Generation
    accepted: bool = False
    error_type: str | None = None
    error: str | None = None
    raw_memory: dict[str, Any] | None = None
    applied_memory: dict[str, Any] | None = None
    action: dict[str, Any] | None = None


@dataclass
class StepRecord:
    step_index: int
    action: dict[str, Any]
    browser_action: str
    fallback: bool
    assessment: dict[str, Any]
    success: bool
    progress: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    episode_id: str
    group_id: str
    task_id: str
    seed: int
    policy_version: str = ""
    initial_state_hash: str = ""
    attempts: list[AttemptRecord] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    terminal_reason: str = "unfinished"
    trainable: bool = True
    infrastructure_error: str | None = None
    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def submitted(self) -> bool:
        return bool(self.steps and self.steps[-1].action.get("name") == "finish"
                    and self.steps[-1].terminated and not self.steps[-1].fallback)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        data = dict(data)
        data["attempts"] = [
            AttemptRecord(**{**a, "generation": Generation(**a["generation"])})
            for a in data.get("attempts", [])
        ]
        data["steps"] = [StepRecord(**s) for s in data.get("steps", [])]
        return cls(**data)
