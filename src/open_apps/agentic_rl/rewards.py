"""Bounded episode rewards anchored to independently verified task success."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .types import Episode


@dataclass(frozen=True)
class RewardConfig:
    success_weight: float = 1.0
    progress_weight: float = 0.15
    format_penalty: float = 0.05
    invalid_penalty: float = 0.05
    repeat_penalty: float = 0.05
    execution_penalty: float = 0.05
    incorrect_finish_penalty: float = 1.0
    step_penalty: float = 0.03
    error_cap: int = 4
    default_step_budget: int = 30

    def __post_init__(self):
        values = {k: v for k, v in vars(self).items() if k not in {"error_cap", "default_step_budget"}}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Reward weights must be finite and non-negative")
        if self.error_cap < 1 or self.default_step_budget < 1:
            raise ValueError("Reward count caps/budgets must be positive")
        # Even the worst successful episode must outrank the best failure.
        # Incorrect-finish penalties apply only to failures, so increasing them
        # cannot weaken this ordering and must not constrain success_weight.
        auxiliary = sum(v for k, v in values.items()
                        if k not in {"success_weight", "incorrect_finish_penalty"})
        if self.success_weight <= auxiliary:
            raise ValueError("Success weight must exceed the auxiliary budget excluding incorrect-finish penalties")


def score_episode(episode: Episode, config: RewardConfig | None = None) -> float:
    config = config or RewardConfig()
    # Evaluation/scripted policies can be non-trainable and still receive a
    # useful score. Only broken rollouts are excluded from reward evaluation.
    if episode.infrastructure_error:
        episode.trainable = False
        episode.reward = 0.0
        episode.reward_components = {"excluded_infrastructure": 1.0}
        return 0.0
    attempts = episode.attempts
    formats = sum((a.error_type or "") in {"format", "schema", "parse"} for a in attempts)
    invalid = sum((a.error_type or "") in {"invalid_action", "guard", "parameter"} for a in attempts)
    repeats = sum((a.error_type or "") in {"repeat", "loop", "loop_guard"} for a in attempts)
    execution = sum(bool(s.info.get("action_error") or s.info.get("last_action_error") or s.info.get("error")) for s in episode.steps)
    # Reward only final retained progress. Undoing an effect removes its credit.
    progress = episode.steps[-1].progress if episode.steps else 0.0
    if not math.isfinite(progress):
        raise ValueError("Non-finite environment progress")
    progress = min(1.0, max(0.0, progress))
    max_steps = max(1, int(episode.metadata.get("max_steps", config.default_step_budget)))
    components = {
        "success": config.success_weight * bool(episode.success),
        "retained_progress": config.progress_weight * progress,
        # Fixed caps: adding valid noops cannot dilute previous errors.
        "format": -config.format_penalty * min(1.0, formats / config.error_cap),
        "invalid": -config.invalid_penalty * min(1.0, invalid / config.error_cap),
        "repeat": -config.repeat_penalty * min(1.0, repeats / config.error_cap),
        "execution": -config.execution_penalty * min(1.0, execution / config.error_cap),
        "incorrect_finish": -config.incorrect_finish_penalty * (episode.terminal_reason == "incorrect_finish"),
        "steps": -config.step_penalty * min(1.0, len(episode.steps) / max_steps),
    }
    episode.reward_components = components
    episode.reward = sum(components.values())
    return episode.reward
