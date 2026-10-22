"""Portable episode artifacts and honest evaluation metrics (no GPU imports)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .types import Episode


def write_episodes(path: str | Path, episodes: Iterable[Episode], *, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "x", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


def read_episodes(path: str | Path) -> list[Episode]:
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                result.append(Episode.from_dict(json.loads(line)))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid episode at {path}:{lineno}: {exc}") from exc
    return result


def summarize(episodes: Iterable[Episode]) -> dict:
    episodes = list(episodes)
    # Evaluation-only policies are allowed; infrastructure failures are reported
    # separately instead of silently counted as either solved or policy failures.
    valid = [e for e in episodes if e.infrastructure_error is None]
    submitted = [e for e in valid if e.submitted]
    attempts = [a for e in valid for a in e.attempts]
    steps = [s for e in valid for s in e.steps]
    failures = Counter(a.error_type for a in attempts if a.error_type)
    groups: dict[tuple, list[Episode]] = defaultdict(list)
    for episode in valid:
        groups[(episode.group_id, episode.task_id, episode.seed,
                episode.initial_state_hash, episode.policy_version)].append(episode)
    complete_groups = [g for g in groups.values() if len(g) > 1]
    component_names = sorted({k for e in valid for k in e.reward_components})
    return {
        "episodes": len(episodes),
        "valid_episodes": len(valid),
        "infrastructure_failures": len(episodes) - len(valid),
        "trainable_episodes": sum(e.trainable for e in episodes),
        "success_rate": mean(e.success for e in valid) if valid else None,
        "submission_rate": len(submitted) / len(valid) if valid else None,
        "submission_success_rate": mean(e.success for e in submitted) if submitted else None,
        "incorrect_finish_rate": sum(e.terminal_reason == "incorrect_finish" for e in valid) / len(valid) if valid else None,
        "mean_reward": mean(e.reward for e in valid) if valid else None,
        "mean_steps": mean(len(e.steps) for e in valid) if valid else None,
        "mean_final_progress": mean(e.steps[-1].progress if e.steps else 0.0 for e in valid) if valid else None,
        "attempts": len(attempts),
        "rejected_attempt_rate": sum(not a.accepted for a in attempts) / len(attempts) if attempts else 0.0,
        "repeat_block_rate": failures["repeat"] / len(attempts) if attempts else 0.0,
        "invalid_action_rate": failures["invalid_action"] / len(attempts) if attempts else 0.0,
        "execution_error_rate": sum(bool(s.info.get("action_error") or s.info.get("last_action_error") or s.info.get("error")) for s in steps) / len(steps) if steps else 0.0,
        "fallback_rate": sum(s.fallback for s in steps) / len(steps) if steps else 0.0,
        "generation_token_count": sum(len(a.generation.response_token_ids) for a in attempts),
        "synthetic_token_ids": any(a.generation.metadata.get("synthetic_token_ids", False) for a in attempts),
        "errors": dict(failures),
        "terminal_reasons": dict(Counter(e.terminal_reason for e in episodes)),
        "policy_versions": sorted({e.policy_version for e in episodes if e.policy_version}),
        "harness_fingerprints": sorted({e.metadata["harness_fingerprint"] for e in episodes if e.metadata.get("harness_fingerprint")}),
        "mean_reward_components": {k: mean(e.reward_components.get(k, 0.0) for e in valid) for k in component_names},
        "zero_variance_group_fraction": (
            mean(max(e.reward for e in g) == min(e.reward for e in g) for g in complete_groups)
            if complete_groups else None
        ),
        "all_failure_group_fraction": mean(not any(e.success for e in g) for g in complete_groups) if complete_groups else None,
    }


def write_report(path: str | Path, episodes: Iterable[Episode]) -> dict:
    report = summarize(episodes)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
