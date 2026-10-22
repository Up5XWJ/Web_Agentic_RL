"""Independent environment episodes, concurrent sampling and same-state groups."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Callable
from uuid import uuid4

from .harness import ContextLimitError, HarnessConfig, MemoryHarness
from .scheduler import RolloutScheduler, SchedulerConfig
from .types import BatchPolicy, Environment, Episode, InfrastructureError, Policy, StepRecord, TaskSpec


RewardFunction = Callable[[Episode], float | None]


def _mark_infrastructure_error(episode: Episode, exc: Exception, *, phase: str) -> None:
    episode.trainable = False
    message = f"{phase}: {type(exc).__name__}: {exc}"
    episode.infrastructure_error = (
        f"{episode.infrastructure_error}; {message}" if episode.infrastructure_error else message
    )
    episode.terminal_reason = "context_limit" if isinstance(exc, ContextLimitError) else "infrastructure_error"
    episode.metadata.setdefault("infrastructure_errors", []).append({"phase": phase, "type": type(exc).__name__, "message": str(exc)})


def _score(episode: Episode, reward_fn: RewardFunction | None) -> None:
    if reward_fn is None:
        from .rewards import score_episode

        reward_fn = score_episode
    reward = reward_fn(episode)
    if reward is not None:
        episode.reward = float(reward)


def run_episode(
    task: TaskSpec,
    policy: Policy,
    environment: Environment,
    *,
    group_id: str = "",
    episode_id: str | None = None,
    harness_config: HarnessConfig | None = None,
    reward_fn: RewardFunction | None = None,
) -> Episode:
    """Run and close one environment; infrastructure failures are not policy loss.

    Each step contains zero or more rejected generations and at most one
    accepted generation. A fallback is only a StepRecord, never a Generation.
    """
    if task.max_steps < 1:
        try:
            environment.close()
        finally:
            raise ValueError("TaskSpec.max_steps must be positive.")
    episode = Episode(episode_id or uuid4().hex, group_id, task.task_id, task.seed)
    phase = "harness_initialization"
    try:
        harness = MemoryHarness(harness_config)
        episode.metadata.update(harness.metadata())
        episode.metadata["max_steps"] = task.max_steps
        episode.metadata["task_spec"] = deepcopy({
            "task_id": task.task_id, "family_id": task.family_id, "seed": task.seed,
            "goal": task.goal, "task_config": task.task_config, "max_steps": task.max_steps,
            "overrides": task.overrides, "progress_predicates": task.progress_predicates,
        })
        phase = "environment_reset"
        initial = environment.reset(deepcopy(task))
        if not isinstance(initial.initial_state_hash, str) or not initial.initial_state_hash:
            raise InfrastructureError("Environment reset must return a non-empty initial_state_hash.")
        episode.initial_state_hash = initial.initial_state_hash
        episode.metadata["reset_info"] = deepcopy(initial.info)
        observation = initial.observation
        harness.reset(task, observation)
        #Agent Loop
        for step_index in range(task.max_steps):
            phase = "policy_generation"
            decision = harness.select_action(policy, observation, episode, step_index)
            phase = "environment_step"
            result = environment.step(decision.browser_action)
            submitted = decision.action.name == "finish" and not decision.fallback
            if result.success and not submitted:
                raise InfrastructureError("Environment reported success without a finish submission")
            if submitted and (not result.terminated or result.truncated):
                raise InfrastructureError("finish must return a terminal, independently verified result")
            phase = "transition_assessment"
            assessment = harness.observe_transition(
                observation, result.observation, decision, step_index,
                submission_success=result.success if submitted else None,
            )
            at_horizon = step_index + 1 == task.max_steps and not (result.success or result.terminated)
            episode.steps.append(StepRecord(
                step_index=step_index, action=decision.action.to_dict(), browser_action=decision.browser_action,
                fallback=decision.fallback, assessment=assessment.to_dict(), success=result.success,
                progress=result.progress, terminated=result.terminated,
                truncated=result.truncated or at_horizon, info=deepcopy(result.info),
            ))
            observation = result.observation
            episode.success = result.success
            if result.success:
                episode.terminal_reason = "success"
                break
            if submitted:
                episode.terminal_reason = "incorrect_finish"
                break
            if result.terminated:
                episode.terminal_reason = "environment_terminated"
                break
            if result.truncated:
                episode.terminal_reason = "environment_truncated"
                break
            if at_horizon:
                episode.terminal_reason = "max_steps"
        # The final observation has been assessed above, rather than waiting for
        # a next get_action call that never occurs at episode termination.
    except Exception as exc:
        _mark_infrastructure_error(episode, exc, phase=phase)
    finally:
        try:
            environment.close()
        except Exception as exc:
            _mark_infrastructure_error(episode, exc, phase="environment_close")
    try:
        _score(episode, reward_fn)
    except Exception as exc:
        _mark_infrastructure_error(episode, exc, phase="reward_scoring")
    return episode


def sample_group(
    task: TaskSpec,
    policy: Policy,
    environment_factory: Callable[[], Environment],
    *,
    group_size: int,
    group_id: str | None = None,
    harness_config: HarnessConfig | None = None,
    reward_fn: RewardFunction | None = None,
) -> list[Episode]:
    """Sample a group without silently dropping failed or incompatible members.

    Every episode starts in its own environment. A broken member, state/version
    mismatch, or a scripted policy disqualifies the entire training group.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("GRPO group_size must be an integer >= 2.")
    actual_group_id = group_id or uuid4().hex
    episodes: list[Episode] = []
    for index in range(group_size):
        episode_id = f"{actual_group_id}:{index}"
        try:
            environment = environment_factory()
        except Exception as exc:
            episode = Episode(episode_id, actual_group_id, task.task_id, task.seed)
            _mark_infrastructure_error(episode, exc, phase="environment_creation")
        else:
            episode = run_episode(
                deepcopy(task), policy, environment, group_id=actual_group_id, episode_id=episode_id,
                harness_config=harness_config, reward_fn=reward_fn,
            )
        episodes.append(episode)

    validate_group(episodes, group_size)
    return episodes


def validate_group(episodes: list[Episode], group_size: int) -> None:
    """The same eligibility rules apply to serial and concurrent sampling."""
    if len(episodes) != group_size or len({ep.episode_id for ep in episodes}) != group_size:
        raise ValueError("Incomplete or duplicate rollout group")
    if len({(ep.group_id, ep.task_id, ep.seed) for ep in episodes}) != 1:
        raise ValueError("Rollout group mixes task identities or environment seeds")
    reasons: list[str] = []
    if any(not episode.trainable for episode in episodes):
        reasons.append("untrainable_member")
    if any(episode.infrastructure_error for episode in episodes):
        reasons.append("infrastructure_error")
    if len({episode.initial_state_hash for episode in episodes}) != 1 or not episodes[0].initial_state_hash:
        reasons.append("initial_state_mismatch")
    if len({episode.policy_version for episode in episodes}) != 1 or not episodes[0].policy_version:
        reasons.append("policy_version_mismatch")
    for episode in episodes:
        episode.metadata["group_size"] = group_size
        episode.metadata["group_eligible"] = not reasons
        if reasons:
            episode.metadata["group_skip_reasons"] = list(reasons)
            episode.trainable = False


@dataclass(frozen=True)
class RolloutJob:
    task: TaskSpec
    group_id: str
    episode_id: str


@dataclass
class RolloutBatch:
    episodes: list[Episode]
    metrics: dict[str, float]


class RolloutSamplingError(InfrastructureError):
    """A shared backend failure, with completed diagnostics safe to persist."""

    def __init__(self, message: str, batch: RolloutBatch):
        super().__init__(message)
        self.batch = batch


def sample_episodes(
    jobs: list[RolloutJob], policy: BatchPolicy, environment_factory: Callable[[], Environment], *,
    scheduler_config: SchedulerConfig | None = None, seed: int = 0,
    harness_config: HarnessConfig | None = None, reward_fn: RewardFunction | None = None,
) -> RolloutBatch:
    """Run bounded independent loops, joining all environments and inference.

    A thread owns each synchronous Harness. BrowserGym still spawns a fresh
    backend/browser process, so app globals and Playwright never share threads.
    No episode is dropped or selected according to its completion time.
    """
    config = scheduler_config or SchedulerConfig()
    if len({job.episode_id for job in jobs}) != len(jobs):
        raise ValueError("Concurrent sampling requires unique episode IDs")
    if not jobs:
        return RolloutBatch([], {})
    started = time.monotonic()
    lock = threading.Lock()
    active = peak = 0
    scheduler = RolloutScheduler(policy, config, seed=seed)
    episodes: list[Episode] = []

    def run(job: RolloutJob) -> Episode:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            try:
                environment = environment_factory()
            except Exception as exc:
                episode = Episode(job.episode_id, job.group_id, job.task.task_id, job.task.seed)
                _mark_infrastructure_error(episode, exc, phase="environment_creation")
                return episode
            return run_episode(
                deepcopy(job.task), scheduler.policy_for(job.episode_id), environment,
                group_id=job.group_id, episode_id=job.episode_id,
                harness_config=harness_config, reward_fn=reward_fn,
            )
        finally:
            with lock:
                active -= 1

    try:
        with scheduler:
            # Keep the dispatcher alive until every environment has closed.
            with ThreadPoolExecutor(max_workers=min(config.max_concurrent_episodes, len(jobs)),
                                    thread_name_prefix="rollout-env") as executor:
                futures = [executor.submit(run, job) for job in jobs]
                try:
                    for future in futures:
                        episodes.append(future.result())
                except BaseException as exc:
                    scheduler.abort(f"Rollout executor interrupted: {exc}")
                    for future in futures:
                        future.cancel()
                    raise
    except InfrastructureError as exc:
        for episode in episodes:
            episode.trainable = False
            episode.metadata["sampling_aborted"] = str(exc)
        failed = RolloutBatch(episodes, scheduler.metrics() | {"rollout/seconds": time.monotonic() - started})
        raise RolloutSamplingError(str(exc), failed) from exc
    elapsed = time.monotonic() - started
    return RolloutBatch(episodes, scheduler.metrics() | {
        "rollout/seconds": elapsed,
        "rollout/episodes": float(len(episodes)),
        "rollout/episodes_per_second": len(episodes) / max(elapsed, 1e-9),
        "rollout/peak_concurrent_episodes": float(peak),
        "rollout/generated_tokens_per_second": scheduler.metrics()["rollout/generated_tokens"] / max(elapsed, 1e-9),
    })


def sample_batch(
    groups: list[tuple[str, TaskSpec]], policy: BatchPolicy, environment_factory: Callable[[], Environment], *,
    group_size: int, scheduler_config: SchedulerConfig | None = None, seed: int = 0,
    harness_config: HarnessConfig | None = None, reward_fn: RewardFunction | None = None,
) -> RolloutBatch:
    """Sample m groups of n episodes with one executor and inference queue."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("GRPO group_size must be an integer >= 2")
    if len({group_id for group_id, _ in groups}) != len(groups):
        raise ValueError("Each task slot needs a unique group_id")
    # Interleave groups so a limited environment pool does not favor the first
    # task. Keep stable IDs, then restore group order independently of timing.
    jobs = [RolloutJob(task, group_id, f"{group_id}:{index}")
            for index in range(group_size) for group_id, task in groups]
    batch = sample_episodes(jobs, policy, environment_factory, scheduler_config=scheduler_config,
                            seed=seed, harness_config=harness_config, reward_fn=reward_fn)
    by_id = {episode.episode_id: episode for episode in batch.episodes}
    batch.episodes = []
    for group_id, _ in groups:
        members = [by_id[f"{group_id}:{index}"] for index in range(group_size)]
        validate_group(members, group_size)
        batch.episodes.extend(members)
    batch.metrics["rollout/groups"] = float(len(groups))
    batch.metrics["rollout/eligible_episodes_per_second"] = (
        sum(ep.trainable for ep in batch.episodes) / max(batch.metrics.get("rollout/seconds", 0), 1e-9)
    )
    return batch
