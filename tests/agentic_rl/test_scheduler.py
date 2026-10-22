"""Concurrency contracts with actual Harness/GRPO logic and CPU test doubles."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading

import pytest

from open_apps.agentic_rl.rollout import RolloutJob, sample_batch, sample_episodes
from open_apps.agentic_rl.scheduler import RolloutScheduler, SchedulerConfig
from open_apps.agentic_rl.train import run_online_loop
from open_apps.agentic_rl.training_data import prepare_training_batch
from open_apps.agentic_rl.types import ContextLimitError, Generation, InfrastructureError, ResetResult, StepResult, TaskSpec


def observation(saved=False):
    return {"url": "http://localhost/todo", "pruned_html": '<button bid="1">Save</button>' + str(saved),
            "axtree_txt": "[1] button Save", "last_action_error": "", "focused_element_bid": "",
            "extra_element_properties": {"1": {"visibility": 1.0}}}


def response(messages, action):
    prompt = messages[1]["content"]
    marker = re.search(r'"automatic_assessment"\s*:\s*', prompt)
    assessment, _ = json.JSONDecoder().raw_decode(prompt[marker.end():])
    return json.dumps({
        "memory_next": {"plan": [{"id": "s", "description": "Save", "status": "in_progress"}],
                        "current_subgoal_id": "s"},
        "previous_action_assessment": assessment,
        "decision": {"observation": "Save", "current_subgoal": "Save", "reason": "Save task", "expected_result": "Saved"},
        "action": action,
    })


class BatchedPolicy:
    policy_version = "update-0"

    def __init__(self, *, row_error=None, corrupt_identity=False):
        self.sample_counter = 0
        self.requests = []
        self.batch_sizes = []
        self.thread_ids = set()
        self.row_error = row_error
        self.corrupt_identity = corrupt_identity
        self.in_call = False

    def generate_batch(self, requests):
        assert not self.in_call, "Shared GPU calls must not overlap"
        self.in_call = True
        try:
            self.thread_ids.add(threading.get_ident())
            self.batch_sizes.append(len(requests))
            self.sample_counter += len(requests)
            result = []
            for request in requests:
                self.requests.append(request)
                if self.row_error and self.row_error in request.request_id:
                    result.append(ContextLimitError("context budget exceeded"))
                    continue
                # Rollout 0 retries invalid JSON, changes the environment, then
                # observes and submits. Rollout 1 submits incorrectly at once.
                episode_id, call = request.request_id.rsplit("/call-", 1)
                call = int(call)
                winner = episode_id.endswith(":0")
                if winner and call == 0:
                    text = "invalid JSON retained for training"
                else:
                    action = {"name": "click", "bid": "1"} if winner and call == 1 else {"name": "finish"}
                    text = response(request.messages, action)
                result.append(Generation(text, [11, 12], [20 + call, 77], [-0.2, -0.3], self.policy_version,
                                         metadata={"request_id": "wrong" if self.corrupt_identity else request.request_id,
                                                   "sample_seed": request.sample_seed}))
            return result
        finally:
            self.in_call = False


class Tracker:
    def __init__(self, barrier=None, fail_task=None):
        self.lock = threading.Lock()
        self.barrier = barrier
        self.fail_task = fail_task
        self.failed = False
        self.envs = []
        self.active = self.peak = 0

    def factory(self):
        tracker = self

        class Environment:
            def __init__(self):
                self.saved = self.closed = False
                self.actions = []
                with tracker.lock:
                    tracker.envs.append(self)
                    tracker.active += 1
                    tracker.peak = max(tracker.peak, tracker.active)

            def reset(self, task):
                self.task = task
                if tracker.barrier:
                    tracker.barrier.wait(timeout=10)
                with tracker.lock:
                    if tracker.fail_task == task.task_id and not tracker.failed:
                        tracker.failed = True
                        raise InfrastructureError("browser reset failed")
                return ResetResult(observation(), f"initial-{task.task_id}")

            def step(self, action):
                self.actions.append(action)
                if action.startswith("click"):
                    self.saved = True
                submitted = action == "finish()"
                return StepResult(observation(self.saved), success=submitted and self.saved,
                                  terminated=submitted, progress=float(self.saved))

            def close(self):
                if not self.closed:
                    with tracker.lock:
                        tracker.active -= 1
                    self.closed = True

        return Environment()


def groups():
    return [(name, TaskSpec(name, "Save", {}, seed=7, max_steps=3)) for name in ("a", "b")]


def reward(episode):
    # Different task reward scales expose accidental batch-wide normalization.
    return (1 if episode.success else 0) if episode.task_id == "a" else (10 if episode.success else 8)


def test_parallel_groups_batch_requests_keep_retries_and_compute_separate_advantages():
    tracker = Tracker(barrier=threading.Barrier(4))
    policy = BatchedPolicy()
    result = sample_batch(groups(), policy, tracker.factory, group_size=2, seed=17,
                          scheduler_config=SchedulerConfig(4, 4, 40), reward_fn=reward)
    assert tracker.peak == 4 and tracker.active == 0
    assert all(env.closed for env in tracker.envs)
    assert [ep.episode_id for ep in result.episodes] == ["a:0", "a:1", "b:0", "b:1"]
    assert all(ep.trainable for ep in result.episodes)
    assert [len(ep.attempts) for ep in result.episodes] == [3, 1, 3, 1]
    assert result.episodes[0].attempts[0].generation.text == "invalid JSON retained for training"
    assert result.episodes[0].attempts[0].error_type == "format"
    assert max(policy.batch_sizes) > 1
    assert len(policy.thread_ids) == 1  # One owner of veRL collectives.
    packed = prepare_training_batch(result.episodes, group_size=2, policy_version="update-0")
    assert [row.advantage for row in packed.rows] == [0.5, 0.5, 0.5, -0.5, 1, 1, 1, -1]
    assert len({r.sample_seed for r in policy.requests}) == len(policy.requests)
    assert result.metrics["rollout/peak_concurrent_episodes"] == 4
    assert result.metrics["rollout/inference_requests"] == 8


@pytest.mark.parametrize("failure", ["environment", "context"])
def test_one_failed_member_keeps_complete_groups_and_does_not_discard_other_task(failure):
    tracker = Tracker(fail_task="a" if failure == "environment" else None)
    policy = BatchedPolicy(row_error="a:0/" if failure == "context" else None)
    batch = sample_batch(groups(), policy, tracker.factory, group_size=2,
                         scheduler_config=SchedulerConfig(4, 4, 10), reward_fn=reward)
    assert len(batch.episodes) == 4
    assert [ep.trainable for ep in batch.episodes] == [False, False, True, True]
    assert tracker.active == 0 and all(env.closed for env in tracker.envs)
    packed = prepare_training_batch(batch.episodes, group_size=2, policy_version="update-0")
    assert {row.group_id for row in packed.rows} == {"b"}
    assert packed.metrics["train/skipped_groups"] == 1


def test_slow_environment_does_not_block_another_loop_or_require_a_full_batch():
    fast_finished = threading.Event()
    tracker = Tracker()

    def factory():
        env = tracker.factory()
        reset, close = env.reset, env.close

        def delayed_reset(task):
            if task.task_id == "slow":
                assert fast_finished.wait(5), "Another loop must progress before this reset completes"
            return reset(task)

        def signal_close():
            if env.task.task_id == "fast":
                fast_finished.set()
            close()

        env.reset, env.close = delayed_reset, signal_close
        return env

    jobs = [RolloutJob(TaskSpec(name, "Save", {}, max_steps=1), name, name + ":1") for name in ("slow", "fast")]
    policy = BatchedPolicy()
    sampled = sample_episodes(jobs, policy, factory, scheduler_config=SchedulerConfig(2, 8, 5))
    assert fast_finished.is_set() and all(ep.trainable for ep in sampled.episodes)
    assert policy.batch_sizes == [1, 1]  # Flushes even though B > active requests.
    assert tracker.active == 0


def test_concurrency_cap_and_stable_seeds_do_not_depend_on_job_order_or_batching():
    records = []
    for selected, config in ((groups(), SchedulerConfig(1, 1, 0)),
                             (list(reversed(groups())), SchedulerConfig(3, 3, 10))):
        policy, tracker = BatchedPolicy(), Tracker()
        batch = sample_batch(selected, policy, tracker.factory, group_size=2,
                             scheduler_config=config, seed=17, reward_fn=reward)
        assert tracker.peak <= config.max_concurrent_episodes
        assert all(ep.trainable for ep in batch.episodes)
        records.append({r.request_id: r.sample_seed for r in policy.requests})
    assert records[0] == records[1]


def test_wrong_model_result_identity_aborts_sampling_and_closes_all_environments():
    tracker = Tracker()
    with pytest.raises(InfrastructureError, match="identity"):
        sample_batch(groups(), BatchedPolicy(corrupt_identity=True), tracker.factory, group_size=2,
                     scheduler_config=SchedulerConfig(4, 4, 10))
    assert tracker.active == 0 and all(env.closed for env in tracker.envs)


def test_request_timeout_poisons_scheduler_and_cannot_cross_update_barrier():
    release = threading.Event()

    class StuckPolicy:
        policy_version = "v1"

        def generate_batch(self, requests):
            release.wait(5)
            return [InfrastructureError("stopped") for _ in requests]

    scheduler = RolloutScheduler(StuckPolicy(), SchedulerConfig(2, 2, 1, 0.1))
    scheduler.__enter__()
    try:
        with ThreadPoolExecutor(2) as executor:
            calls = [executor.submit(scheduler.policy_for(str(i)).generate, []) for i in range(2)]
            for call in calls:
                with pytest.raises(InfrastructureError, match="timed out"):
                    call.result(timeout=3)
        with pytest.raises(InfrastructureError, match="timed out"):
            scheduler.policy_for("later").generate([])
    finally:
        release.set()
    with pytest.raises(InfrastructureError, match="timed out"):
        scheduler.close()
    assert not scheduler._thread.is_alive()


class Backend:
    def __init__(self, tracker):
        self.tracker = tracker
        self.policy = BatchedPolicy()
        self.update_count = 0

    def update(self, rows):
        assert self.tracker.active == 0, "No policy update while any environment is active"
        assert not self.policy.in_call
        assert {row.policy_version for row in rows} == {self.policy.policy_version}
        self.update_count += 1
        self.policy.policy_version = f"update-{self.update_count}"
        return {"actor/test": 1.0}

    def state_dict(self):
        return {"update_count": self.update_count, "policy_version": self.policy.policy_version,
                "sample_counter": self.policy.sample_counter}

    def save_checkpoint(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, directory, state):
        self.update_count = state["update_count"]
        self.policy.policy_version = state["policy_version"]
        self.policy.sample_counter = state["sample_counter"]


def test_production_online_loop_uses_concurrent_sampling_and_restores_request_seeds(tmp_path):
    config = {"output_dir": str(tmp_path / "train"), "epochs": 2, "max_updates": 2,
              "group_size": 2, "tasks_per_update": 2, "seed": 17,
              "checkpoint_every": 1, "eval_every": 0, "grpo": {"normalize_std": False},
              "rollout_scheduler": {"max_concurrent_episodes": 4, "inference_batch_size": 4, "batch_wait_ms": 30}}
    tasks = [task for _, task in groups()]
    tracker = Tracker()
    backend = Backend(tracker)
    result = run_online_loop(config, tasks, backend, tracker.factory, reward_fn=reward)
    assert result["updates"] == 2 and result["batches"] == 2
    assert max(backend.policy.batch_sizes) > 1
    first_checkpoint = Path(config["output_dir"]) / "checkpoints/update-000001-batch-000001"
    restored_tracker = Tracker()
    restored = Backend(restored_tracker)
    resumed = config | {"output_dir": str(tmp_path / "resume"), "resume_from": str(first_checkpoint)}
    second = run_online_loop(resumed, tasks, restored, restored_tracker.factory, reward_fn=reward)
    assert second["updates"] == 2
    original_requests = {r.request_id: r.sample_seed for r in backend.policy.requests if r.policy_version == "update-1"}
    assert {r.request_id: r.sample_seed for r in restored.policy.requests} == original_requests
    assert restored.policy.sample_counter == backend.policy.sample_counter
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert all(record["rollout/inference_batches"] > 0 for record in records)


def test_fatal_inference_error_is_persisted_before_training_stops(tmp_path):
    config = {"output_dir": str(tmp_path / "train"), "epochs": 1, "max_updates": 1,
              "group_size": 2, "tasks_per_update": 2, "seed": 17,
              "checkpoint_every": 1, "eval_every": 0}
    tracker = Tracker()
    backend = Backend(tracker)
    backend.policy.corrupt_identity = True
    with pytest.raises(InfrastructureError, match="identity"):
        run_online_loop(config, [task for _, task in groups()], backend, tracker.factory)
    assert backend.update_count == 0 and tracker.active == 0
    directory = Path(config["output_dir"]) / "episodes"
    episodes = [json.loads(line) for line in (directory / "batch-000001.jsonl").read_text().splitlines()]
    assert len(episodes) == 4 and all(not episode["trainable"] for episode in episodes)
    assert "identity" in json.loads((directory / "batch-000001.failure.json").read_text())["error"]


@pytest.mark.parametrize("options", [
    {"max_concurrent_episodes": 0}, {"inference_batch_size": True}, {"batch_wait_ms": -1},
    {"request_timeout_seconds": float("nan")}, {"request_timeout_seconds": 0},
])
def test_invalid_scheduler_limits_fail_before_startup(options):
    with pytest.raises(ValueError):
        SchedulerConfig(**options)
