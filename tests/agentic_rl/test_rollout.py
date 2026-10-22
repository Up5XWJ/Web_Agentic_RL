from __future__ import annotations

from copy import deepcopy
import json

import pytest

from open_apps.agentic_rl.harness import ContextLimitError, HarnessConfig
from open_apps.agentic_rl.rollout import run_episode, sample_group
from open_apps.agentic_rl.types import Generation, ResetResult, StepResult, TaskSpec


def obs(text="initial"):
    return {"url": "http://localhost/todo", "pruned_html": '<button bid="1">Save</button>' + text,
            "axtree_txt": "[1] button Save", "last_action_error": "", "focused_element_bid": "",
            "extra_element_properties": {"1": {"visibility": 1.0}}}


def output(status="not_applicable", changed=False, action=None):
    return json.dumps({
        "memory_next": {"plan": [{"id": "s", "description": "Save", "status": "in_progress"}], "current_subgoal_id": "s"},
        "previous_action_assessment": {"status": status, "page_changed": changed, "evidence": "feedback"},
        "decision": {"observation": "Save", "current_subgoal": "Save", "reason": "Save task", "expected_result": "Saved"},
        "action": action or {"name": "finish"},
    })


class Policy:
    def __init__(self, outputs=None, versions=None, trainable=True):
        self.outputs = list(outputs or [])
        self.versions = list(versions or [])
        self.trainable = trainable

    def generate(self, messages):
        text = self.outputs.pop(0) if self.outputs else output()
        if isinstance(text, Exception):
            raise text
        return Generation(text, [1, 2], [3, 4], [-1.0, -1.0], self.versions.pop(0) if self.versions else "v1", metadata={"trainable": self.trainable})


class Environment:
    def __init__(self, *, state_hash="state", reset_error=None, step_error=None, close_error=None, steps=None):
        self.state_hash = state_hash
        self.reset_error = reset_error
        self.step_error = step_error
        self.close_error = close_error
        self.steps = list(steps or [StepResult(obs("saved"), success=True, terminated=True, progress=1.0)])
        self.actions = []
        self.closed = False
        self.task = None

    def reset(self, task):
        self.task = deepcopy(task)
        if self.reset_error:
            raise self.reset_error
        return ResetResult(obs(), self.state_hash)

    def step(self, action):
        self.actions.append(action)
        if self.step_error:
            raise self.step_error
        return self.steps.pop(0)

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def reward(episode):
    episode.reward_components = {"success": float(episode.success)}
    return float(episode.success)


def task(max_steps=2):
    return TaskSpec("t", "Save", {"hidden_target": "not_for_prompt"}, seed=7, max_steps=max_steps)


def test_success_episode_assesses_last_action_and_closes_environment():
    environment = Environment()
    episode = run_episode(task(), Policy(), environment, reward_fn=reward)
    assert environment.closed
    assert episode.success and episode.trainable
    assert episode.reward == 1.0
    assert episode.steps[-1].assessment["status"] == "success"
    assert episode.steps[-1].terminated
    assert episode.submitted and environment.actions == ["finish()"]
    assert len(episode.attempts) == len(episode.steps) == 1
    assert episode.policy_version == "v1" and episode.initial_state_hash == "state"
    assert "not_for_prompt" not in json.dumps(episode.attempts[0].messages)
    assert episode.metadata["harness_config"]["max_attempts"] == 3


def test_horizon_and_uncertain_feedback_do_not_fabricate_negative_business_reward():
    environment = Environment(steps=[StepResult(obs(), progress=0.0)])
    episode = run_episode(task(1), Policy([output(action={"name": "click", "bid": "1"})]),
                          environment, reward_fn=reward)
    assert episode.terminal_reason == "max_steps"
    assert episode.steps[-1].truncated
    assert episode.steps[-1].assessment["status"] == "uncertain"
    assert episode.reward == 0.0
    assert episode.trainable


def test_completed_state_is_observed_before_model_submits_on_last_step():
    environment = Environment(steps=[
        StepResult(obs("saved"), progress=1.0),
        StepResult(obs("saved"), success=True, terminated=True, progress=1.0),
    ])
    policy = Policy([
        output(action={"name": "click", "bid": "1"}),
        output(status="success", changed=True),
    ])
    episode = run_episode(task(2), policy, environment, reward_fn=reward)
    assert episode.success and episode.trainable and episode.submitted
    assert environment.actions == ["click('1', button='left')", "finish()"]
    assert len(episode.steps) == len(episode.attempts) == 2
    assert not episode.steps[0].success and not episode.steps[0].terminated
    assert episode.steps[-1].terminated and not episode.steps[-1].truncated
    assert "saved" in episode.attempts[-1].messages[1]["content"]
    assert "1 action step(s) remain, including finish" in episode.attempts[-1].messages[1]["content"]


def test_incorrect_finish_is_a_trainable_terminal_failure_without_retry():
    from open_apps.agentic_rl.rewards import score_episode

    environment = Environment(steps=[StepResult(obs(), terminated=True)])
    episode = run_episode(task(5), Policy(), environment, reward_fn=score_episode)
    assert episode.trainable and episode.submitted and not episode.success
    assert not episode.infrastructure_error
    assert episode.terminal_reason == "incorrect_finish"
    assert len(episode.attempts) == len(episode.steps) == 1
    assert episode.steps[-1].assessment["status"] == "failed"
    assert episode.reward_components["success"] == 0
    assert episode.reward_components["incorrect_finish"] == -1.0


def test_complete_progress_without_submission_does_not_earn_success_reward():
    from open_apps.agentic_rl.rewards import score_episode

    environment = Environment(steps=[StepResult(obs("saved"), progress=1.0)])
    episode = run_episode(task(1), Policy([output(action={"name": "click", "bid": "1"})]),
                          environment, reward_fn=score_episode)
    assert episode.trainable and not episode.success and not episode.submitted
    assert episode.terminal_reason == "max_steps" and episode.steps[-1].truncated
    assert episode.reward_components["success"] == 0
    assert episode.reward_components["retained_progress"] > 0


@pytest.mark.parametrize("action,result", [
    ({"name": "click", "bid": "1"}, StepResult(obs(), success=True, terminated=True)),
    ({"name": "finish"}, StepResult(obs())),
    ({"name": "finish"}, StepResult(obs(), truncated=True)),
])
def test_inconsistent_environment_submission_protocol_is_not_trained(action, result):
    episode = run_episode(task(), Policy([output(action=action)]), Environment(steps=[result]),
                          reward_fn=reward)
    assert not episode.trainable and episode.infrastructure_error


def test_successful_and_incorrect_submissions_both_enter_grpo_with_original_tokens():
    from open_apps.agentic_rl.rewards import score_episode
    from open_apps.agentic_rl.training_data import prepare_training_batch

    environments = [Environment(), Environment(steps=[StepResult(obs(), terminated=True)])]
    episodes = sample_group(task(), Policy(), lambda: environments.pop(0),
                            group_size=2, reward_fn=score_episode)
    assert all(episode.trainable and episode.metadata["group_eligible"] for episode in episodes)
    batch = prepare_training_batch(episodes, group_size=2, policy_version="v1")
    assert len(batch.rows) == 2
    assert batch.rows[0].advantage > 0 and batch.rows[1].advantage < 0
    for row, episode in zip(batch.rows, episodes):
        generated = episode.attempts[0].generation
        assert row.response_token_ids == generated.response_token_ids
        assert row.old_log_probs == generated.old_log_probs
        assert episode.attempts[0].action["name"] == "finish"
    assert batch.metrics["train/submission_rate"] == 1.0
    assert batch.metrics["train/incorrect_finish_rate"] == 0.5


def test_fallback_executes_but_all_rejected_generations_remain_trainable():
    environment = Environment(steps=[StepResult(obs())])
    episode = run_episode(task(1), Policy(outputs=["bad", "bad again"]), environment,
                          harness_config=HarnessConfig(max_attempts=2), reward_fn=reward)
    assert episode.trainable and len(episode.attempts) == 2
    assert episode.steps[0].fallback
    assert environment.actions == ["noop(wait_ms=1000)"]
    assert all(not attempt.accepted for attempt in episode.attempts)


@pytest.mark.parametrize("phase", ["reset", "step", "close", "policy"])
def test_infrastructure_failures_preserve_provenance_and_close(phase):
    environment = Environment(**{f"{phase}_error": RuntimeError("broken")} if phase != "policy" else {})
    policy = Policy(outputs=[RuntimeError("network")]) if phase == "policy" else Policy()
    episode = run_episode(task(), policy, environment, reward_fn=reward)
    assert environment.closed
    assert not episode.trainable
    assert episode.infrastructure_error
    if phase == "policy":
        assert not episode.attempts
        assert episode.metadata["model_errors"][0]["error"] == "network"
    if phase == "step":
        assert episode.attempts[0].accepted
        assert not episode.steps


def test_context_limit_is_not_a_model_failure_reward():
    environment = Environment()
    episode = run_episode(task(), Policy(outputs=[ContextLimitError("budget")]), environment, reward_fn=reward)
    assert episode.terminal_reason == "context_limit"
    assert not episode.trainable and not episode.attempts


def test_groups_use_separate_reset_environments_and_keep_all_members():
    environments = []

    def factory():
        environment = Environment()
        environments.append(environment)
        return environment

    episodes = sample_group(task(), Policy(), factory, group_size=3, group_id="g", reward_fn=reward)
    assert len(episodes) == 3 and len(environments) == 3
    assert all(environment.closed and environment.task.seed == 7 for environment in environments)
    assert {episode.group_id for episode in episodes} == {"g"}
    assert all(episode.trainable and episode.metadata["group_eligible"] for episode in episodes)


@pytest.mark.parametrize("mismatch", ["state", "version", "infrastructure", "scripted"])
def test_one_bad_group_member_disqualifies_whole_group_without_dropping_it(mismatch):
    environments = [Environment(), Environment(state_hash="other" if mismatch == "state" else "state",
                    step_error=RuntimeError("broken") if mismatch == "infrastructure" else None)]
    queue = list(environments)
    policy = Policy(versions=["v1", "v2"] if mismatch == "version" else None, trainable=mismatch != "scripted")
    episodes = sample_group(task(), policy, lambda: queue.pop(0), group_size=2, reward_fn=reward)
    assert len(episodes) == 2
    assert all(environment.closed for environment in environments)
    assert all(not episode.trainable and not episode.metadata["group_eligible"] for episode in episodes)
    expected = {"state": "initial_state_mismatch", "version": "policy_version_mismatch",
                "infrastructure": "infrastructure_error", "scripted": "untrainable_member"}[mismatch]
    assert all(expected in episode.metadata["group_skip_reasons"] for episode in episodes)


def test_environment_factory_error_still_produces_group_member():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("startup failed")
        return Environment()

    episodes = sample_group(task(), Policy(), factory, group_size=2, reward_fn=reward)
    assert len(episodes) == 2
    assert not any(episode.trainable for episode in episodes)
    assert "environment_creation" in episodes[0].infrastructure_error


def test_episode_roundtrip_retains_raw_and_applied_memory():
    from open_apps.agentic_rl.types import Episode

    episode = run_episode(task(), Policy(), Environment(), reward_fn=reward)
    restored = Episode.from_dict(json.loads(json.dumps(episode.to_dict())))
    assert restored.to_dict() == episode.to_dict()
    assert restored.submitted
