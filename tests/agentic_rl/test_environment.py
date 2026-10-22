from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from html.parser import HTMLParser
from unittest.mock import Mock

import pytest

from open_apps.agentic_rl.environment import (BrowserGymEnvironment,
    initial_state_hash, strict_state, validate_browser_action)
from open_apps.agentic_rl.tasks import generate_curriculum
from open_apps.agentic_rl.types import InfrastructureError
from open_apps.agentic_rl.types import TaskSpec


class FindBid(HTMLParser):
    def __init__(self, key, value):
        super().__init__()
        self.key, self.value, self.bid = key, value, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get(self.key) == self.value and attrs.get("bid"):
            self.bid = attrs["bid"]


def bid(obs, key, value):
    parser = FindBid(key, value)
    parser.feed(obs["dom_txt"])
    assert parser.bid is not None, obs["axtree_txt"]
    return parser.bid


def test_state_probe_fails_closed_and_disabled_shop_is_not_queried():
    client = Mock()
    client.get.return_value.json.return_value = []
    assert strict_state("http://localhost:9999", session=client)["online_shop"] == []
    assert client.get.call_count == 5
    client.get.side_effect = RuntimeError("offline")
    with pytest.raises(InfrastructureError, match="todo"):
        strict_state("http://localhost:9999", session=client)


def test_initial_hash_ignores_ports_but_changes_with_business_state():
    root = Path(__file__).resolve().parents[2]
    state = json.loads((root / "tests/states/initial_state.json").read_text())
    a = deepcopy(state)
    b = deepcopy(state)
    a["_url"], b["_url"] = "http://localhost:1234", "http://localhost:5678"
    assert initial_state_hash(a) == initial_state_hash(b)
    b["todo"].append({"title": "new task", "done": False})
    assert initial_state_hash(a) != initial_state_hash(b)


@pytest.mark.parametrize("action", ["goto('http://localhost/todo_all')",
    "page.evaluate('fetch()')", "click('a'); click('b')", "fill('a', __import__('os'))",
    "finish('done')", "finish(success=True)", "finish(**{})"])
def test_environment_rejects_control_plane_and_raw_code_actions(action):
    with pytest.raises(ValueError):
        validate_browser_action(action)


def test_ui_actions_accept_literal_arguments():
    validate_browser_action("fill('17', 'Prepare project')")
    validate_browser_action("scroll(0, 300)")
    assert validate_browser_action("finish()") == "finish"


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1", reason="opt-in real browser")
def test_real_browser_screenshot_prompt_and_retry_artifact():
    from open_apps.agentic_rl.harness import HarnessConfig
    from open_apps.agentic_rl.rollout import run_episode
    from open_apps.agentic_rl.testing import ScriptedTodoPolicy, smoke_task
    from open_apps.agentic_rl.types import Episode
    from open_apps.agentic_rl.vision import _processor_messages

    task = smoke_task()
    result = run_episode(task, ScriptedTodoPolicy(task.task_config["todo_name"], inject_invalid=True),
                         BrowserGymEnvironment(), harness_config=HarnessConfig(use_screenshot=True))
    assert result.success and not result.infrastructure_error
    assert not result.trainable  # Scripted success must not be called a model evaluation.
    restored = Episode.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.attempts[0].messages[1] == restored.attempts[1].messages[1]
    for attempt in restored.attempts:
        _, images = _processor_messages(attempt.messages)
        assert len(images) == 1 and min(images[0].size) > 0


def test_server_reserves_distinct_sockets_until_startup():
    import socket
    from open_apps.mcp.appserver import _reserve_socket

    sockets = [_reserve_socket("127.0.0.1", None) for _ in range(8)]
    try:
        ports = [sock.getsockname()[1] for sock in sockets]
        assert len(set(ports)) == 8
        with socket.socket() as competitor:
            with pytest.raises(OSError):
                competitor.bind(("127.0.0.1", ports[0]))
    finally:
        for sock in sockets:
            sock.close()


def test_rl_storage_cannot_be_redirected_to_another_rollout(tmp_path):
    from open_apps.mcp.appserver import _load_hydra_config

    own = tmp_path / "own"
    own.mkdir()
    cfg, path = _load_hydra_config(["logs_dir=/tmp/shared", "databases_dir=/tmp/shared/databases"],
                                  storage_directory=str(own))
    assert path == str(own)
    assert Path(cfg.logs_dir) == own
    assert Path(cfg.databases_dir) == own / "databases"
    assert all(Path(app.database_path).is_relative_to(own) for app in cfg.apps.values() if hasattr(app, "database_path"))
    with pytest.raises(ValueError, match="escapes"):
        _load_hydra_config([f"apps.todo.database_path={tmp_path}/another/todo.db"], storage_directory=str(own))
    assert not (tmp_path / "another").exists()


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to launch two concurrent browser/backend processes")
def test_concurrent_browsers_isolate_business_state_verification_and_cleanup():
    title = "Concurrent isolated todo 6291"
    goal = f"Add '{title}' to my todo list."
    task = TaskSpec("concurrent_add", goal, {"_target_": "open_apps.tasks.tasks.AddToDoTask",
                    "goal": goal, "todo_name": title, "is_done": False}, max_steps=8)
    with BrowserGymEnvironment(timeout_seconds=120) as first_env, BrowserGymEnvironment(timeout_seconds=120) as second_env:
        with ThreadPoolExecutor(2) as executor:
            first_call = executor.submit(first_env.reset, task)
            second_call = executor.submit(second_env.reset, task)
            first, second = first_call.result(), second_call.result()
        assert first.initial_state_hash == second.initial_state_hash
        assert first.info["worker_pid"] != second.info["worker_pid"]
        assert first.info["base_url"] != second.info["base_url"]
        directories = [Path(item.info["storage_directory"]) for item in (first, second)]
        assert directories[0] != directories[1] and all(path.exists() for path in directories)
        assert set(first.info["database_paths"].values()).isdisjoint(second.info["database_paths"].values())
        for item, directory in zip((first, second), directories):
            assert all(Path(path).is_relative_to(directory) for path in item.info["database_paths"].values())

        before_second = strict_state(second.info["base_url"])
        todo = first_env.step(f"click({bid(first.observation, 'href', '/todo')!r})")
        filled = first_env.step(f"fill({bid(todo.observation, 'placeholder', 'New Todo')!r}, {title!r})")
        saved = first_env.step(f"click({bid(filled.observation, 'id', 'submit-button')!r})")
        assert saved.progress == 1
        assert strict_state(second.info["base_url"]) == before_second
        assert first_env.step("finish()").success
        first_env.close()
        assert not directories[0].exists() and directories[1].exists()
        assert second_env.step("noop(wait_ms=0)").progress == 0
        submitted = second_env.step("finish()")
        assert submitted.terminated and not submitted.success
    assert not any(path.exists() for path in directories)


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to kill an owned test environment worker")
def test_worker_crash_cleans_parent_owned_storage():
    task = generate_curriculum(count=1, max_steps=2)[0]
    with BrowserGymEnvironment(timeout_seconds=120) as env:
        initial = env.reset(task)
        directory = Path(initial.info["storage_directory"])
        assert directory.exists()
        env._process.kill()
        env._process.join(timeout=5)
        with pytest.raises(InfrastructureError, match="unavailable|unexpectedly"):
            env.step("noop(wait_ms=0)")
        assert not directory.exists()
        assert env._process is None


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to execute the concurrent Harness with real browsers")
def test_parallel_scheduler_drives_complete_real_browser_group():
    from open_apps.agentic_rl.rollout import sample_batch
    from open_apps.agentic_rl.scheduler import SchedulerConfig
    from open_apps.agentic_rl.testing import ScriptedTodoPolicy, smoke_task

    task = smoke_task()

    class ScriptedBatchPolicy:
        policy_version = "scripted-smoke-v1"

        def __init__(self):
            self.policies = {}

        def generate_batch(self, requests):
            results = []
            for request in requests:
                episode_id = request.request_id.rsplit("/call-", 1)[0]
                if episode_id not in self.policies:
                    self.policies[episode_id] = ScriptedTodoPolicy(task.task_config["todo_name"], inject_invalid=True)
                generated = self.policies[episode_id].generate(request.messages)
                generated.metadata.update(request_id=request.request_id, sample_seed=request.sample_seed)
                results.append(generated)
            return results

    batch = sample_batch([("parallel-smoke", task)], ScriptedBatchPolicy(), BrowserGymEnvironment,
                         group_size=2, scheduler_config=SchedulerConfig(2, 2, 30))
    assert len(batch.episodes) == 2 and all(episode.success for episode in batch.episodes)
    assert all(episode.submitted and not episode.trainable for episode in batch.episodes)
    assert all(episode.attempts[0].error_type == "format" for episode in batch.episodes)
    assert len({ep.initial_state_hash for ep in batch.episodes}) == 1
    assert len({ep.metadata["reset_info"]["base_url"] for ep in batch.episodes}) == 2
    assert all(not Path(ep.metadata["reset_info"]["storage_directory"]).exists() for ep in batch.episodes)
    assert batch.metrics["rollout/peak_concurrent_episodes"] == 2


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to launch local BrowserGym/app subprocesses")
def test_real_browser_reset_isolates_state_and_enforces_horizon():
    task = generate_curriculum(count=1, max_steps=1)[0]
    with BrowserGymEnvironment(timeout_seconds=120) as env:
        first = env.reset(task)
        assert first.observation["axtree_txt"]
        result = env.step("noop(wait_ms=0)")
        assert result.truncated and not result.success
        second = env.reset(task)
        assert second.initial_state_hash == first.initial_state_hash


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to launch local BrowserGym/app subprocesses")
@pytest.mark.parametrize("last_action,success,terminated,truncated", [
    ("finish()", True, True, False),
    ("noop(wait_ms=0)", False, False, True),
    ("report_infeasible('Unable to verify completion')", False, True, False),
])
def test_real_browser_completed_state_requires_submission_and_reset_discards_mutation(
    last_action, success, terminated, truncated,
):
    title = "RL smoke task unique 937"
    goal = f"Add '{title}' to my todo list."
    task = TaskSpec("smoke_add", goal, {"_target_": "open_apps.tasks.tasks.AddToDoTask",
                    "goal": goal, "todo_name": title, "is_done": False}, max_steps=4)

    with BrowserGymEnvironment(timeout_seconds=120) as env:
        first = env.reset(task)
        todo = env.step(f"click({bid(first.observation, 'href', '/todo')!r})")
        field = bid(todo.observation, "placeholder", "New Todo")
        filled = env.step(f"fill({field!r}, {title!r})")
        submit = bid(filled.observation, "id", "submit-button")
        result = env.step(f"click({submit!r})")
        assert not result.success and not result.terminated and not result.truncated
        assert result.progress == 1
        assert not result.info["action_error"]
        assert result.info["http_errors"] == []
        assert title in result.observation["axtree_txt"]
        ended = env.step(last_action)
        assert (ended.success, ended.terminated, ended.truncated) == (success, terminated, truncated)
        assert ended.progress == 1
        assert ended.info["submitted"] == success
        assert ended.info["submission_correct"] is (True if success else None)
        assert ended.info["step_count"] == task.max_steps
        second = env.reset(task)
        assert second.initial_state_hash == first.initial_state_hash
        premature = env.step("finish()")
        assert premature.terminated and not premature.success and not premature.truncated
        assert premature.progress == 0 and not premature.info["action_error"]
        assert premature.info["submitted"] and premature.info["submission_correct"] is False


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to launch local BrowserGym/app subprocesses")
def test_real_browser_finish_rechecks_full_current_state_instead_of_cached_progress():
    title = "Verify fresh submission 419"
    goal = f"Add '{title}' to my todo list."
    task = TaskSpec("fresh_submission", goal, {"_target_": "open_apps.tasks.tasks.AddToDoTask",
                    "goal": goal, "todo_name": title, "is_done": False}, max_steps=8)
    with BrowserGymEnvironment(timeout_seconds=120) as env:
        first = env.reset(task)
        todo = env.step(f"click({bid(first.observation, 'href', '/todo')!r})")
        filled = env.step(f"fill({bid(todo.observation, 'placeholder', 'New Todo')!r}, {title!r})")
        complete = env.step(f"click({bid(filled.observation, 'id', 'submit-button')!r})")
        assert complete.progress == 1 and not complete.terminated
        # An extra, unrequested mutation preserves the scoped progress signal
        # but violates the task's full-state success predicate.
        filled_extra = env.step(f"fill({bid(complete.observation, 'placeholder', 'New Todo')!r}, 'Unrequested extra todo')")
        extra = env.step(f"click({bid(filled_extra.observation, 'id', 'submit-button')!r})")
        assert extra.progress == 1 and not extra.terminated
        submitted = env.step("finish()")
        assert submitted.terminated and not submitted.success and not submitted.truncated
        assert submitted.progress == 1 and not submitted.info["action_error"]
        assert submitted.info["submission_correct"] is False


@pytest.mark.skipif(os.environ.get("OPENAPPS_RL_BROWSER_SMOKE") != "1",
                    reason="opt in to launch local BrowserGym/app subprocesses")
def test_real_browser_cross_app_partial_reward_and_final_success():
    title = "Cross app RL task 418"
    message = "Cross app task 418 is ready."
    todo = {"_target_": "open_apps.tasks.tasks.AddToDoTask",
            "goal": f"Add '{title}' to my todo list.", "todo_name": title, "is_done": False}
    messenger = {"_target_": "open_apps.tasks.tasks.SendMessageTask",
                 "goal": f"Send Alice the message '{message}'", "to": "Alice", "message": message}
    goal = todo["goal"] + " " + messenger["goal"]
    task = TaskSpec("smoke_cross_app", goal,
                    {"_target_": "open_apps.tasks.tasks.CompositeTask", "_convert_": "all",
                     "goal": goal, "subtasks": [todo, messenger]}, max_steps=30)

    with BrowserGymEnvironment(timeout_seconds=120) as env:
        first = env.reset(task)
        observation = first.observation

        def act(action):
            nonlocal observation
            result = env.step(action)
            assert not result.info["action_error"], result.info
            assert not result.info["http_errors"]
            observation = result.observation
            return result

        def visible_bid(key, value):
            found = bid(observation, key, value)
            for _ in range(6):
                properties = observation["extra_element_properties"].get(found, {})
                if properties.get("visibility", 0) >= .5:
                    return found
                act("scroll(0, 600)")
                found = bid(observation, key, value)
            pytest.fail(f"Could not scroll target into view: {key}={value}")

        act(f"click({visible_bid('href', '/todo')!r})")
        act(f"fill({visible_bid('placeholder', 'New Todo')!r}, {title!r})")
        partial = act(f"click({visible_bid('id', 'submit-button')!r})")
        assert partial.progress == .5
        assert not partial.success and not partial.terminated
        act(f"click({visible_bid('href', '/')!r})")
        act(f"click({visible_bid('href', '/messages')!r})")
        act(f"click({visible_bid('href', '/messages/Alice')!r})")
        act(f"fill({visible_bid('placeholder', 'Type a message')!r}, {message!r})")
        final = act(f"click({visible_bid('type', 'submit')!r})")
        assert not final.success and not final.terminated and final.progress == 1
        assert message in final.observation["axtree_txt"]
        submitted = act("finish()")
        assert submitted.success and submitted.terminated and submitted.progress == 1
        second = env.reset(task)
        assert second.initial_state_hash == first.initial_state_hash
