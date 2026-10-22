from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from open_apps.agentic_rl.harness import ContextLimitError, HarnessConfig, MemoryHarness, validate_generation
from open_apps.agentic_rl.types import Episode, Generation, InfrastructureError, TaskSpec


def observation(text="A page", *, changed=False):
    return {
        "url": "http://localhost/todo", "pruned_html": '<button bid="12">Save</button>' + text,
        "axtree_txt": "[12] button 'Save'", "focused_element_bid": "",
        "last_action_error": "", "extra_element_properties": {"12": {"visibility": 1.0}},
        "screenshot": np.zeros((640, 1024, 3), dtype=np.uint8),
    }


def response(*, action=None, status="not_applicable", changed=False, pending=None):
    return json.dumps({
        "memory_next": {
            "plan": [{"id": "task", "description": "Save the task", "status": "in_progress"}],
            "current_subgoal_id": "task", "verified_facts": [], "pending": pending or [],
            "failures": [], "uncertainties": [],
            "transient_page_state": {"url": "model-url", "focused_bid": "wrong", "bids": {"save": "12", "stale": "999"}},
        },
        "previous_action_assessment": {"status": status, "page_changed": changed, "evidence": "from observation"},
        "decision": {"observation": "Save button", "current_subgoal": "Save", "reason": "Requested change", "expected_result": "Saved"},
        "action": action or {"name": "click", "bid": "12"},
    })


def generation(text, *, version="v1", finish_reason="stop", trainable=True):
    return Generation(text, [10, 11], [12, 13, 14], [-1.0, -2.0, -3.0], version, finish_reason, {"trainable": trainable})


class QueuePolicy:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def generate(self, messages):
        self.messages.append(deepcopy(messages))
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def harness_and_episode(config=None):
    harness = MemoryHarness(config)
    harness.reset(TaskSpec("t", "Save the task", {}), observation())
    return harness, Episode("e", "g", "t", 0)


def test_retry_records_original_generations_and_raw_applied_memory_separately():
    harness, episode = harness_and_episode()
    invalid = response(action={"name": "click", "bid": "999"})
    valid = response(pending=["next", "next"])
    policy = QueuePolicy([generation(invalid), generation(valid)])
    selected = harness.select_action(policy, observation(), episode, 0)

    assert not selected.fallback
    assert [record.error_type for record in episode.attempts] == ["invalid_action", None]
    assert [record.accepted for record in episode.attempts] == [False, True]
    assert episode.attempts[0].generation.text == invalid
    assert episode.attempts[1].generation.text == valid
    assert episode.attempts[1].messages == policy.messages[1]
    assert episode.attempts[1].messages[-2] == {"role": "assistant", "content": invalid}
    accepted = episode.attempts[1]
    assert accepted.raw_memory["pending"] == ["next", "next"]
    assert accepted.applied_memory["pending"] == ["next"]
    assert accepted.raw_memory["transient_page_state"]["url"] == "model-url"
    assert accepted.applied_memory["transient_page_state"] == {
        "url": observation()["url"], "focused_bid": "", "bids": {"save": "12"},
    }
    assert episode.attempts[0].applied_memory is None


def test_rejected_outputs_are_all_recorded_but_fallback_has_no_generation():
    harness, episode = harness_and_episode(HarnessConfig(max_attempts=2))
    outputs = [generation("bad json"), generation(response(action={"name": "click", "bid": "missing"}))]
    selected = harness.select_action(QueuePolicy(outputs), observation(), episode, 0)
    assert selected.fallback and selected.action.name == "noop"
    assert len(episode.attempts) == 2
    assert [attempt.error_type for attempt in episode.attempts] == ["format", "invalid_action"]
    assert all(not attempt.accepted for attempt in episode.attempts)
    assert harness.memory.plan == []


def test_next_step_rebuilds_context_with_applied_memory_and_bounded_replay():
    harness, episode = harness_and_episode(HarnessConfig(recent_turn_replay_size=1))
    policy = QueuePolicy([generation("failed-old-output"), generation(response()), generation(response(status="success", changed=True))])
    selected = harness.select_action(policy, observation(), episode, 0)
    after = observation("Saved")
    harness.observe_transition(observation(), after, selected, 0)
    harness.select_action(policy, after, episode, 1)
    next_messages = policy.messages[-1]
    assert len(next_messages) == 2
    assert all(message["role"] != "assistant" for message in next_messages)
    assert "failed-old-output" not in json.dumps(next_messages)
    assert '"url": "http://localhost/todo"' in next_messages[1]["content"]
    assert "retained=1" in next_messages[1]["content"]


def test_text_mode_does_not_claim_image_input_and_rejects_coordinate_clicks():
    harness, episode = harness_and_episode(HarnessConfig(max_attempts=1))
    messages = harness.build_messages(observation())
    text = "\n".join(message["content"] for message in messages)
    assert "No screenshot or image is supplied" in text
    assert "all three" not in text
    assert "Inspect this image" not in text
    assert all(isinstance(message["content"], str) for message in messages)
    result = harness.select_action(QueuePolicy([generation(response(action={"name": "mouse_click", "coordinate": [10, 10]}))]), observation(), episode, 0)
    assert result.fallback and episode.attempts[0].error_type == "invalid_action"
    with pytest.raises(ValueError, match="boolean"):
        HarnessConfig(use_screenshot="false")


def test_screenshot_mode_records_one_current_png_and_reuses_it_on_retry():
    harness, episode = harness_and_episode(HarnessConfig(use_screenshot=True))
    policy = QueuePolicy([generation("bad json"), generation(response(action={"name": "finish"}))])
    decision = harness.select_action(policy, observation(), episode, 0)
    assert decision.action.name == "finish"
    first, retry = policy.messages
    assert first[1] == retry[1]
    assert first[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "including finish" in first[1]["content"][0]["text"]
    assert "three EQUAL" in first[0]["content"]
    assert "No screenshot" not in first[0]["content"]
    assert "independently verifies" in first[0]["content"]
    assert "- mouse_click:" in first[0]["content"]
    assert len(first[1]["content"]) == 2
    assert Episode.from_dict(json.loads(json.dumps(episode.to_dict()))).attempts[1].messages == retry
    after = observation()
    after["screenshot"][:] = 255
    next_messages = harness.build_messages(after)
    assert len(next_messages) == 2
    assert next_messages[1]["content"][1] != first[1]["content"][1]


def test_screenshot_mode_missing_image_does_not_fall_back_to_text():
    harness, _ = harness_and_episode(HarnessConfig(use_screenshot=True))
    obs = observation()
    del obs["screenshot"]
    with pytest.raises(InfrastructureError, match="requires a valid"):
        harness.build_messages(obs)


def test_screenshot_mode_enables_coordinate_actions():
    harness, episode = harness_and_episode(HarnessConfig(use_screenshot=True, max_attempts=1))
    policy = QueuePolicy([generation(response(action={"name": "mouse_click", "coordinate": [100, 200]}))])
    selected = harness.select_action(policy, observation(), episode, 0)
    assert not selected.fallback and selected.action.name == "mouse_click"


@pytest.mark.parametrize("extra", [{"bid": "12"}, {"reason": "done"}, {"success": True}])
def test_finish_requires_no_arguments_and_keeps_rejected_attempts(extra):
    harness, episode = harness_and_episode()
    invalid = response(action={"name": "finish", **extra})
    valid = response(action={"name": "finish"})
    policy = QueuePolicy([generation(invalid), generation(valid)])
    decision = harness.select_action(policy, observation(), episode, 0)
    assert decision.action.name == "finish" and decision.browser_action == "finish()"
    assert not decision.fallback
    assert [a.accepted for a in episode.attempts] == [False, True]
    assert episode.attempts[0].error_type == "invalid_action"
    assert episode.attempts[0].generation.text == invalid
    assert episode.attempts[1].generation.text == valid
    # Declaring completion is allowed even without a self-marked verified plan;
    # only the independent environment can decide whether the claim is true.
    assert harness.memory.plan[0].status == "in_progress"
    prompt = "\n".join(m["content"] for m in policy.messages[0])
    assert '- finish:' in prompt and '|noop|finish|report_infeasible' in prompt
    assert 'success after every action' not in prompt
    assert 'a valid finish ends the episode' in prompt.lower()


def test_repeat_guard_is_separate_from_parameter_errors_and_final_assessment_exists():
    harness, episode = harness_and_episode(HarnessConfig(max_attempts=1))
    policy = QueuePolicy([generation(response()), generation(response(status="uncertain")), generation(response(status="uncertain"))])
    obs = observation()
    for index in range(2):
        selected = harness.select_action(policy, obs, episode, index)
        assessment = harness.observe_transition(obs, obs, selected, index)
        assert assessment.status == "uncertain"
    third = harness.select_action(policy, obs, episode, 2)
    assert third.fallback
    assert episode.attempts[-1].error_type == "repeat"


def test_truncated_output_is_rejected_even_if_json_happens_to_parse():
    harness, episode = harness_and_episode(HarnessConfig(max_attempts=1))
    result = harness.select_action(QueuePolicy([generation(response(), finish_reason="length")]), observation(), episode, 0)
    assert result.fallback
    assert episode.attempts[0].generation.response_token_ids == [12, 13, 14]
    assert episode.attempts[0].error_type == "format"
    assert episode.attempts[0].generation.metadata["rejection_reason"] == "output_truncated"


@pytest.mark.parametrize("field,value", [
    ("old_log_probs", [-1.0]), ("old_log_probs", [-1.0, math.nan, -2.0]),
    ("old_log_probs", [0.2, -1.0, -2.0]), ("response_token_ids", []),
    ("prompt_token_ids", [True]), ("policy_version", ""),
])
def test_invalid_generation_metadata_fails_closed(field, value):
    item = generation(response())
    setattr(item, field, value)
    with pytest.raises(InfrastructureError):
        validate_generation(item)


def test_policy_version_changes_are_not_silently_mixed():
    harness, episode = harness_and_episode()
    policy = QueuePolicy([generation("bad", version="v1"), generation(response(), version="v2")])
    with pytest.raises(InfrastructureError, match="version changed"):
        harness.select_action(policy, observation(), episode, 0)
    assert len(episode.attempts) == 2
    assert episode.attempts[-1].error_type == "infrastructure"


def test_network_failure_has_no_synthetic_generation_and_context_limit_is_explicit():
    harness, episode = harness_and_episode()
    with pytest.raises(ContextLimitError):
        harness.select_action(QueuePolicy([ContextLimitError("too long")]), observation(), episode, 0)
    assert not episode.attempts
    assert episode.metadata["model_errors"][0]["messages"]


def test_scripted_policy_trainable_false_is_respected():
    harness, episode = harness_and_episode()
    harness.select_action(QueuePolicy([generation(response(), trainable=False)]), observation(), episode, 0)
    assert not episode.trainable
