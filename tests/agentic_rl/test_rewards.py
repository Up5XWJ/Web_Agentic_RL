from copy import deepcopy
import pytest

from open_apps.agentic_rl.rewards import RewardConfig, score_episode
from open_apps.agentic_rl.types import AttemptRecord, Episode, Generation, StepRecord


def episode(success=False, progress=0):
    result = Episode("episode", "group", "task", 0, success=success,
                     metadata={"max_steps": 10})
    result.attempts.append(AttemptRecord(0, 0, [], Generation("{}", [1], [2], [-.5], "v0"), accepted=True))
    result.steps.append(StepRecord(0, {}, "noop()", False, {}, success,
                                   progress, success, False))
    return result


def test_success_always_dominates_auxiliary_rewards():
    success = episode(True, 0)
    success.attempts[0].accepted = False
    success.attempts[0].error_type = "invalid_action"
    success.steps[0].info["action_error"] = "error"
    failure = episode(False, 1)
    assert score_episode(success) > score_episode(failure)
    with pytest.raises(ValueError, match="auxiliary"):
        RewardConfig(success_weight=.1)


@pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
def test_incorrect_finish_has_no_success_reward_and_remains_negative_with_progress(progress):
    result = episode(False, progress)
    result.terminal_reason = "incorrect_finish"
    assert score_episode(result) == pytest.approx(0.15 * progress - 1.0 - 0.03 / 10)
    assert result.reward_components["success"] == 0
    assert result.reward_components["incorrect_finish"] == -1.0
    assert result.trainable


def test_correct_and_incorrect_submissions_have_symmetric_outcome_weights():
    correct = episode(True, 1.0)
    correct.terminal_reason = "success"
    incorrect = episode(False, 1.0)
    incorrect.terminal_reason = "incorrect_finish"
    assert score_episode(correct) - score_episode(incorrect) == pytest.approx(2.0)
    assert correct.reward_components["success"] == 1.0
    assert correct.reward_components["incorrect_finish"] == 0


def test_stronger_failure_only_penalty_preserves_success_dominance_validation():
    config = RewardConfig(success_weight=0.39, incorrect_finish_penalty=2.0)
    success = episode(True, 0.0)
    success.attempts = [AttemptRecord(0, i, [], Generation("bad", [1], [2], [-.5], "v0"),
                                    error_type=kind)
                        for i, kind in enumerate(["format", "invalid_action", "repeat"] * 4)]
    success.steps *= 10
    for step in success.steps:
        step.info["action_error"] = "Execution failed"
    failure = episode(False, 1.0)
    failure.steps[0].truncated = True
    failure.terminal_reason = "max_steps"
    assert score_episode(success, config) > score_episode(failure, config)
    with pytest.raises(ValueError, match="auxiliary"):
        RewardConfig(success_weight=0.37, incorrect_finish_penalty=2.0)


def test_repeating_valid_noops_cannot_accumulate_positive_reward():
    short = episode(False, 0)
    long = deepcopy(short)
    long.attempts *= 10
    long.steps *= 10
    assert score_episode(long) < score_episode(short)
    assert long.reward_components["format"] == short.reward_components["format"]


def test_progress_is_last_retained_state_not_high_water_mark():
    result = episode(False, 1)
    before = score_episode(result)
    result.steps.append(StepRecord(1, {}, "noop()", False, {}, False, 0, False, True))
    assert score_episode(result) < before
    assert result.reward_components["retained_progress"] == 0


def test_infrastructure_errors_are_excluded_not_negative_training_examples():
    result = episode(True, 1)
    result.infrastructure_error = "state endpoint down"
    assert score_episode(result) == 0
    assert result.trainable is False
    assert result.reward_components == {"excluded_infrastructure": 1}


def test_format_parameter_repeat_and_execution_failures_have_penalties():
    for kind in ("format", "invalid_action", "repeat"):
        result = episode()
        result.attempts[0].accepted = False
        result.attempts[0].error_type = kind
        score_episode(result)
        component = {"repeat": "repeat", "format": "format", "invalid_action": "invalid"}[kind]
        assert result.reward_components[component] < 0
    result = episode()
    result.steps[0].info["action_error"] = "Element disappeared"
    score_episode(result)
    assert result.reward_components["execution"] < 0


def test_valid_attempts_do_not_dilute_existing_error_penalties():
    result = episode()
    error = deepcopy(result.attempts[0])
    error.accepted, error.error_type = False, "format"
    result.attempts.insert(0, error)
    score_episode(result)
    penalty = result.reward_components["format"]
    result.attempts.extend(deepcopy(result.attempts[-1]) for _ in range(100))
    score_episode(result)
    assert result.reward_components["format"] == penalty
    assert result.reward_components["invalid"] == 0


def test_scripted_evaluation_gets_score_without_becoming_trainable():
    result = episode(True, 1)
    result.trainable = False
    assert score_episode(result) > 1
    assert not result.trainable
