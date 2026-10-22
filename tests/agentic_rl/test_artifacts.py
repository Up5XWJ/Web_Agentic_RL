import json

import pytest

from open_apps.agentic_rl.artifacts import read_episodes, summarize, write_episodes
from open_apps.agentic_rl.types import AttemptRecord, Episode, Generation, StepRecord


def test_artifacts_preserve_raw_memory_retry_and_logprobs(tmp_path):
    rejected = AttemptRecord(0, 0, [{"role": "user", "content": "exact prompt"}],
                             Generation("bad", [8], [3, 4], [-0.4, -0.2], "v0"), error_type="format")
    accepted = AttemptRecord(0, 1, [{"role": "user", "content": "actual retry prompt"}],
                             Generation("ok", [7, 9], [5], [-0.1], "v0"), accepted=True,
                             raw_memory={"bid": "stale"}, applied_memory={})
    episode = Episode("e", "g", "task", 0, attempts=[rejected, accepted])
    path = tmp_path / "episodes.jsonl"
    write_episodes(path, [episode])
    restored = read_episodes(path)[0]
    assert restored == episode
    with pytest.raises(FileExistsError):
        write_episodes(path, [episode])


def test_metrics_do_not_hide_infrastructure_failures_or_exclude_eval_only():
    solved = Episode("a", "g", "t", 0, success=True, trainable=False, reward=1)
    failed = Episode("b", "g", "t", 0, reward=0)
    broken = Episode("c", "h", "t", 0, trainable=False, infrastructure_error="browser crashed")
    report = summarize([solved, failed, broken])
    assert report["episodes"] == 3
    assert report["infrastructure_failures"] == 1
    assert report["success_rate"] == 0.5
    assert report["zero_variance_group_fraction"] == 0
    json.dumps(report, allow_nan=False)


def test_empty_report_is_serializable_and_not_a_zero_success_claim():
    assert summarize([])["success_rate"] is None
    assert summarize([])["submission_success_rate"] is None


def test_submission_metrics_distinguish_wrong_submissions_from_not_submitting():
    def submitted(identifier, success):
        return Episode(identifier, "g", "t", 0, success=success,
                       terminal_reason="success" if success else "incorrect_finish",
                       steps=[StepRecord(0, {"name": "finish"}, "finish()", False, {},
                                         success, float(success), True, False)])

    episodes = [submitted("a", True), submitted("b", False),
                Episode("c", "g", "t", 0, terminal_reason="max_steps"),
                Episode("d", "h", "t", 0, trainable=False, infrastructure_error="browser crashed")]
    report = summarize(episodes)
    assert report["submission_rate"] == pytest.approx(2 / 3)
    assert report["submission_success_rate"] == 0.5
    assert report["incorrect_finish_rate"] == pytest.approx(1 / 3)
    assert report["infrastructure_failures"] == 1


def test_evaluation_tasks_are_not_misreported_as_a_grpo_comparison_group():
    episodes = [Episode("a", "eval-1", "task-a", 0), Episode("b", "eval-1", "task-b", 0)]
    assert summarize(episodes)["zero_variance_group_fraction"] is None
