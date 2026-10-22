from copy import deepcopy
import json
from pathlib import Path

import pytest
from hydra.utils import instantiate

from open_apps.agentic_rl.tasks import (assign_grouped_splits, derive_predicates,
    export_hydra_tasks, family_id, generate_curriculum, load_manifest,
    progress_fraction, save_manifest)


ROOT = Path(__file__).resolve().parents[2]


def state():
    return json.loads((ROOT / "tests/states/initial_state.json").read_text())


def test_paraphrases_nested_goals_stay_in_same_split(tmp_path):
    original = generate_curriculum(count=1)[0]
    paraphrase = deepcopy(original)
    paraphrase.task_id += "_paraphrase"
    paraphrase.goal = "Please " + original.goal
    paraphrase.task_config["goal"] = paraphrase.goal
    splits = assign_grouped_splits([original, paraphrase])
    assert sorted(map(len, splits.values())) == [0, 0, 2]
    save_manifest([original, paraphrase], tmp_path / "tasks.jsonl")
    assert load_manifest(tmp_path / "tasks.jsonl") == [original, paraphrase]
    nested = {"subtasks": [original.task_config]}
    nested2 = {"subtasks": [paraphrase.task_config]}
    assert family_id(nested) == family_id(nested2)


def test_manifest_rejects_spoofed_family_and_goal(tmp_path):
    tasks = generate_curriculum(count=1)
    tasks[0].family_id = "fake-split-family"
    with pytest.raises(ValueError, match="family"):
        save_manifest(tasks, tmp_path / "bad.jsonl")
    tasks[0].family_id = ""
    tasks[0].goal = "Different task"
    with pytest.raises(ValueError, match="mismatch"):
        save_manifest(tasks, tmp_path / "bad.jsonl")


def test_curriculum_is_reproducible_and_targets_are_satisfiable():
    specs = generate_curriculum(count=30, max_subgoals=3)
    assert specs == generate_curriculum(count=30, max_subgoals=3)
    assert len({s.family_id for s in specs}) == 30
    initial = state()
    for spec in specs:
        task = instantiate(spec.task_config)
        target = task.get_target_state(initial)
        assert task.check_if_task_is_complete(initial, target)
        assert not task.check_if_task_is_complete(initial, initial)
        assert progress_fraction(spec.progress_predicates, initial, target) == 1
        assert progress_fraction(spec.progress_predicates, initial, initial) == 0


def test_retained_progress_handles_sibling_changes_undo_and_duplicate_requests():
    spec = next(s for s in generate_curriculum(count=3, max_subgoals=3)
                if len(s.progress_predicates) == 3)
    task = instantiate(spec.task_config)
    initial = state()
    current = task.subtasks[0].get_target_state(initial)
    assert progress_fraction(spec.progress_predicates, initial, current) == pytest.approx(1/3)
    current = task.get_target_state(initial)
    assert progress_fraction(spec.progress_predicates, initial, current) == 1
    assert progress_fraction(spec.progress_predicates, initial, initial) == 0
    add = {"_target_": "open_apps.tasks.tasks.AddToDoTask", "todo_name": "unique", "is_done": False}
    predicates = derive_predicates(add) * 2
    current = deepcopy(initial)
    current["todo"].append({"title": "unique", "done": False})
    assert progress_fraction(predicates, initial, current) == .5


def test_export_loads_default_composition_and_long_horizon_separately():
    base = export_hydra_tasks(ROOT / "config/tasks/all_tasks.yaml")
    long = export_hydra_tasks(ROOT / "config/tasks/longer_horizon.yaml")
    assert len(base) == 274
    assert len(long) == 100
    assert all(2 <= len(t.progress_predicates) <= 4 for t in long)


def test_live_message_sender_shape_and_replies_do_not_fake_progress():
    initial = {"messenger": [{"user": "Alice", "messages": []}]}
    predicates = [{"kind": "message", "to": "Alice", "message": "Hello"}]
    reply = {"messenger": [{"user": "Alice", "messages": [["Hello", "Alice", "time"]]}]}
    sent = {"messenger": [{"user": "Alice", "messages": [["Hello", "you", "time"]]}]}
    assert progress_fraction(predicates, initial, reply) == 0
    assert progress_fraction(predicates, initial, sent) == 1
