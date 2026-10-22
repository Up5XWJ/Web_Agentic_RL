"""Offline checks for the release's seed merge and integrity verification."""
import importlib.util
from pathlib import Path

import pytest

from open_apps.agentic_rl.tasks import load_manifest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rebuild_v4", ROOT / "scripts/rebuild_v4.py")
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


def test_seed_merge_matches_published_manifest(tmp_path):
    destination = tmp_path / "tasks.jsonl"
    count = rebuild.merge_seeds([
        ROOT / "data/agentic_rl", ROOT / "data/agentic_rl_long",
        ROOT / "data/seed_sources/imported-task-check",
    ], destination)
    assert count == 248
    assert destination.read_bytes() == (ROOT / "data/train/tasks.jsonl").read_bytes()
    assert rebuild.verify_manifest(destination, ROOT / "data/train/tasks.jsonl")["count"] == 248
    with pytest.raises(FileExistsError):
        rebuild.merge_seeds([], destination)


def test_verify_rejects_changed_manifest(tmp_path):
    other = tmp_path / "tasks.jsonl"
    other.write_text("")
    with pytest.raises(ValueError, match="differs"):
        rebuild.verify_manifest(other, ROOT / "data/train/tasks.jsonl")


def test_v4_splits_are_disjoint():
    seen_ids, seen_families = set(), set()
    for split, count in (("train", 1686), ("val", 184), ("test", 178)):
        tasks = load_manifest(ROOT / "data" / split / "tasks_v4.jsonl")
        assert len(tasks) == count
        ids = {task.task_id for task in tasks}
        families = {task.family_id for task in tasks}
        assert len(ids) == len(families) == count
        assert not ids & seen_ids
        assert not families & seen_families
        seen_ids.update(ids)
        seen_families.update(families)
