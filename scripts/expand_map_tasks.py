"""Add offline map tasks while preserving every existing train/val/test assignment."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import re

from expand_agentic_rl_tasks import ROOT, leaf_types, sha256
from hydra.utils import instantiate
import yaml

from open_apps.agentic_rl.tasks import (
    assign_grouped_splits, derive_predicates, family_id, generate_curriculum,
    load_manifest, save_manifest,
)
from open_apps.agentic_rl.types import TaskSpec


def remove_place(name: str) -> dict:
    return {"_target_": "open_apps.tasks.tasks.RemoveLandmarkTask",
            "goal": f"Remove '{name}' from my favorite places.", "name": name}


def combine(children: list[dict]) -> dict:
    if len(children) == 1:
        return children[0]
    return {"_target_": "open_apps.tasks.tasks.CompositeTask", "_convert_": "all",
            "goal": " ".join(child["goal"] for child in children), "subtasks": children}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--base-version", default="v2")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--generation-seed", type=int, default=20261008)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--environment-seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", value)
                             for value in (args.version, args.base_version)):
        parser.error("Use a positive count and simple alphanumeric version labels")

    paths = {split: args.output_root / split / f"tasks_{args.version}.jsonl"
             for split in ("train", "val", "test")}
    metadata_path = args.output_root / "metadata" / f"tasks_{args.version}.json"
    registry_path = metadata_path.with_name(f"tasks_{args.version}_registry.jsonl")
    for path in [*paths.values(), metadata_path, registry_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {path}")
    sources = {split: args.source_root / split / f"tasks_{args.base_version}.jsonl"
               for split in paths}
    splits = {split: load_manifest(path) for split, path in sources.items()}
    old_tasks = [task for tasks in splits.values() for task in tasks]
    old_families = {task.family_id for task in old_tasks}
    if len(old_families) != len(old_tasks):
        raise ValueError("Base dataset has duplicate task families or overlapping splits")

    map_config_path = ROOT / "config/apps/maps/content/default.yaml"
    places = yaml.safe_load(map_config_path.read_text(encoding="utf-8"))["saved_places"]
    names = sorted(place["name"] for place in places)
    if not names or len(set(names)) != len(names):
        raise ValueError("Default map must have uniquely named saved places")
    generated, kinds = [], {}
    seen = set(old_families)

    def add(config: dict, kind: str) -> None:
        fid = family_id(config)
        if len(generated) >= args.count or fid in seen:
            return
        task = TaskSpec(f"web_{fid}", config["goal"], config, family_id=fid,
                        seed=args.environment_seed,
                        max_steps=15 + 15 * len(leaf_types(config)),
                        progress_predicates=derive_predicates(config))
        generated.append(task)
        kinds[task.task_id] = kind
        seen.add(fid)

    # Each unordered subset is emitted once; reordering deletions is not a new task.
    for count in range(1, min(3, len(names)) + 1):
        for subset in combinations(names, count):
            add(combine([remove_place(name) for name in subset]), "remove_saved_places")
    for source in ("start_page", "todo", "calendar", "messages"):
        add({"_target_": "open_apps.tasks.tasks.NavigateToAppTask",
             "goal": "Navigate to the Maps app.", "source_app": source, "target_app": "map"},
            "map_navigation")
    for target, title in (("todo", "Todo"), ("calendar", "Calendar"), ("messages", "Messenger")):
        add({"_target_": "open_apps.tasks.tasks.NavigateToAppTask",
             "goal": f"Navigate from Maps to the {title} app.",
             "source_app": "map", "target_app": target}, "map_navigation")

    rng = random.Random(args.generation_seed)
    for task in generate_curriculum(count=args.count, seed=args.generation_seed, max_subgoals=3):
        if len(generated) == args.count:
            break
        children = task.task_config.get("subtasks", [task.task_config])
        add(combine([remove_place(rng.choice(names)), *children]), "map_cross_app")
    if len(generated) != args.count:
        raise ValueError("Insufficient new task families; use a different generation seed")

    additions = assign_grouped_splits(generated, seed=args.split_seed)
    additions["val"] = additions.pop("dev")
    for split in splits:
        splits[split] = sorted(splits[split] + additions[split], key=lambda task: task.family_id)
    all_tasks = [task for tasks in splits.values() for task in tasks]
    if (len({task.task_id for task in all_tasks}) != len(all_tasks)
            or len({task.family_id for task in all_tasks}) != len(all_tasks)):
        raise ValueError("Duplicate task IDs or overlapping task families")
    for task in all_tasks:
        instantiate(task.task_config)  # Pure constructors, no browser/model execution.

    registry = [
        {"task_id": task.task_id, "family_id": task.family_id, "split": split,
         "origin": "generated_map" if task.task_id in kinds else args.base_version,
         "kind": kinds.get(task.task_id, "retained"),
         "subgoal_count": len(leaf_types(task.task_config)),
         "operation_types": leaf_types(task.task_config)}
        for split, tasks in splits.items() for task in tasks
    ]
    for split, tasks in splits.items():
        save_manifest(tasks, paths[split])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in registry),
                             encoding="utf-8")
    metadata = {
        "version": args.version, "base_version": args.base_version,
        "sources": {split: {"path": str(path.relative_to(args.source_root)), "sha256": sha256(path),
                            "tasks": len(splits[split]) - len(additions[split])}
                    for split, path in sources.items()},
        "generated_tasks": len(generated), "generated_kind_counts": dict(Counter(kinds.values())),
        "new_split_counts": {split: len(tasks) for split, tasks in additions.items()},
        "split_counts": {split: len(tasks) for split, tasks in splits.items()},
        "generated_subgoal_counts": dict(sorted(Counter(len(leaf_types(task.task_config))
                                                        for task in generated).items())),
        "generation_seed": args.generation_seed, "split_seed": args.split_seed,
        "environment_seed": args.environment_seed, "max_steps_rule": "15 + 15 * subgoal_count",
        "split_policy": "Preserve all base assignments; new families use stable 80/10/10 hashing.",
        "generalization_claim": "Parameter-instance split; no held-out templates, places or compositions.",
        "subtask_dependencies": "Generated cross-app subtasks are independent final-state requirements.",
        "checks": {"schema_and_constructors": "passed", "split_family_overlap": 0,
                   "browser_execution": "not_performed", "model_execution": "not_performed"},
        "map_scope": {"saved_places": names, "operations": ["navigation", "remove_saved_place"],
                      "search_and_save": "not_generated", "route_planning": "not_generated",
                      "base_compatibility_issues": "Retained unchanged from the base dataset."},
        "files": {split: {"path": str(path.relative_to(args.output_root)), "sha256": sha256(path)}
                  for split, path in paths.items()},
        "registry": {"path": str(registry_path.relative_to(args.output_root)), "sha256": sha256(registry_path)},
        "generator_sources": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/expand_agentic_rl_tasks.py",
            ROOT / "src/open_apps/agentic_rl/tasks.py", map_config_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "generated_tasks", "generated_kind_counts", "new_split_counts", "split_counts")}, indent=2))


if __name__ == "__main__":
    main()
