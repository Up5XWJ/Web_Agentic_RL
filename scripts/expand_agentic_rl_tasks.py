"""Expand task manifests offline; never run a browser, policy, or rollout."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra.utils import instantiate

from open_apps.agentic_rl.tasks import (
    assign_grouped_splits, generate_curriculum, load_manifest, save_manifest,
)


def leaf_types(config: dict) -> list[str]:
    name = config["_target_"].rsplit(".", 1)[-1]
    if name == "CompositeTask":
        return [name for child in config["subtasks"] for name in leaf_types(child)]
    return [name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "data/train/tasks.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--generation-seed", type=int, default=20261007)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--environment-seed", type=int, default=42)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.version):
        parser.error("version must contain only letters, digits, underscores or hyphens")
    if args.count < 1:
        parser.error("count must be positive")

    paths = {split: args.output_root / split / f"tasks_{args.version}.jsonl"
             for split in ("train", "val", "test")}
    metadata_path = args.output_root / "metadata" / f"tasks_{args.version}.json"
    registry_path = metadata_path.with_name(f"tasks_{args.version}_registry.jsonl")
    for path in [*paths.values(), metadata_path, registry_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {path}")

    base = load_manifest(args.base_manifest)
    base_families = {task.family_id for task in base}
    if len(base_families) != len(base):
        raise ValueError("Base manifest must contain unique task families")
    generated = generate_curriculum(count=args.count, seed=args.generation_seed, max_subgoals=3)
    budgets = {1: 30, 2: 45, 3: 60}
    generated = [replace(task, task_id=f"web_{task.family_id}", seed=args.environment_seed,
                         max_steps=budgets[len(leaf_types(task.task_config))])
                 for task in generated]
    new_families = {task.family_id for task in generated}
    if len(new_families) != args.count or new_families & base_families:
        raise ValueError("Generated tasks overlap; choose a different generation seed")

    # Environment and split seeds are independent of task parameter generation.
    new_splits = assign_grouped_splits(generated, seed=args.split_seed)
    splits = {"train": base + new_splits["train"],
              "val": new_splits["dev"], "test": new_splits["test"]}
    seen_ids, seen_families = set(), set()
    for split, tasks in splits.items():
        tasks.sort(key=lambda task: task.family_id)
        for task in tasks:
            if task.task_id in seen_ids or task.family_id in seen_families:
                raise ValueError("Duplicate task or overlapping dataset splits")
            seen_ids.add(task.task_id)
            seen_families.add(task.family_id)
            # Constructors only: no environment startup or model execution.
            instantiate(task.task_config)
    if seen_families != base_families | new_families:
        raise ValueError("Task coverage mismatch")

    registry = [
        {"task_id": task.task_id, "family_id": task.family_id, "split": split,
         "origin": "base" if task.family_id in base_families else "generated",
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
        "version": args.version,
        "base_manifest": str(args.base_manifest.resolve().relative_to(ROOT))
        if args.base_manifest.resolve().is_relative_to(ROOT) else str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "base_tasks_retained_in_train": len(base),
        "generated_tasks": len(generated),
        "generation_seed": args.generation_seed,
        "split_seed": args.split_seed,
        "environment_seed": args.environment_seed,
        "max_steps_by_generated_subgoal_count": budgets,
        "new_split_counts": {"train": len(new_splits["train"]), "val": len(new_splits["dev"]),
                             "test": len(new_splits["test"])},
        "split_counts": {split: len(tasks) for split, tasks in splits.items()},
        "split_subgoal_counts": {split: dict(sorted(Counter(len(leaf_types(task.task_config))
                                                           for task in tasks).items()))
                                  for split, tasks in splits.items()},
        "generated_subgoal_counts": dict(sorted(Counter(len(leaf_types(task.task_config))
                                                        for task in generated).items())),
        "split_policy": "Existing tasks pinned to train; new families use a stable 80/10/10 hash split.",
        "generalization_claim": "Parameter-instance split only; no held-out templates or compositions.",
        "checks": {"schema_and_constructors": "passed", "unique_ids_and_families": "passed",
                   "split_family_overlap": 0, "browser_execution": "not_performed",
                   "model_execution": "not_performed", "gpu_training": "not_performed"},
        "files": {split: {"path": str(path.relative_to(args.output_root)), "sha256": sha256(path)}
                  for split, path in paths.items()},
        "registry": {"path": str(registry_path.relative_to(args.output_root)),
                     "sha256": sha256(registry_path)},
        "generator": {"function": "open_apps.agentic_rl.tasks.generate_curriculum",
                      "max_subgoals": 3, "source_sha256": sha256(ROOT / "src/open_apps/agentic_rl/tasks.py"),
                      "script_sha256": sha256(Path(__file__))},
        "base_compatibility_note": "Base tasks are retained unchanged; existing environment incompatibilities are not repaired by expansion.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"generated_tasks": len(generated), "new_split_counts": metadata["new_split_counts"],
                      "split_counts": metadata["split_counts"],
                      "generated_subgoal_counts": metadata["generated_subgoal_counts"],
                      "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
