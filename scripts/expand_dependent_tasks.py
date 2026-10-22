"""Generate 600 dependency tasks with verified, materialized scene overrides.

This command performs only offline configuration and symbolic state checks. It
never launches OpenApps, a browser, a model, or an RL training process.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from open_apps.agentic_rl.dependent_tasks import TEMPLATES, build_case, validate_case
from open_apps.agentic_rl.task_scenes import SceneBuilder, TIERS, business_state, digest
from open_apps.agentic_rl.tasks import load_manifest, save_manifest


def file_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_splits(cases: list, seed: int) -> dict[str, list]:
    """8/1/1 scenes per template and complexity tier; variants follow parents."""
    groups = defaultdict(list)
    for case in cases:
        groups[(case.metadata["template_id"], case.metadata["complexity_tier"])].append(case)
    result = {name: [] for name in ("train", "val", "test")}
    seen_parents, seen_scenes = set(), set()
    for rows in groups.values():
        if len(rows) != 10:
            raise ValueError("Expected ten independent scenes per template/tier stratum")
        rows.sort(key=lambda case: digest([seed, case.metadata["parent_case_id"]]))
        for index, case in enumerate(rows):
            parent, scene = case.metadata["parent_case_id"], case.metadata["scene_id"]
            if parent in seen_parents or scene in seen_scenes:
                raise ValueError("Repeated parent/scene; do not split related variants independently")
            seen_parents.add(parent)
            seen_scenes.add(scene)
            split = "train" if index < 8 else "val" if index == 8 else "test"
            case.metadata["split"] = split
            result[split].append(case)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--base-version", default="v3")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--generation-seed", type=int, default=20261010)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--environment-seed", type=int, default=42)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (args.version, args.base_version)):
        parser.error("Use simple alphanumeric version labels")
    names = ("train", "val", "test")
    sources = {split: args.source_root / split / f"tasks_{args.base_version}.jsonl" for split in names}
    paths = {split: args.output_root / split / f"tasks_{args.version}.jsonl" for split in names}
    dependency_paths = {split: args.output_root / split / f"tasks_dependency_{args.version}.jsonl" for split in names}
    meta_path = args.output_root / "metadata" / f"tasks_{args.version}.json"
    registry_path = meta_path.with_name(f"tasks_{args.version}_registry.jsonl")
    examples_path = meta_path.with_name(f"tasks_{args.version}_examples.json")
    for path in [*paths.values(), *dependency_paths.values(), meta_path, registry_path, examples_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {path}")
    source_hashes = {split: file_hash(path) for split, path in sources.items()}
    base = {split: load_manifest(path) for split, path in sources.items()}
    builder = SceneBuilder(ROOT)
    cases = []
    totals = Counter()
    # Use the same Hydra composition as AppServer, with its fixed RL overrides.
    # No AppServer import is needed: importing it would also load web modules.
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for template in TEMPLATES:
            for index in range(50):
                case = build_case(builder, template, index, generation_seed=args.generation_seed,
                                  environment_seed=args.environment_seed)
                cfg = compose(config_name="config", overrides=[*case.task.overrides,
                    "apps.onlineshop.enable=False", f"seed={case.task.seed}",
                    "logs_dir=/tmp/openapps_offline_scene", "databases_dir=/tmp/openapps_offline_scene/databases"])
                actual_apps = OmegaConf.to_container(cfg.apps, resolve=True)
                if business_state(actual_apps) != business_state(case.apps):
                    raise ValueError(f"Hydra changed the seeded business state: {case.task.task_id}")
                for source in case.metadata["variants"]:
                    app = source["app"]
                    if actual_apps[app] != case.apps[app]:
                        raise ValueError(f"Variant was not faithfully materialized: {source['path']}")
                checks = validate_case(case)
                checks["runtime_hydra_composition"] = True
                case.metadata["checks"] = checks
                totals.update({name: value for name, value in checks.items() if type(value) is int})
                cases.append(case)
            print(json.dumps({"validated_template": template, "generated_so_far": len(cases)}), flush=True)
    additions = grouped_splits(cases, args.split_seed)
    splits = {split: sorted(base[split] + [case.task for case in additions[split]], key=lambda task: task.family_id)
              for split in names}
    tasks = [task for rows in splits.values() for task in rows]
    if len({task.task_id for task in tasks}) != len(tasks) or len({task.family_id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs/families or overlapping dataset splits")
    for split in names:
        retained = {task.task_id: asdict(task) for task in splits[split]}
        if any(retained.get(task.task_id) != asdict(task) for task in base[split]):
            raise ValueError("An existing task or split assignment was changed")
        if file_hash(sources[split]) != source_hashes[split]:
            raise ValueError("Source manifest changed during generation")
    new_by_id = {case.task.task_id: case.metadata for case in cases}
    registry = [new_by_id.get(task.task_id, {"task_id": task.task_id, "family_id": task.family_id,
                                           "origin": args.base_version, "split": split})
                for split, rows in splits.items() for task in rows]
    for split, rows in splits.items():
        save_manifest(rows, paths[split])
        save_manifest(sorted([case.task for case in additions[split]], key=lambda task: task.family_id), dependency_paths[split])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in registry), encoding="utf-8")
    examples = [{"task": asdict(case.task), "metadata": case.metadata} for case in cases if case.metadata["case_index"] == 40]
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    used_variants = Counter(source["path"] for case in cases for source in case.metadata["variants"])
    metadata = {
        "version": args.version, "base_version": args.base_version, "generated_tasks": len(cases),
        "sources": {split: {"path": str(path.relative_to(args.source_root)), "sha256": source_hashes[split],
                            "tasks": len(base[split])} for split, path in sources.items()},
        "generation_seed": args.generation_seed, "split_seed": args.split_seed,
        "environment_seed": args.environment_seed,
        "new_split_counts": {split: len(rows) for split, rows in additions.items()},
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "template_counts": dict(Counter(case.metadata["template_id"] for case in cases)),
        "complexity_counts": dict(Counter(case.metadata["complexity_tier"] for case in cases)),
        "variant_count_distribution": dict(sorted(Counter(case.metadata["variant_count"] for case in cases).items())),
        "variant_sources": dict(sorted(used_variants.items())),
        "split_strata": {split: {template: dict(Counter(c.metadata["complexity_tier"] for c in rows
                                                       if c.metadata["template_id"] == template))
                                 for template in TEMPLATES} for split, rows in additions.items()},
        "split_policy": f"Preserve {args.base_version} assignments; group unique source scenes/parents, stratify each template and complexity tier 8/1/1.",
        "generalization_claim": "Held-out parameter/scene instances of known templates and variant families; not unseen-template generalization.",
        "verification_scope": "Existing final-state validators; source-reading order is not graded. Answers are frozen from initial facts.",
        "background_policy": "0/1/2-4 selected YAMLs; at most one per app. Resolve + keys and interpolations offline; sample bounded background records.",
        "language_scope": "English task instructions and authoritative case notes; YAMLs supply German/Chinese UI and background content.",
        "supported_apps": ["todo", "calendar", "messenger", "maps", "start_page", "pop_ups"],
        "excluded_apps": {"onlineshop": "disabled in the RL environment", "code_editor": "not a task source or destination"},
        "max_steps_rule": "30 + 15 * read_count + 5 * complexity_tier; includes finish; not browser-validated",
        "checks": {"schema_and_constructors": "passed", "runtime_hydra_compositions": len(cases),
                   "unique_sources": "passed", "positive_and_negative_final_states": "passed",
                   "split_family_overlap": 0, "split_parent_or_scene_overlap": 0,
                   "retained_base_records_and_splits": "passed", **dict(totals),
                   "browser_execution": "not_performed", "model_execution": "not_performed",
                   "gpu_training": "not_performed", "tokenizer_context_fit": "not_measured"},
        "base_compatibility_note": "Existing disabled-shop and unvalidated map-save tasks remain inherited from v3; no automatic filtering.",
        "files": {split: {"path": str(path.relative_to(args.output_root)), "sha256": file_hash(path)} for split, path in paths.items()},
        "dependency_only_files": {split: {"path": str(path.relative_to(args.output_root)), "sha256": file_hash(path)}
                                  for split, path in dependency_paths.items()},
        "registry": {"path": str(registry_path.relative_to(args.output_root)), "sha256": file_hash(registry_path)},
        "examples": {"path": str(examples_path.relative_to(args.output_root)), "sha256": file_hash(examples_path)},
        "generator_sources": {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), ROOT / "src/open_apps/agentic_rl/dependent_tasks.py",
            ROOT / "src/open_apps/agentic_rl/task_scenes.py", ROOT / "src/open_apps/agentic_rl/tasks.py",
            ROOT / "src/open_apps/tasks/tasks.py", ROOT / "config/config.yaml",
            *sorted((ROOT / "config/apps").rglob("*.yaml"))]},
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("generated_tasks", "new_split_counts", "split_counts", "variant_count_distribution", "checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
