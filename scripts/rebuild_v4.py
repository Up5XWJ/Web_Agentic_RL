"""Rebuild seeds and v2 -> v3 -> v4 offline, then verify published manifest hashes.

No model calls, browser sessions or GPU training. Outputs must be in a new or
empty directory; the published data is never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_apps.agentic_rl.tasks import load_manifest, save_manifest


def merge_seeds(source_dirs: list[Path], destination: Path) -> int:
    """Match historical first-occurrence semantic deduplication and output order."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    unique = {}
    for directory in source_dirs:
        for split in ("train", "dev", "test"):
            for task in load_manifest(directory / f"{split}.jsonl"):
                unique.setdefault(task.family_id, replace(task, task_id=f"web_{task.family_id}"))
    save_manifest([unique[key] for key in sorted(unique)], destination)
    return len(unique)


def verify_manifest(actual: Path, expected: Path) -> dict:
    digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    if digest != hashlib.sha256(expected.read_bytes()).hexdigest():
        raise ValueError(f"Rebuilt manifest differs from published data: {expected}")
    return {"count": len(load_manifest(actual)), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New or empty directory outside the published data directory")
    args = parser.parse_args()
    output = args.output_root.resolve()
    published = (ROOT / "data").resolve()
    if output == published or output.is_relative_to(published):
        parser.error("Use a directory outside the published data directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(script: str, *arguments: str) -> None:
        subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                       cwd=ROOT, env=env, check=True)

    sources = [output / "agentic_rl", output / "agentic_rl_long",
               output / "seed_sources/imported-task-check"]
    for directory, count, subgoals in ((sources[0], 90, 1), (sources[1], 30, 3)):
        run("agentic_rl.py", "prepare", "--curriculum", str(count), "--seed", "42",
            "--max-steps", "30", "--max-subgoals", str(subgoals), "--output", str(directory))
    run("agentic_rl.py", "prepare", "--source", "config/tasks/all_tasks.yaml",
        "--source", "config/tasks/longer_horizon.yaml", "--curriculum", "0",
        "--seed", "0", "--max-steps", "30", "--output", str(sources[2]))

    # The last source historically lived under outputs/. Its public snapshot now
    # lives under data/seed_sources; old provenance paths are historical labels.
    for source in sources:
        for split in ("train", "dev", "test"):
            relative = source.relative_to(output) / f"{split}.jsonl"
            verify_manifest(output / relative, published / relative)
    base = output / "train/tasks.jsonl"
    merge_seeds(sources, base)
    results = {"seed": verify_manifest(base, published / "train/tasks.jsonl")}
    run("scripts/expand_agentic_rl_tasks.py", "--base-manifest", str(base),
        "--output-root", str(output))
    run("scripts/expand_map_tasks.py", "--source-root", str(output),
        "--output-root", str(output))
    run("scripts/expand_dependent_tasks.py", "--source-root", str(output),
        "--output-root", str(output))
    for version in ("v2", "v3", "v4"):
        results[version] = {}
        for split in ("train", "val", "test"):
            relative = Path(split) / f"tasks_{version}.jsonl"
            results[version][split] = verify_manifest(output / relative, published / relative)
    for split in ("train", "val", "test"):
        relative = Path(split) / "tasks_dependency_v4.jsonl"
        verify_manifest(output / relative, published / relative)
    print(json.dumps({"verified": results, "output_root": str(output)}, indent=2))


if __name__ == "__main__":
    main()
