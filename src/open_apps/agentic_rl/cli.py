"""Laptop preparation, browser checks, reporting and GPU training commands."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _doctor(args) -> int:
    from open_apps import config_dir

    modules = {"browsergym": "browsergym", "agentlab": "agentlab", "playwright": "playwright",
               "hydra": "hydra-core", "yaml": "PyYAML", "torch": "torch", "ray": "ray", "verl": "verl", "vllm": "vllm",
               "transformers": "transformers", "tokenizers": "tokenizers", "torchvision": "torchvision"}
    deps = {}
    for module, distribution in modules.items():
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = "installed (no distribution metadata)" if available else None
        deps[module] = {"installed": available, "version": version}
    _print({"python": sys.version.split()[0], "config_dir": str(config_dir()),
            "dependencies": deps,
            "note": "GPU availability is not probed. verl/ray/vllm are needed only for GPU training/evaluation."})
    return 0 if all(deps[x]["installed"] for x in ("browsergym", "agentlab", "playwright", "hydra", "yaml")) else 1


def _prepare(args) -> int:
    from .tasks import assign_grouped_splits, export_hydra_tasks, generate_curriculum, save_manifest

    tasks = []
    for source in args.source:
        tasks.extend(export_hydra_tasks(source, max_steps=args.max_steps))
    count = args.curriculum if args.curriculum is not None else (0 if tasks else 90)
    if count:
        tasks.extend(generate_curriculum(count=count, seed=args.seed, max_steps=args.max_steps,
                                         max_subgoals=args.max_subgoals))
    if not tasks:
        raise ValueError("No tasks requested")
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs across input sources")
    splits = assign_grouped_splits(tasks, seed=args.seed)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / f"{name}.jsonl" for name in splits}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Manifests already exist in {destination}; choose a new output directory")
    for name, items in splits.items():
        save_manifest(items, paths[name])
    metadata = {"seed": args.seed, "max_steps": args.max_steps,
                "sources": [str(Path(p).resolve()) for p in args.source], "curriculum_count": count,
                "max_subgoals": args.max_subgoals,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "split_policy": "grouped_by_underlying_task_instance_not_goal_wording",
                "generalization_claim": "parameter-instance split; not an unseen-template benchmark"}
    (destination / "manifest_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print({"output": str(destination), **metadata})
    return 0


def _smoke(args) -> int:
    from .artifacts import write_episodes, write_report
    from .harness import HarnessConfig
    from .rollout import sample_group
    from .testing import FixtureEnvironment, ScriptedTodoPolicy, smoke_task

    if args.use_screenshot and args.environment != "browser":
        raise ValueError("Screenshot smoke requires --environment browser; the fixture has no real screenshot")
    task = smoke_task()
    policy = ScriptedTodoPolicy(task.task_config["todo_name"], inject_invalid=args.inject_invalid)
    if args.environment == "browser":
        from .environment import BrowserGymEnvironment
        factory = lambda: BrowserGymEnvironment(headless=not args.headed, timeout_seconds=args.timeout)
    else:
        factory = FixtureEnvironment
    episodes = sample_group(task, policy, factory, group_size=args.group_size,
                            harness_config=HarnessConfig(use_screenshot=args.use_screenshot))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output or f"outputs/agentic_rl/smoke-{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    write_episodes(output / "episodes.jsonl", episodes)
    report = write_report(output / "report.json", episodes)
    _print({"output": str(output), "environment": args.environment,
            "warning": "Scripted smoke test only: no LLM, GPU training or learned-policy performance claim.", **report})
    return 0 if episodes and all(e.success and not e.infrastructure_error for e in episodes) else 1


def _report(args) -> int:
    from .artifacts import read_episodes, summarize
    episodes = []
    for path in args.episodes:
        episodes.extend(read_episodes(path))
    _print(summarize(episodes))
    return 0


def _training(args) -> int:
    from .train import load_config, run, validate_config
    config = load_config(args.config, overrides=args.override)
    if args.command == "check":
        validate_config(config, check_paths=True)
        _print({"config": str(Path(args.config).resolve()), "valid": True,
                "gpu_started": False, "note": "Configuration and manifests checked; GPU runtime is not validated."})
    else:
        _print(run(config, evaluate_only=args.command == "evaluate"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openapps-rl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check dependencies without initializing CUDA")
    doctor.set_defaults(handler=_doctor)
    prepare = commands.add_parser("prepare", help="Generate/export tasks and leak-resistant grouped splits")
    prepare.add_argument("--source", action="append", default=[], help="Task YAML; may be repeated")
    prepare.add_argument("--curriculum", type=int, default=None, help="Number of generated tasks; defaults to 90 without sources")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--max-steps", type=int, default=30)
    prepare.add_argument("--max-subgoals", type=int, choices=[1, 2, 3], default=1,
                         help="Maximum generated cross-app subgoals (start with 1)")
    prepare.add_argument("--output", default="data/agentic_rl")
    prepare.set_defaults(handler=_prepare)
    smoke = commands.add_parser("smoke", help="Run the complete harness with a non-trainable scripted policy")
    smoke.add_argument("--environment", choices=["fixture", "browser"], default="fixture")
    smoke.add_argument("--group-size", type=int, default=2)
    smoke.add_argument("--inject-invalid", action="store_true", help="Exercise rejected-output retry recording")
    smoke.add_argument("--headed", action="store_true")
    smoke.add_argument("--use-screenshot", action="store_true", help="Record real PNG prompt inputs (requires --environment browser)")
    smoke.add_argument("--timeout", type=float, default=120)
    smoke.add_argument("--output")
    smoke.set_defaults(handler=_smoke)
    report = commands.add_parser("report", help="Summarize previously recorded JSONL episodes")
    report.add_argument("episodes", nargs="+")
    report.set_defaults(handler=_report)
    for name, help_text in (("check", "Validate training configuration without importing veRL/CUDA"),
                            ("train", "Run real online veRL training (GPU host)"),
                            ("evaluate", "Evaluate the configured model/checkpoint with the same harness (GPU host)")):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--config", default="config/agentic_rl/verl.yaml")
        cmd.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
        cmd.set_defaults(handler=_training)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"openapps-rl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
