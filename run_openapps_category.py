#!/usr/bin/env python3
"""Run OpenApps tasks by application category with SiliconFlow Qwen3.6."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from dotenv import load_dotenv

from open_apps.tasks import _load_tasks_cfg


PROJECT_ROOT = Path(__file__).resolve().parent
AGENT_CONFIG = "qwen36-siliconflow"

CATEGORY_ALIASES = {
    "calendar": "calendar",
    "opencalendar": "calendar",
    "messenger": "messenger",
    "messages": "messenger",
    "openmessenger": "messenger",
    "maps": "maps",
    "map": "maps",
    "openmaps": "maps",
    "todo": "todo",
    "open-todo": "todo",
    "opentodo": "todo",
    "code_editor": "code_editor",
    "code-editor": "code_editor",
    "codeeditor": "code_editor",
    "opencodeeditor": "code_editor",
    "shop": "shop",
    "online_shop": "shop",
    "onlineshop": "shop",
    "openshop": "shop",
}

TASK_CLASS_CATEGORIES = {
    "AddEventTask": "calendar",
    "RemoveEventTask": "calendar",
    "AddToDoTask": "todo",
    "MarkToDoDoneTask": "todo",
    "DeleteToDoTask": "todo",
    "SendMessageTask": "messenger",
    "SavePlaceTask": "maps",
    "RemoveLandmarkTask": "maps",
}

TARGET_APP_CATEGORIES = {
    "calendar": "calendar",
    "messages": "messenger",
    "map": "maps",
    "todo": "todo",
    "codeeditor": "code_editor",
    "onlineshop": "shop",
}


def normalize_category(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    try:
        return CATEGORY_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(
            ["calendar", "messenger", "maps", "todo", "code_editor", "shop"]
        )
        raise argparse.ArgumentTypeError(
            f"unknown category {value!r}; choose one of: {choices}"
        ) from exc


def task_category(task_cfg) -> str | None:
    class_name = task_cfg.get("_target_", "").rsplit(".", 1)[-1]
    if class_name == "NavigateToAppTask":
        return TARGET_APP_CATEGORIES.get(task_cfg.get("target_app"))
    return TASK_CLASS_CATEGORIES.get(class_name)


def list_category_tasks(category: str) -> list[str]:
    tasks_cfg = _load_tasks_cfg()
    return [
        name
        for name, task_cfg in tasks_cfg.items()
        if task_category(task_cfg) == category
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenApps by app category using SiliconFlow "
            "Qwen/Qwen3.6-35B-A3B. By default, only task index 0 runs."
        )
    )
    parser.add_argument(
        "--category",
        required=True,
        type=normalize_category,
        metavar="CATEGORY",
        help=(
            "calendar, messenger, maps, todo, code_editor, or shop; "
            "OpenCalendar/OpenMessenger/etc. aliases are accepted"
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="zero-based task number shown by --list-tasks (default: 0)",
    )
    selection.add_argument("--task-name", help="run one exact task key")
    selection.add_argument(
        "--all", action="store_true", help="run every task in this category"
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="print category task data and exit"
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--headed", action="store_true", help="show Chromium")
    browser.add_argument("--headless", action="store_true", help="hide Chromium")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without calling the API"
    )
    return parser


def print_tasks(category: str, task_names: Iterable[str]) -> None:
    tasks_cfg = _load_tasks_cfg()
    print(f"Category: {category}")
    for index, name in enumerate(task_names):
        cfg = tasks_cfg[name]
        goal = " ".join(str(cfg.get("goal", "")).split())
        print(f"[{index:>3}] {name}: {goal}")


def select_tasks(args, available: list[str], parser: argparse.ArgumentParser) -> list[str]:
    if args.all:
        return available
    if args.task_name is not None:
        if args.task_name not in available:
            parser.error(
                f"task {args.task_name!r} is not in category {args.category!r}; "
                "use --list-tasks to inspect valid names"
            )
        return [args.task_name]
    if not 0 <= args.task_index < len(available):
        parser.error(
            f"--task-index must be in [0, {len(available) - 1}] for "
            f"category {args.category!r}"
        )
    return [available[args.task_index]]


def shop_environment(env: dict[str, str], parser: argparse.ArgumentParser) -> None:
    shop_root = PROJECT_ROOT / "src/open_apps/apps/onlineshop_app"
    java_home = shop_root / "java/jdk-21.0.1"
    required = [
        shop_root / "data/items_shuffle_1000.json",
        shop_root / "data/items_ins_v2_1000.json",
        shop_root / "search_engine/indexes",
        java_home / "bin/java",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        parser.error(
            "OpenShop is not installed. Missing: "
            + ", ".join(missing)
            + ". Run ./setup.sh and ./setup_pyserini.sh first."
        )
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = f"{java_home / 'bin'}:{env.get('PATH', '')}"


def command_for_task(args, task_name: str) -> list[str]:
    headless = not args.headed
    command = [
        sys.executable,
        str(PROJECT_ROOT / "launch_agent.py"),
        f"agent={AGENT_CONFIG}",
        f"task_name={task_name}",
        f"browsergym_env_args.headless={str(headless).lower()}",
        f"browsergym_env_args.max_steps={args.max_steps}",
        "use_wandb=False",
    ]
    if args.category == "shop":
        command.append("apps.onlineshop.enable=True")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than 0")

    available = list_category_tasks(args.category)
    if not available:
        parser.error(f"no benchmark tasks found for category {args.category!r}")
    if args.list_tasks:
        print_tasks(args.category, available)
        return 0

    selected = select_tasks(args, available, parser)
    env = os.environ.copy()
    load_dotenv(PROJECT_ROOT / ".env")
    env.update(os.environ)
    if not args.dry_run and not env.get("SILICONFLOW_API_KEY"):
        parser.error(
            "SILICONFLOW_API_KEY is missing; put it in OpenApps/.env or export it"
        )
    if args.category == "shop" and not args.dry_run:
        shop_environment(env, parser)

    for index, task_name in enumerate(selected, start=1):
        command = command_for_task(args, task_name)
        print(f"\n=== [{index}/{len(selected)}] {args.category}: {task_name} ===")
        print("Command:", " ".join(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if result.returncode != 0:
            print(f"Task {task_name!r} failed with exit code {result.returncode}.")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
