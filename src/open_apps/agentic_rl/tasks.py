"""Task manifests, leakage-aware splits and observable business predicates.

Environment seeds control replay randomness; they do not generate new tasks.
The curriculum generator explicitly creates different goals and parameters.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .types import TaskSpec


def _semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _semantic(v) for k, v in value.items()
                if k not in {"goal", "goal_style", "task_id", "_convert_"}}
    if isinstance(value, list):
        return [_semantic(v) for v in value]
    return value


def family_id(task_config: dict[str, Any]) -> str:
    """Group paraphrases by the underlying task, including nested subgoals."""
    canonical = json.dumps(_semantic(task_config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def validate_task(task: TaskSpec) -> None:
    if not task.task_id or not task.goal or task.max_steps <= 0:
        raise ValueError("Task requires an id, goal and positive max_steps")
    if task.task_config.get("goal") != task.goal:
        raise ValueError(f"Goal/config mismatch for {task.task_id}")
    actual_family = family_id(task.task_config)
    if task.family_id and task.family_id != actual_family:
        raise ValueError(f"Incorrect semantic family_id for {task.task_id}")
    task.family_id = actual_family


def load_manifest(path: str | Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = TaskSpec(**json.loads(line))
            validate_task(task)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def save_manifest(tasks: list[TaskSpec], path: str | Path) -> None:
    seen: set[str] = set()
    lines = []
    for task in tasks:
        validate_task(task)
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        seen.add(task.task_id)
        lines.append(json.dumps(asdict(task), ensure_ascii=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def assign_grouped_splits(tasks: list[TaskSpec], seed: int = 0,
                          ratios: tuple[float, float, float] = (.8, .1, .1)
                          ) -> dict[str, list[TaskSpec]]:
    """Stable hash assignment: adding another paraphrase cannot change its split.

    These are instance-grouped splits, not claims of unseen-template generalization.
    Small datasets can produce empty validation/test splits.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Three non-negative ratios must sum to one")
    result: dict[str, list[TaskSpec]] = {"train": [], "dev": [], "test": []}
    for task in tasks:
        validate_task(task)
        digest = hashlib.sha256(f"{seed}:{task.family_id}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        split = "train" if draw < ratios[0] else "dev" if draw < sum(ratios[:2]) else "test"
        result[split].append(task)
    return result


def export_hydra_tasks(path: str | Path, *, max_steps: int = 30,
                       seed: int = 0, overrides: list[str] | None = None) -> list[TaskSpec]:
    """Read task YAML without starting apps; resolve local task-file defaults."""
    from omegaconf import OmegaConf

    def read(source: Path, active: set[Path]) -> dict:
        source = source.resolve()
        if source in active:
            raise ValueError("Cyclic task YAML defaults")
        cfg = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
        defaults = cfg.pop("defaults", None)
        if defaults is None:
            return cfg
        merged = {}
        for entry in defaults:
            if entry == "_self_":
                merged.update(cfg)
            elif isinstance(entry, str):
                merged.update(read(source.parent / f"{entry}.yaml", active | {source}))
            else:
                raise ValueError("Only local task-file defaults are supported")
        return merged

    tasks = []
    for key, cfg in read(Path(path), set()).items():
        task = TaskSpec(key, cfg["goal"], cfg, seed=seed, max_steps=max_steps,
                        overrides=list(overrides or []), progress_predicates=derive_predicates(cfg))
        validate_task(task)
        tasks.append(task)
    return tasks


def generate_curriculum(count: int = 100, seed: int = 0, max_steps: int = 30,
                        max_subgoals: int = 1) -> list[TaskSpec]:
    """Generate explicit add-todo/add-event/send-message tasks (1–3 apps).

    Uses only contacts present in the default seed. No hidden target entities
    are invented for delete/mark operations. All produced values appear in goal.
    """
    if count < 1 or not 1 <= max_subgoals <= 3:
        raise ValueError("count must be positive and max_subgoals in 1..3")
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        token = hashlib.sha256(f"curriculum:{seed}:{index}".encode()).hexdigest()[:10]
        kinds = rng.sample(["todo", "calendar", "message"], 1 + index % max_subgoals)
        subtasks = []
        for kind in kinds:
            if kind == "todo":
                title = f"Prepare project {token}"
                cfg = {"_target_": "open_apps.tasks.tasks.AddToDoTask",
                       "goal": f"Add '{title}' to my todo list, leaving it incomplete.",
                       "todo_name": title, "is_done": False}
            elif kind == "calendar":
                title = f"Review {token}"
                when = (date(2027, 1, 1) + timedelta(days=rng.randrange(365))).isoformat()
                cfg = {"_target_": "open_apps.tasks.tasks.AddEventTask",
                       "goal": f"Add '{title}' to my calendar on {when}.",
                       "title": title, "date": when, "description": None,
                       "location": None, "url": None, "invitees": []}
            else:
                contact = rng.choice(["Alice", "Bob", "Charlie"])
                message = f"Project {token} is ready for review."
                cfg = {"_target_": "open_apps.tasks.tasks.SendMessageTask",
                       "goal": f"Send {contact} the message '{message}'",
                       "to": contact, "message": message}
            subtasks.append(cfg)
        cfg = subtasks[0] if len(subtasks) == 1 else {
            "_target_": "open_apps.tasks.tasks.CompositeTask", "_convert_": "all",
            "goal": " ".join(t["goal"] for t in subtasks), "subtasks": subtasks}
        task = TaskSpec(f"curriculum_{seed}_{index:06d}", cfg["goal"], cfg,
                        seed=seed, max_steps=max_steps,
                        progress_predicates=derive_predicates(cfg))
        validate_task(task)
        tasks.append(task)
    return tasks


def derive_predicates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Only explicit supported predicates; never reuse full-state subtask checks."""
    cls = cfg.get("_target_", "").rsplit(".", 1)[-1]
    if cls == "CompositeTask":
        return [p for child in cfg["subtasks"] for p in derive_predicates(child)]
    if cls == "AddToDoTask":
        return [{"kind": "add", "app": "todo", "item": {"title": cfg["todo_name"], "done": bool(cfg["is_done"])}}]
    if cls == "AddEventTask":
        item = {k: cfg.get(k) or ([] if k == "invitees" else "")
                for k in ("title", "date", "description", "location", "url", "invitees")}
        return [{"kind": "add", "app": "calendar", "item": item}]
    if cls in {"DeleteToDoTask", "RemoveEventTask", "RemoveLandmarkTask"}:
        app, item = ({"DeleteToDoTask": ("todo", {"title": cfg.get("todo_name")}),
                      "RemoveEventTask": ("calendar", {"title": cfg.get("title"), "date": cfg.get("date")}),
                      "RemoveLandmarkTask": ("map", {"name": cfg.get("name")})})[cls]
        return [{"kind": "remove", "app": app, "item": item}]
    if cls == "MarkToDoDoneTask":
        return [{"kind": "mark", "app": "todo", "item": {"title": cfg["todo_name"], "done": True}}]
    if cls == "SendMessageTask":
        return [{"kind": "message", "to": cfg["to"], "message": cfg["message"]}]
    # Navigation and approximate map placement have terminal rewards only.
    return []


def _normalized(value: Any) -> Any:
    from open_apps.tasks.tasks import StringSimilarityOperator
    if isinstance(value, str):
        return StringSimilarityOperator.normalize_string(value)
    if isinstance(value, list):
        return sorted((_normalized(v) for v in value), key=str)
    return value


def _matches(actual: dict, expected: dict) -> bool:
    for key, value in expected.items():
        actual_value = actual.get(key)
        if key == "invitees":
            actual_value = actual_value or []
            if isinstance(actual_value, str):
                actual_value = [s.strip() for s in actual_value.split(",") if s.strip()]
        elif key == "done":
            actual_value = actual_value in (True, 1, "1")
        else:
            actual_value = actual_value or ""
        if _normalized(actual_value) != _normalized(value):
            return False
    return True


def progress_fraction(predicates: list[dict], initial: dict, current: dict) -> float:
    """Current retained progress, not cumulative credit for repeated completion."""
    if not predicates:
        return 0.0
    completed = 0
    # Count identical requested effects only once, so one mutation cannot satisfy
    # multiple requested identical additions/sends.
    occurrences: dict[str, int] = {}
    for predicate in predicates:
        signature = json.dumps(predicate, sort_keys=True)
        occurrences[signature] = occurrences.get(signature, 0) + 1
        needed = occurrences[signature]
        kind = predicate["kind"]
        if kind == "message":
            def count_messages(state):
                count = 0
                for contact in state["messenger"]:
                    if contact["user"] == predicate["to"]:
                        for message in contact["messages"]:
                            # Captured legacy fixtures use False for outgoing;
                            # the live endpoint emits sender='you'.
                            if (len(message) >= 2 and (message[1] is False or message[1] == "you")
                                    and _normalized(message[0]) == _normalized(predicate["message"])):
                                count += 1
                return count
            completed += count_messages(current) - count_messages(initial) >= needed
        elif kind in {"add", "remove", "mark"}:
            app, item = predicate["app"], predicate["item"]
            before = sum(_matches(row, item) for row in initial[app])
            after = sum(_matches(row, item) for row in current[app])
            if kind == "add":
                completed += after - before >= needed
            elif kind == "remove":
                completed += before >= needed and before - after >= needed
            else:
                # Already done at reset is not newly achieved progress.
                completed += after - before >= needed
        else:
            raise ValueError(f"Unsupported progress predicate: {kind}")
    return completed / len(predicates)
