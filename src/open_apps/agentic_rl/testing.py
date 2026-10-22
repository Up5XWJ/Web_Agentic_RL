"""Explicitly non-trainable fixtures for laptop integration checks.

The scripted policy emits synthetic character IDs, NOT LLM token probabilities.
Its trajectories must never enter a policy update. The browser smoke uses actual
UI observations and normal browser actions; no business-state write shortcuts.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from typing import Any

from .environment import validate_browser_action
from .types import Generation, ResetResult, StepResult, TaskSpec


def smoke_task() -> TaskSpec:
    title = "OpenApps RL browser smoke"
    return TaskSpec(
        task_id="rl_browser_smoke", goal=f"Add '{title}' to my todo list.",
        task_config={"_target_": "open_apps.tasks.tasks.AddToDoTask", "todo_name": title, "is_done": False,
                     "goal": f"Add '{title}' to my todo list."},
        max_steps=8,
    )


class FixtureEnvironment:
    """Fill, click, then finish to check retries, memory and submission logging."""

    def reset(self, task: TaskSpec) -> ResetResult:
        self.task = task
        self.value = ""
        self.added = False
        self.added_title = None
        self.steps = 0
        self.closed = False
        self.finished = False
        return ResetResult(self._obs(), hashlib.sha256(b"empty-todo-fixture-v2").hexdigest())

    def _obs(self, error: str = "") -> dict[str, Any]:
        return {
            "url": "http://fixture.local/todo", "goal": self.task.goal,
            "pruned_html": f'<input bid="1" id="new-title" placeholder="New Todo" value="{html.escape(self.value)}"><button bid="2" id="submit-button">Add</button>' + ("<li>" + html.escape(self.added_title) + "</li>" if self.added_title is not None else ""),
            "axtree_txt": f"[1] textbox 'New Todo' value='{self.value}'\n[2] button 'Add'" + ("\nAdded: " + self.added_title if self.added_title is not None else ""),
            "focused_element_bid": "1", "last_action_error": error,
            "extra_element_properties": {"1": {"visibility": 1.0}, "2": {"visibility": 1.0}},
        }

    def step(self, action: str) -> StepResult:
        if self.finished or self.closed:
            raise ValueError("Step outside an active fixture episode")
        name = validate_browser_action(action)
        self.steps += 1
        call = ast.parse(action, mode="eval").body
        args = [ast.literal_eval(a) for a in call.args]
        error = ""
        if name == "fill" and args[0] == "1":
            self.value = args[1]
        elif name == "click" and args[0] == "2":
            self.added_title = self.value
            self.added = self.value == self.task.task_config["todo_name"]
        elif name not in {"noop", "finish", "report_infeasible"}:
            error = "Unsupported fixture action"
        submitted = name == "finish"
        success = submitted and self.added
        terminated = submitted or name == "report_infeasible"
        truncated = self.steps >= self.task.max_steps and not terminated
        self.finished = terminated or truncated
        return StepResult(
            self._obs(error), success=success, terminated=terminated, truncated=truncated,
            progress=float(self.added), info={"fixture": True, "action_error": error or None,
                "submitted": submitted, "submission_correct": success if submitted else None},
        )

    def close(self) -> None:
        self.closed = True


class _Elements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.texts: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.texts.append(data.strip())


class ScriptedTodoPolicy:
    """UI-only smoke policy, intentionally incapable of training a model."""

    def __init__(self, title: str, *, inject_invalid: bool = False):
        self.title = title
        self.inject_invalid = inject_invalid

    def generate(self, messages: list[dict[str, Any]]) -> Generation:
        if self.inject_invalid:
            self.inject_invalid = False
            return self._generation(messages, "deliberately malformed response for retry test")
        content = messages[1]["content"]
        text = content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content)
        marker = re.search(r'"automatic_assessment"\s*:\s*', text)
        if marker is None:
            raise ValueError("Smoke policy requires the harness assessment in the prompt")
        assessment, _ = json.JSONDecoder().raw_decode(text[marker.end():])
        current_dom = text.split("# Current DOM view", 1)[-1].split("# Current AXTree", 1)[0]
        tree = text.split("# Current AXTree", 1)[-1].split("# Loop guard", 1)[0]
        parser = _Elements()
        parser.feed(current_dom)
        # The title in an input's value is not evidence of an added todo.
        # Submit only after seeing the rendered result in the next observation.
        action = {"name": "finish"} if self.title in parser.texts else None
        for tag, attrs in parser.elements:
            if action is not None:
                break
            if tag == "input" and (attrs.get("id") == "new-title" or attrs.get("placeholder") == "New Todo") and attrs.get("bid"):
                if attrs.get("value") == self.title:
                    continue
                # Some flattened DOMs do not carry input values; the previous
                # action is an observation in the harness, not hidden task state.
                feedback = text.split("# Previous action and deterministic feedback", 1)[-1].split("# Current page metadata", 1)[0]
                try:
                    previous = json.loads(feedback.strip()).get("action") or {}
                except ValueError:
                    previous = {}
                if previous.get("name") == "fill" and previous.get("value") == self.title:
                    continue
                action = {"name": "fill", "bid": attrs["bid"], "value": self.title}
                break
        if action is None:
            for tag, attrs in parser.elements:
                if tag == "button" and attrs.get("id") == "submit-button" and attrs.get("bid"):
                    action = {"name": "click", "bid": attrs["bid"]}
                    break
        if action is None:
            # The initial OpenApps launcher is traversed through a visible link.
            match = re.search(r"\[([^\]]+)\]\s+link\s+['\"][^\n]*[Tt]odo[^\n]*", tree)
            if match:
                action = {"name": "click", "bid": match.group(1)}
        if action is None:
            raise ValueError("Smoke policy could not find the expected Todo UI; inspect the captured observation")
        submitting = action["name"] == "finish"
        payload = {
            "memory_next": {
                "plan": [{"id": "todo", "description": f"Add {self.title}",
                          "status": "verified" if submitting else "in_progress", "app": "todo",
                          "evidence": "The requested todo is visible in the list." if submitting else ""}],
                "current_subgoal_id": "todo", "pending": [] if submitting else ["Verify the added todo"],
                "verified_facts": [], "failures": [], "uncertainties": [], "transient_page_state": {},
            },
            "previous_action_assessment": assessment,
            "decision": {"observation": "The requested todo is visible" if submitting else "The Todo controls are visible",
                         "current_subgoal": "Add the requested todo",
                         "reason": "Submit the observed result" if submitting else "Use the visible form",
                         "expected_result": "The environment verifies the submission" if submitting else "The form is filled or the todo is added"},
            "action": action,
        }
        return self._generation(messages, json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _generation(messages, text) -> Generation:
        # These IDs only exercise persistence/masking code; metadata ensures
        # that no trainer can mistake them for sampled model tokens.
        ids = [ord(c) for c in text]
        return Generation(text, [ord(c) for c in json.dumps(messages, ensure_ascii=False)], ids,
                          [0.0] * len(ids), "scripted-smoke-v1", metadata={
                              "backend": "scripted", "trainable": False, "synthetic_token_ids": True,
                          })
