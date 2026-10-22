"""Process-isolated, fail-closed BrowserGym rollouts for online training.

The initial implementation creates a new process, database directory and browser
context on every reset. This favors correct group sampling over server reuse.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any

from .tasks import derive_predicates, progress_fraction, validate_task
from .types import InfrastructureError, ResetResult, StepResult, TaskSpec


_STATE_ENDPOINTS = {
    "todo": "/todo_all", "calendar": "/calendar_all",
    "map": "/maps/landmarks", "messenger": "/messages_all",
    "codeeditor": "/codeeditor_all",
}
_UI_ACTIONS = {"click", "fill", "focus", "select_option", "keyboard_press",
               "keyboard_type", "mouse_click", "mouse_dblclick", "scroll",
               "noop", "report_infeasible"}


def strict_state(base_url: str, *, session=None) -> dict[str, Any]:
    """No silent [] fallback; disabled online shop deliberately has no probe."""
    import requests
    client = session or requests
    state: dict[str, Any] = {}
    for name, endpoint in _STATE_ENDPOINTS.items():
        try:
            response = client.get(base_url.rstrip("/") + endpoint, timeout=10)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, (list, dict)):
                raise ValueError(f"Expected collection, got {type(value).__name__}")
            if name != "codeeditor" and not isinstance(value, list):
                raise ValueError("Expected a list of business records")
            state[name] = value
        except Exception as exc:
            raise InfrastructureError(f"Cannot read {name} business state: {type(exc).__name__}: {exc}") from exc
    state["online_shop"] = []
    return state


def initial_state_hash(state: dict[str, Any]) -> str:
    """Hash the scorer's canonical initial state, excluding URL/port metadata."""
    from open_apps.tasks.tasks import AppStateComparison
    canonical = AppStateComparison(state, state).state1
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_browser_action(action: str) -> str:
    """Return the name of a single literal UI call or parameterless finish()."""
    try:
        tree = ast.parse(action.strip(), mode="eval")
        call = tree.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Expected one tool call")
        if call.func.id not in _UI_ACTIONS | {"finish"}:
            raise ValueError(f"Unsupported UI action {call.func.id}")
        if call.func.id == "finish" and (call.args or call.keywords):
            raise ValueError("finish takes no arguments")
        for arg in call.args:
            ast.literal_eval(arg)
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("Expanded arguments are not permitted")
            ast.literal_eval(keyword.value)
        return call.func.id
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid browser action: {exc}") from exc


def _worker(connection, spec: TaskSpec, options: dict) -> None:
    # Descendant Playwright/browser processes can be reaped together on timeout.
    if hasattr(os, "setsid"):
        os.setsid()
    environment = server = None
    try:
        # Announce process-group ownership before slow imports/browser startup.
        # The parent can reap descendants even if the worker subsequently dies.
        connection.send(("started", {"pid": os.getpid(),
                                     "process_group_id": os.getpgrp() if hasattr(os, "getpgrp") else None}))
        import random
        random.seed(spec.seed)
        from hydra.utils import instantiate
        from agentlab.agents.dynamic_prompting import ObsFlags, make_obs_preprocessor
        from browsergym.core.env import BrowserEnv
        from browsergym.core.task import AbstractBrowserTask
        from browsergym.core.action.highlevel import HighLevelActionSet
        from open_apps.mcp.appserver import AppServer

        task = instantiate(spec.task_config)
        predicates = spec.progress_predicates or derive_predicates(spec.task_config)
        # Shop requires Java/downloaded artifacts and is outside this curriculum.
        overrides = [*spec.overrides, "apps.onlineshop.enable=False", f"seed={spec.seed}"]
        # Do not redirect stderr: uvicorn retains its logging stream beyond
        # construction, so a temporary redirected stream would later be closed.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            server = AppServer("todo", extra_overrides=overrides,
                               storage_directory=options["storage_directory"])
        http_errors: list[str] = []

        class StrictTask(AbstractBrowserTask):
            def __init__(self, seed=None):
                super().__init__(seed)
                self.viewport = {"width": options["viewport"][0], "height": options["viewport"][1]}
                self.slow_mo = 0
                self.timeout = 5000
                self.current_state = None
                self.submitted = False

            def setup(self, page):
                self.page = page
                def check_response(response):
                    if response.url.startswith(server.base_url + "/") and response.status >= 500:
                        http_errors.append(f"HTTP {response.status}: {response.url}")
                page.on("response", check_response)
                # Navigation tasks encode a required source app; other tasks
                # start at the cross-app landing page.
                source_app = spec.task_config.get("source_app")
                page.goto(server.url_for(source_app) if source_app else server.base_url)
                self.initial_state = strict_state(server.base_url)
                # All core apps are populated by the default configuration.
                expected_counts = {
                    "todo": len(server.config.apps.todo.init_todos),
                    "calendar": len(server.config.apps.calendar.events),
                    "messenger": len(server.config.apps.messenger.chat_history),
                }
                for app, expected in expected_counts.items():
                    if len(self.initial_state[app]) != expected:
                        raise InfrastructureError(f"Incomplete initialization for {app}")
                # Catch invalid/no-op generated tasks before sampling a policy.
                with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
                    if task.check_if_task_is_complete(self.initial_state, self.initial_state):
                        raise InfrastructureError("Task is already satisfied at reset")
                self.current_state = self.initial_state
                return spec.goal, {}

            def validate(self, page, chat_messages):
                self.current_state = strict_state(server.base_url)
                self.current_state["_url"] = page.url
                # Keep returning fresh observations even after the goal state
                # is reached. Only a model-authored finish requests a verdict.
                if not self.submitted:
                    return 0.0, False, "", {}
                with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
                    success = task.check_if_task_is_complete(self.initial_state, self.current_state)
                # An incorrect submission is a terminal policy failure, not
                # a retry with access to hidden task-verification feedback.
                return float(success), True, "", {}

            def teardown(self):
                pass

            def cheat(self, page, chat_messages):
                raise NotImplementedError

        action_set = HighLevelActionSet(subsets=["bid", "coord", "infeas"],
                                       multiaction=False, strict=True)
        environment = BrowserEnv(task_entrypoint=StrictTask, headless=options["headless"],
                                 action_mapping=action_set.to_python_code)
        preprocess = make_obs_preprocessor(ObsFlags(use_html=True, use_ax_tree=True,
                                                     use_screenshot=True, extract_visible_tag=True))
        observation, info = environment.reset(seed=spec.seed)
        if http_errors:
            raise InfrastructureError("Application failed during reset: " + http_errors[0])
        connection.send(("ok", ResetResult(preprocess(observation),
                         initial_state_hash(environment.task.initial_state),
                         {"seed": spec.seed, "isolation": "fresh_process_database_browser_context",
                          "worker_pid": os.getpid(), "base_url": server.base_url,
                          "storage_directory": options["storage_directory"],
                          "database_paths": {name: str(Path(app.database_path).resolve())
                                             for name, app in server.config.apps.items()
                                             if hasattr(app, "database_path")}})))
        steps = 0
        finished = False
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command != "step" or finished:
                raise InfrastructureError("Step outside an active episode")
            action_name = validate_browser_action(argument)
            submitted = action_name == "finish"
            environment.task.submitted = submitted
            # finish is an environment control action, not browser Python.
            # A zero-duration noop refreshes the page and invokes validation
            # without modifying business state or reusing a stale verdict.
            browser_action = "noop(wait_ms=0)" if submitted else argument
            observation, reward, terminated, truncated, info = environment.step(browser_action)
            if http_errors:
                raise InfrastructureError("Application request failed: " + http_errors[0])
            steps += 1
            success = submitted and reward >= 1
            truncated = bool(truncated or (steps >= spec.max_steps and not terminated))
            finished = bool(terminated or truncated)
            current = environment.task.current_state
            if current is None:
                raise InfrastructureError("BrowserGym did not validate business state")
            progress = progress_fraction(predicates, environment.task.initial_state, current)
            connection.send(("ok", StepResult(
                preprocess(observation), success=success, terminated=bool(terminated),
                truncated=truncated, progress=progress,
                info={"action_error": observation.get("last_action_error") or None,
                      "http_errors": [],
                      "submitted": submitted,
                      "submission_correct": bool(success) if submitted else None,
                      "step_count": steps})))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        if server is not None:
            try:
                server.close()
            except Exception:
                pass
        connection.close()


class BrowserGymEnvironment:
    def __init__(self, *, headless: bool = True, timeout_seconds: float = 120,
                 viewport: tuple[int, int] = (1024, 640)):
        if timeout_seconds <= 0 or min(viewport) <= 0:
            raise ValueError("Timeout and viewport dimensions must be positive")
        self.timeout_seconds = timeout_seconds
        self.options = {"headless": headless, "viewport": viewport}
        self._process = None
        self._connection = None
        self._storage_directory = None
        self._process_group_id = None

    def _receive(self):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                if self._connection is None or not self._connection.poll(max(0, deadline - time.monotonic())):
                    self.close()
                    raise InfrastructureError("Environment worker timed out")
                status, value = self._connection.recv()
            except (EOFError, OSError) as exc:
                self.close()
                raise InfrastructureError("Environment worker exited unexpectedly") from exc
            if status == "started":
                if value.get("pid") != self._process.pid:
                    self.close()
                    raise InfrastructureError("Environment worker identity mismatch")
                if value.get("process_group_id") == self._process.pid:
                    self._process_group_id = self._process.pid
                continue
            if status != "ok":
                self.close()
                raise InfrastructureError(value)
            return value

    def reset(self, task: TaskSpec) -> ResetResult:
        validate_task(task)
        self.close()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        # Parent ownership also permits cleanup after a worker crash/SIGKILL.
        self._storage_directory = tempfile.mkdtemp(prefix="openapps_rl_episode_")
        options = self.options | {"storage_directory": self._storage_directory}
        process = context.Process(target=_worker, args=(child, task, options), daemon=False)
        self._connection, self._process = parent, process
        try:
            process.start()
        except BaseException:
            self.close()
            raise
        finally:
            child.close()
        return self._receive()

    def step(self, action: str) -> StepResult:
        validate_browser_action(action)
        if self._connection is None:
            raise InfrastructureError("Call reset before step")
        try:
            self._connection.send(("step", action))
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise InfrastructureError("Environment worker is unavailable") from exc
        return self._receive()

    def close(self) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        storage_directory, self._storage_directory = self._storage_directory, None
        process_group_id, self._process_group_id = self._process_group_id, None
        if connection is not None:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            connection.close()
        if process is not None and process.pid is not None:
            process.join(timeout=3)
            if process_group_id is None and process.is_alive() and hasattr(os, "getpgid"):
                try:
                    if os.getpgid(process.pid) == process.pid:
                        process_group_id = process.pid
                except ProcessLookupError:
                    pass
            # Also clean descendants when the group leader has already exited.
            if process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                try:
                    if process_group_id is None:
                        process.terminate()
                except ProcessLookupError:
                    pass
                process.join(timeout=3)
            if process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        if process is not None:
            process.close()
        if storage_directory is not None:
            shutil.rmtree(storage_directory, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
