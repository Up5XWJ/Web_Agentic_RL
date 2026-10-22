"""Text/multimodal execution harness with lossless policy-attempt provenance.

The harness owns validation and state, not policy optimization. Each model call
is recorded *before* interpreting its output. System repairs are never emitted
as model generations or substituted for the sampled token sequence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from agentlab.llm.llm_utils import ParseError

from open_apps.react_agent.action_guard import ActionGuard, ActionGuardConfig, SUPPORTED_ACTIONS, extract_valid_bids
from open_apps.react_agent.action_parser import BrowserGymActionSerializer, ReactResponseParser
from open_apps.react_agent.loop_guard import LoopGuard
from open_apps.react_agent.memory import MemoryConfig, MemoryManager
from open_apps.react_agent.prompt import DEFAULT_SYSTEM_PROMPT, ReactPromptBuilder
from open_apps.react_agent.schemas import ActionAssessment, ActionCall, AgentResponse, SchemaError
from open_apps.react_agent.spatial import PageExplorationTracker, extract_scroll_state
from open_apps.react_agent.verifier import OutcomeVerifier, observation_signature

from .types import AttemptRecord, ContextLimitError, Episode, Generation, InfrastructureError, Policy, TaskSpec
from .vision import screenshot_data_url


@dataclass
class HarnessConfig:
    max_attempts: int = 3  # Includes the first generation.
    memory_max_tokens: int = 1024
    memory_max_subgoals: int = 12
    memory_max_verified_facts: int = 20
    memory_max_failures: int = 3
    memory_max_uncertainties: int = 5
    max_same_action_retries: int = 2
    recent_turn_replay_size: int = 3
    offscreen_element_limit: int = 20
    coordinate_space: int = 1000
    allow_unfocused_keyboard_type: bool = False
    use_screenshot: bool = False
    fallback_wait_ms: int = 1000
    config_version: str = "multiview-memory-finish-v3"

    def __post_init__(self) -> None:
        if not isinstance(self.use_screenshot, bool):
            raise ValueError("use_screenshot must be a boolean.")
        for name in (
            "max_attempts", "memory_max_tokens", "memory_max_subgoals",
            "memory_max_verified_facts", "memory_max_failures", "memory_max_uncertainties",
            "max_same_action_retries", "coordinate_space",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("recent_turn_replay_size", "offscreen_element_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not 0 <= self.fallback_wait_ms <= 10_000:
            raise ValueError("fallback_wait_ms must be between 0 and 10000.")


@dataclass
class HarnessDecision:
    action: ActionCall
    browser_action: str
    fallback: bool
    response: AgentResponse | None = None


def _submission_wording(text: str) -> str:
    """Both observation modes use explicit, independently verified submission."""
    replacements = {
        "The environment, not you, decides final benchmark success after every action. If this turn is running, choose the next useful action rather than claiming completion.":
            'After each browser action, inspect the resulting observations. When all parts of the user goal are complete, submit action {"name":"finish"}. Ordinary actions never automatically finish the task. The environment independently verifies the current business state on finish. A valid finish ends the episode even when your claim is incorrect. Do not use finish as a progress-check tool. Reserve one action step for finish; reaching the step limit without submitting is not success.',
        "- report_infeasible:":
            '- finish: declare the entire task complete and end the episode for independent verification. Takes no parameters: {"name":"finish"}.\n- report_infeasible:',
        "|noop|report_infeasible": "|noop|finish|report_infeasible",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _text_observation_wording(text: str) -> str:
    """Remove visual claims/tools only when screenshots are disabled."""
    text = _submission_wording(text)
    replacements = {
        "The current DOM, AXTree, and screenshot are three EQUAL, independent, and complementary views of the same page. Always inspect all three; none is merely a fallback.":
            "The current DOM and AXTree are complementary TEXT observations. No screenshot or image is supplied to the policy. Use only the supplied text and spatial metadata.",
        "- Screenshot: visual layout, clipping, long-list boundaries, canvases, and evidence that useful content may lie outside the viewport.\n": "",
        "a fourth page view": "an additional observation view",
        "the three current views": "the two text views",
        "DOM, AXTree, screenshot, recent-turn replay": "DOM, AXTree, recent-turn replay",
        "Coordinate-action preference concerns execution only; it does not make the screenshot less important for reasoning.":
            "Coordinate clicks are disabled because no image is provided.",
        "all three current views": "both current text views",
        "all three views": "both text views",
        "all three equal current views: DOM, AXTree, and screenshot.": "both supplied text views: DOM and AXTree.",
        "Mouse and keyboard actions operate the rendered web page, not browser chrome that is absent from the screenshot.":
            "Keyboard actions operate the web page, not browser chrome.",
        "(equal observation view 1/3)": "(text observation view 1/2)",
        "(equal observation view 2/3)": "(text observation view 2/2)",
        "screenshot_viewport_pixels": "browser_viewport_pixels",
        "|mouse_click|mouse_dblclick": "",
        "For coordinate actions, use the normalized 0-": "For scroll deltas, use the normalized 0-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = text.splitlines()
    return "\n".join(line for line in lines if not line.startswith(("- mouse_click:", "- mouse_dblclick:")))


TEXT_SYSTEM_PROMPT = _text_observation_wording(DEFAULT_SYSTEM_PROMPT)
VISION_SYSTEM_PROMPT = _submission_wording(DEFAULT_SYSTEM_PROMPT)


class AgentActionSerializer(BrowserGymActionSerializer):
    """RL adds a submission control action; the legacy browser agent stays unchanged."""

    def serialize(self, action: ActionCall, viewport: tuple[int, int]) -> str:
        if action.name == "finish":
            return "finish()"
        return super().serialize(action, viewport)


def validate_generation(generation: Generation) -> None:
    """Fail closed for corrupted likelihood metadata, even in scripted tests."""
    if not isinstance(generation, Generation):
        raise InfrastructureError("Policy.generate must return a Generation.")
    if not isinstance(generation.text, str):
        raise InfrastructureError("Generation.text must be a string.")
    if not isinstance(generation.policy_version, str) or not generation.policy_version:
        raise InfrastructureError("Generation requires a non-empty policy_version.")
    for name in ("prompt_token_ids", "response_token_ids"):
        values = getattr(generation, name)
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise InfrastructureError(f"Generation.{name} must contain actual non-negative integer token IDs.")
    if len(generation.response_token_ids) != len(generation.old_log_probs):
        raise InfrastructureError("response_token_ids and old_log_probs lengths differ.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value > 1e-6
        for value in generation.old_log_probs
    ):
        raise InfrastructureError("old_log_probs must be finite non-positive token log probabilities.")


class MemoryHarness:
    """Reuse the existing ReAct components around an interchangeable policy."""

    def __init__(self, config: HarnessConfig | None = None):
        self.config = deepcopy(config or HarnessConfig())
        self.config.__post_init__()
        self.parser = ReactResponseParser()
        self.serializer = AgentActionSerializer(self.config.coordinate_space)
        self.action_guard = ActionGuard(ActionGuardConfig(
            coordinate_space=self.config.coordinate_space,
            allow_unfocused_keyboard_type=self.config.allow_unfocused_keyboard_type,
            allowed_actions=(SUPPORTED_ACTIONS if self.config.use_screenshot else
                             SUPPORTED_ACTIONS - {"mouse_click", "mouse_dblclick"}) | {"finish"},
        ))
        self.memory_manager = MemoryManager(MemoryConfig(
            max_tokens=self.config.memory_max_tokens,
            max_subgoals=self.config.memory_max_subgoals,
            max_verified_facts=self.config.memory_max_verified_facts,
            max_failures=self.config.memory_max_failures,
            max_uncertainties=self.config.memory_max_uncertainties,
        ))
        self.verifier = OutcomeVerifier()
        self.loop_guard = LoopGuard(repeat_limit=self.config.max_same_action_retries)
        self.exploration_tracker = PageExplorationTracker()
        self.prompt_builder = ReactPromptBuilder(
            # RL serializes its own immutable PNG below, rather than using the
            # inference client's message/image encoder from the legacy builder.
            client_type="vllm", use_html=True, use_axtree=True, use_screenshot=False,
            coordinate_space=self.config.coordinate_space,
            offscreen_element_limit=self.config.offscreen_element_limit,
        )
        self.memory = self.memory_manager.initialize()
        self.goal = ""
        self.previous_action: ActionCall | None = None
        self.previous_expected_result = ""
        self.assessment = ActionAssessment()
        self.recent_turns: list[dict[str, Any]] = []
        self.page_exploration: dict[str, Any] = {}

    def reset(self, task: TaskSpec, observation: dict[str, Any]) -> None:
        self.goal = task.goal
        self.max_steps = task.max_steps
        self.memory = self.memory_manager.initialize()
        self.previous_action = None
        self.previous_expected_result = ""
        self.assessment = ActionAssessment()
        self.recent_turns = []
        self.loop_guard.reset()
        self.exploration_tracker.reset()
        self.page_exploration = self.exploration_tracker.update(observation)

    def build_messages(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = self.prompt_builder.build(
            immutable_goal=self.goal, memory=self.memory, automatic_assessment=self.assessment,
            previous_action=self.previous_action, previous_expected_result=self.previous_expected_result,
            obs=observation, loop_context=self.loop_guard.prompt_context(),
            recent_turns=self.recent_turns, page_exploration=self.page_exploration,
        )
        if self.config.use_screenshot:
            return [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": _submission_wording(prompt.text) +
                     "\n# Current screenshot view (equal observation view 3/3)\n"
                     "This is the current browser viewport, not the whole page. "
                     "Use it together with DOM, AXTree and offscreen information."},
                    {"type": "image_url", "image_url": {
                        "url": screenshot_data_url(observation.get("screenshot")),
                    }},
                ]},
            ]
        return [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": _text_observation_wording(prompt.text)},
        ]

    def select_action(
        self, policy: Policy, observation: dict[str, Any], episode: Episode, step_index: int,
    ) -> HarnessDecision:
        messages = self.build_messages(observation)
        budget = (
            f"\n# Action budget\n{self.max_steps - step_index} action step(s) remain, including finish. "
            "Use the visible observations to decide when to submit; a completed goal still requires finish."
        )
        if isinstance(messages[1]["content"], str):
            messages[1]["content"] += budget
        else:
            messages[1]["content"][0]["text"] += budget
        for attempt_index in range(self.config.max_attempts):
            actual_messages = deepcopy(messages)
            try:
                generation = policy.generate(deepcopy(actual_messages))
            except Exception as exc:
                # No fabricated Generation: a failed service call has no policy tokens.
                episode.metadata.setdefault("model_errors", []).append({
                    "step_index": step_index, "attempt_index": attempt_index,
                    "messages": actual_messages, "error": str(exc), "error_type": type(exc).__name__,
                })
                raise
            if not isinstance(generation, Generation):
                raise InfrastructureError("Policy.generate must return a Generation.")
            record = AttemptRecord(step_index, attempt_index, actual_messages, deepcopy(generation))
            episode.attempts.append(record)
            if generation.metadata.get("trainable") is False:
                episode.trainable = False
            if generation.finish_reason in {"context_limit", "context_length_exceeded"}:
                record.error_type = "infrastructure"
                record.error = "Policy context budget exceeded."
                raise ContextLimitError(record.error)
            try:
                validate_generation(generation)
                if episode.policy_version and generation.policy_version != episode.policy_version:
                    raise InfrastructureError("Policy version changed within an episode.")
                episode.policy_version = generation.policy_version
            except InfrastructureError as exc:
                record.error_type = "infrastructure"
                record.error = str(exc)
                raise

            response, rejection = self._parse_and_validate(record, observation)
            if response is not None:
                self.memory = response.memory_next
                record.accepted = True
                record.applied_memory = deepcopy(self.memory.to_dict())
                return HarnessDecision(
                    response.action, self.serializer.serialize(response.action, self._viewport(observation)),
                    False, response,
                )
            # Every repair sees the actual failed output, not just an orphan
            # error message. The next environment step starts a fresh context.
            messages.extend([
                {"role": "assistant", "content": generation.text},
                {"role": "user", "content": f"Harness rejected this output ({record.error_type}): {rejection}\nReturn a complete corrected JSON response. Keep memory concise."},
            ])
        action = ActionCall(name="noop", wait_ms=self.config.fallback_wait_ms)
        return HarnessDecision(action, self.serializer.serialize(action, self._viewport(observation)), True)

    def _parse_and_validate(
        self, record: AttemptRecord, observation: dict[str, Any],
    ) -> tuple[AgentResponse | None, str | None]:
        generation = record.generation
        stage = "format"
        try:
            if generation.finish_reason in {"length", "max_tokens", "truncated"}:
                record.generation.metadata["rejection_reason"] = "output_truncated"
                raise ParseError("The generated response reached its output-token limit; provide shorter complete JSON.")
            if generation.finish_reason not in {"stop", "eos", "end_turn"}:
                raise ParseError(f"Unsupported completion finish_reason: {generation.finish_reason!r}.")
            payload = self.parser._extract_json(generation.text)
            if isinstance(payload.get("memory_next"), dict):
                record.raw_memory = deepcopy(payload["memory_next"])
            if isinstance(payload.get("action"), dict):
                record.action = deepcopy(payload["action"])
            response = self.parser.parse(generation.text)
            record.action = deepcopy(response.action.to_dict())
            if (
                response.previous_action_assessment.status != self.assessment.status
                or response.previous_action_assessment.page_changed != self.assessment.page_changed
            ):
                raise ParseError("previous_action_assessment must copy the supplied status and page_changed exactly.")
            response.previous_action_assessment = deepcopy(self.assessment)
            stage = "invalid_action"
            if response.action.name == "finish" and set(payload["action"]) != {"name"}:
                raise ParseError('finish takes no parameters; use {"name":"finish"}.')
            for values in (response.action.coordinate, response.action.delta, [response.action.wait_ms]):
                if values is not None and any(not math.isfinite(value) for value in values):
                    raise ParseError("Action numeric parameters must be finite.")
            self.action_guard.validate(response.action, observation)
            # Validate serialization before committing any memory state.
            self.serializer.serialize(response.action, self._viewport(observation))
            stage = "repeat"
            self.loop_guard.validate_candidate(response.action)
            stage = "format"
            response.memory_next = self.memory_manager.prepare_update(
                self.memory, response.memory_next, self.assessment,
                current_url=str(observation.get("url", "") or ""),
                focused_bid=str(observation.get("focused_element_bid", "") or ""),
                valid_bids=extract_valid_bids(observation),
            )
            return response, None
        except (ParseError, SchemaError, ValueError, OverflowError) as exc:
            record.error_type = stage
            record.error = str(exc)
            return None, record.error

    def observe_transition(
        self, before: dict[str, Any], after: dict[str, Any], decision: HarnessDecision, step_index: int,
        *, submission_success: bool | None = None,
    ) -> ActionAssessment:
        """Assess immediately, including the final action of an episode."""
        if decision.action.name == "finish" and submission_success is not None:
            assessment = ActionAssessment(
                status="success" if submission_success else "failed",
                evidence="Submission verified against business state." if submission_success
                else "Submitted state does not satisfy the full task.",
                page_changed=observation_signature(before) != observation_signature(after),
            )
        else:
            assessment = self.verifier.assess(before, after, decision.action)
        self.loop_guard.record(decision.action, observation_signature(before), observation_signature(after), assessment)
        self.page_exploration = self.exploration_tracker.update(
            after, previous_obs=before, previous_action=decision.action, assessment=assessment,
        )
        model_decision = decision.response.decision if decision.response else None
        expected = model_decision.expected_result if model_decision else "Wait and re-observe after rejected model output."
        if self.config.recent_turn_replay_size:
            self.recent_turns.append({
                "turn": step_index, "url_before": str(before.get("url", "")),
                "url_after": str(after.get("url", "")),
                "subgoal": model_decision.current_subgoal if model_decision else (self.memory.current_subgoal_id or "unknown"),
                "reason": model_decision.reason if model_decision else "Harness fallback after validation failures.",
                "action": decision.action.to_dict(), "fallback": decision.fallback,
                "expected_result": expected, "assessment": assessment.status,
                "assessment_evidence": assessment.evidence, "page_changed": assessment.page_changed,
                "scroll_before": extract_scroll_state(before), "scroll_after": extract_scroll_state(after),
            })
            self.recent_turns = self.recent_turns[-self.config.recent_turn_replay_size:]
        self.previous_action = decision.action
        self.previous_expected_result = expected
        self.assessment = assessment
        return assessment

    @staticmethod
    def _viewport(observation: dict[str, Any]) -> tuple[int, int]:
        screenshot = observation.get("screenshot")
        if screenshot is not None and hasattr(screenshot, "shape"):
            return int(screenshot.shape[1]), int(screenshot.shape[0])
        scroll = extract_scroll_state(observation)
        return max(1, int(scroll["viewport_width"])), max(1, int(scroll["viewport_height"]))

    def metadata(self) -> dict[str, Any]:
        config = asdict(self.config)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        digest.update((VISION_SYSTEM_PROMPT if self.config.use_screenshot else TEXT_SYSTEM_PROMPT).encode())
        legacy_dir = Path(__file__).resolve().parents[1] / "react_agent"
        source_paths = [Path(__file__), Path(__file__).with_name("vision.py")] + [legacy_dir / name for name in (
            "action_parser.py", "action_guard.py", "loop_guard.py", "memory.py",
            "prompt.py", "schemas.py", "spatial.py", "verifier.py",
        )]
        for source in source_paths:
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        return {
            "harness_config": config, "config_version": self.config.config_version,
            "harness_fingerprint": digest.hexdigest(),
            "observation_mode": "dom_axtree_screenshot" if self.config.use_screenshot else "text_dom_axtree",
            "response_loss_scope": "all_original_generated_tokens",
        }
