"""MemAgent-inspired structured ReAct agent for OpenApps."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import agentlab.agents.dynamic_prompting as dp
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage
from browsergym.experiments.agent import Agent, AgentInfo

from open_apps.agent.utils import retry
from open_apps.agent.vLLM_agent import AgentArgs as LegacyAgentArgs

from .action_guard import ActionGuard, ActionGuardConfig, SUPPORTED_ACTIONS, extract_valid_bids
from .action_parser import BrowserGymActionSerializer, ReactResponseParser
from .loop_guard import LoopGuard
from .memory import MemoryConfig, MemoryManager
from .prompt import (
    DEFAULT_SYSTEM_PROMPT,
    ReactPromptBuilder,
    extract_goal_text,
)
from .schemas import (
    ActionAssessment,
    ActionCall,
    AgentResponse,
    MemoryBlock,
    ReactDecision,
    SchemaError,
)
from .spatial import PageExplorationTracker, extract_scroll_state
from .trace_logger import TraceLogger
from .verifier import OutcomeVerifier, observation_signature


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AgentArgs(LegacyAgentArgs):
    """Hydra-compatible arguments for :class:`ReactMemoryAgent`."""

    memory_max_tokens: int = 1024
    memory_max_subgoals: int = 12
    memory_max_verified_facts: int = 20
    memory_max_failures: int = 3
    memory_max_uncertainties: int = 5
    coordinate_space: int = 1000
    max_same_action_retries: int = 2
    recent_turn_replay_size: int = 3
    offscreen_element_limit: int = 20
    allow_unfocused_keyboard_type: bool = False
    react_max_retry: int = 3
    react_system_prompt: str | None = None

    def make_agent(self) -> Agent:
        return ReactMemoryAgent(
            chat_model_args=self.make_chat_model_flags(),
            flags=self.make_flags(),
            custom_actions=self.custom_actions,
            memory_config=MemoryConfig(
                max_tokens=self.memory_max_tokens,
                max_subgoals=self.memory_max_subgoals,
                max_verified_facts=self.memory_max_verified_facts,
                max_failures=self.memory_max_failures,
                max_uncertainties=self.memory_max_uncertainties,
            ),
            coordinate_space=self.coordinate_space,
            max_same_action_retries=self.max_same_action_retries,
            recent_turn_replay_size=self.recent_turn_replay_size,
            offscreen_element_limit=self.offscreen_element_limit,
            allow_unfocused_keyboard_type=self.allow_unfocused_keyboard_type,
            max_retry=self.react_max_retry,
            system_prompt=self.react_system_prompt or DEFAULT_SYSTEM_PROMPT,
            save_dir=self.save_dir,
        )


class ReactMemoryAgent(Agent):
    def __init__(
        self,
        *,
        chat_model_args,
        flags,
        custom_actions: list[str],
        memory_config: MemoryConfig | None = None,
        coordinate_space: int = 1000,
        max_same_action_retries: int = 2,
        recent_turn_replay_size: int = 3,
        offscreen_element_limit: int = 20,
        allow_unfocused_keyboard_type: bool = False,
        max_retry: int = 3,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        save_dir: str | Path | None = None,
    ):
        self.chat_model_args = chat_model_args
        self.chat_llm = chat_model_args.make_model()
        self.flags = flags
        self.action_set = flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.max_retry = max_retry
        self.system_prompt = system_prompt
        if recent_turn_replay_size < 0:
            raise ValueError("recent_turn_replay_size must be non-negative.")
        if offscreen_element_limit < 0:
            raise ValueError("offscreen_element_limit must be non-negative.")
        self.recent_turn_replay_size = int(recent_turn_replay_size)

        configured_actions = set(custom_actions or SUPPORTED_ACTIONS)
        configured_actions.add("noop")
        allowed_actions = configured_actions & SUPPORTED_ACTIONS
        self.response_parser = ReactResponseParser()
        self.action_serializer = BrowserGymActionSerializer(coordinate_space)
        self.action_guard = ActionGuard(
            ActionGuardConfig(
                coordinate_space=coordinate_space,
                allow_unfocused_keyboard_type=allow_unfocused_keyboard_type,
                allowed_actions=allowed_actions,
            )
        )
        self.memory_manager = MemoryManager(memory_config)
        self.verifier = OutcomeVerifier()
        self.loop_guard = LoopGuard(repeat_limit=max_same_action_retries)
        self.exploration_tracker = PageExplorationTracker()
        self.prompt_builder = ReactPromptBuilder(
            client_type=chat_model_args.client_type,
            use_html=flags.obs.use_html,
            use_axtree=flags.obs.use_ax_tree,
            use_screenshot=flags.obs.use_screenshot,
            coordinate_space=coordinate_space,
            offscreen_element_limit=offscreen_element_limit,
        )
        self.trace_logger = TraceLogger(save_dir)
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def get_action(self, obs: dict[str, Any]) -> tuple[str, AgentInfo]:
        self.turn_index += 1
        if self.immutable_goal is None:
            self.immutable_goal = extract_goal_text(obs)

        automatic_assessment = self.verifier.assess(
            self.previous_obs, obs, self.previous_action
        )
        self.page_exploration = self.exploration_tracker.update(
            obs,
            previous_obs=self.previous_obs,
            previous_action=self.previous_action,
            assessment=automatic_assessment,
        )
        if self.previous_obs is not None and self.previous_action is not None:
            self._append_recent_turn(obs, automatic_assessment)
            self.loop_guard.record(
                self.previous_action,
                observation_signature(self.previous_obs),
                observation_signature(obs),
                automatic_assessment,
            )

        memory_before = self.memory
        prompt_result = self.prompt_builder.build(
            immutable_goal=self.immutable_goal,
            memory=memory_before,
            automatic_assessment=automatic_assessment,
            previous_action=self.previous_action,
            previous_expected_result=self.previous_expected_result,
            obs=obs,
            loop_context=self.loop_guard.prompt_context(),
            recent_turns=self.recent_turns,
            page_exploration=self.page_exploration,
        )
        chat_messages = Discussion(
            [SystemMessage(self.system_prompt), prompt_result.message]
        )

        screenshot = obs.get("screenshot")
        if screenshot is not None and hasattr(screenshot, "shape"):
            viewport = (int(screenshot.shape[1]), int(screenshot.shape[0]))
        else:
            viewport = (1000, 1000)

        parser_calls = 0
        last_raw_response = ""

        def parse_and_validate(raw_response: str) -> dict[str, Any]:
            nonlocal parser_calls, last_raw_response
            parser_calls += 1
            last_raw_response = raw_response
            try:
                response = self.response_parser.parse(raw_response)
                if (
                    response.previous_action_assessment.status
                    != automatic_assessment.status
                    or response.previous_action_assessment.page_changed
                    != automatic_assessment.page_changed
                ):
                    raise ParseError(
                        "previous_action_assessment must copy automatic_assessment.status "
                        "and automatic_assessment.page_changed exactly."
                    )
                response.previous_action_assessment = automatic_assessment
                self.action_guard.validate(response.action, obs)
                self.loop_guard.validate_candidate(response.action)
                response.memory_next = self.memory_manager.prepare_update(
                    memory_before,
                    response.memory_next,
                    automatic_assessment,
                    current_url=str(obs.get("url", "") or ""),
                    focused_bid=str(obs.get("focused_element_bid", "") or ""),
                    valid_bids=extract_valid_bids(obs),
                )
                browsergym_action = self.action_serializer.serialize(response.action, viewport)
            except ParseError:
                raise
            except (SchemaError, ValueError) as exc:
                raise ParseError(str(exc)) from exc
            return {
                "response": response,
                "action": browsergym_action,
                "displayed_action": response.action.to_dict(),
                "raw_response": raw_response,
            }

        parsed_response: AgentResponse | None = None
        error: str | None = None
        try:
            result = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=parse_and_validate,
            )
            parsed_response = result["response"]
            browsergym_action = result["action"]
            selected_action = parsed_response.action
            self.memory = parsed_response.memory_next
            self.previous_decision = parsed_response.decision
            self.previous_expected_result = parsed_response.decision.expected_result
            busted_retry = 0
        except (ParseError, SchemaError, ValueError) as exc:
            error = str(exc)
            logger.error("ReAct response failed validation: %s", error)
            selected_action = ActionCall(name="noop", wait_ms=1000)
            browsergym_action = self.action_serializer.serialize(selected_action, viewport)
            self.previous_decision = None
            self.previous_expected_result = "Wait briefly, then re-observe after invalid model output."
            busted_retry = 1

        self.previous_obs = obs
        self.previous_action = selected_action
        self.actions.append(browsergym_action)

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = max(0, parser_calls - 1)
        stats["busted_retry"] = busted_retry
        stats["memory_estimated_tokens"] = self.memory_manager.estimate_tokens(self.memory)
        stats["react_turn"] = self.turn_index

        self.trace_logger.log_turn(
            turn=self.turn_index,
            obs=obs,
            memory_before=memory_before,
            automatic_assessment=automatic_assessment,
            prompt_text=prompt_result.text,
            raw_response=last_raw_response,
            parsed_response=parsed_response,
            browsergym_action=browsergym_action,
            spatial_state={
                "scroll_state": prompt_result.scroll_state,
                "page_exploration": prompt_result.page_exploration,
                "offscreen_interactive_elements": (
                    prompt_result.offscreen_interactive_elements
                ),
            },
            error=error,
        )

        decision = parsed_response.decision if parsed_response else None
        agent_info = AgentInfo(
            think=decision.reason if decision else error,
            chat_messages=chat_messages,
            stats=stats,
            extra_info={
                "chat_model_args": self._safe_model_args(),
                "immutable_goal": self.immutable_goal,
                "memory": self.memory.to_dict(),
                "automatic_assessment": automatic_assessment.to_dict(),
                "decision": dataclasses.asdict(decision) if decision else None,
                "structured_action": selected_action.to_dict(),
                "recent_turns": list(self.recent_turns),
                "scroll_state": prompt_result.scroll_state,
                "page_exploration": prompt_result.page_exploration,
                "offscreen_interactive_elements": (
                    prompt_result.offscreen_interactive_elements
                ),
                "validation_error": error,
            },
        )
        return browsergym_action, agent_info

    def _append_recent_turn(
        self,
        current_obs: dict[str, Any],
        assessment: ActionAssessment,
    ) -> None:
        """Finalize the previous turn once its resulting observation exists."""
        if (
            self.recent_turn_replay_size == 0
            or self.previous_action is None
            or self.previous_obs is None
        ):
            return

        decision = self.previous_decision
        if decision is not None:
            subgoal = decision.current_subgoal
            reason = decision.reason
        else:
            subgoal = self.memory.current_subgoal_id or "unknown"
            reason = "The previous model response was invalid; BrowserGym executed a noop."

        self.recent_turns.append(
            {
                "turn": max(1, self.turn_index - 1),
                "url_before": str(self.previous_obs.get("url", "") or ""),
                "url_after": str(current_obs.get("url", "") or ""),
                "subgoal": subgoal,
                "reason": reason,
                "action": self.previous_action.to_dict(),
                "expected_result": self.previous_expected_result,
                "assessment": assessment.status,
                "assessment_evidence": assessment.evidence,
                "page_changed": assessment.page_changed,
                "scroll_before": extract_scroll_state(self.previous_obs),
                "scroll_after": extract_scroll_state(current_obs),
            }
        )
        self.recent_turns = self.recent_turns[-self.recent_turn_replay_size :]

    def _safe_model_args(self) -> dict[str, Any]:
        args = asdict(self.chat_model_args)
        for key in ("api_key", "aws_access_key", "aws_secret_key", "aws_session_token"):
            if args.get(key):
                args[key] = "***REDACTED***"
        return args

    def reset(self, seed=None):
        self.seed = seed
        self.turn_index = 0
        self.immutable_goal: str | None = None
        self.memory: MemoryBlock = self.memory_manager.initialize()
        self.previous_obs: dict[str, Any] | None = None
        self.previous_action: ActionCall | None = None
        self.previous_decision: ReactDecision | None = None
        self.previous_expected_result = ""
        self.recent_turns: list[dict[str, Any]] = []
        self.page_exploration: dict[str, Any] = {}
        self.actions: list[str] = []
        self.loop_guard.reset()
        self.exploration_tracker.reset()
